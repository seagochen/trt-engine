#include "trtengine/c_apis/c_pose_detection.h"
#include "trtengine/c_apis/aux_batch_process.h" // Includes InferenceResult, run_pose_detection_stage, run_efficientnet_stage
#include "trtengine/serverlet/models/inference/model_init_helper.hpp" // For ModelFactory
#include "trtengine/utils/logger.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <any>
#include <cstdlib> // For malloc, free
#include <cstring> // For memcpy (if used for structs)

// --- Global Model Pointers ---
// 使用全局智能指针管理模型生命周期
static std::unique_ptr<InferModelBaseMulti> g_pose_model = nullptr;
static std::unique_ptr<InferModelBaseMulti> g_efficient_model = nullptr;

// C++ 内部参数，用于 postprocess 调用
static std::map<std::string, std::any> g_pose_pp_params;
static std::map<std::string, std::any> g_efficient_pp_params;

// 全局用的queue，来存储待处理的图片数据
static std::vector<cv::Mat> g_image_queue;

// --- Helper for safe parameter retrieval ---
template<typename T>
T get_param_safe(const std::map<std::string, std::any>& params, const std::string& key, const T& default_value) {
    auto it = params.find(key);
    if (it != params.end()) {
        try {
            return std::any_cast<T>(it->second);
        } catch (const std::bad_any_cast& e) {
            LOG_WARNING("C_API", "Parameter '" + key + "' type mismatch. Expected type: " + typeid(T).name() + ", got: " + it->second.type().name() + ". Using default value. Error: " + e.what());
        } catch (const std::exception& e) {
            LOG_WARNING("C_API", "Error accessing parameter '" + key + "': " + e.what() + ". Using default value.");
        }
    } else {
        LOG_WARNING("C_API", "Parameter '" + key + "' not found. Using default value.");
    }
    return default_value;
}


// --- C Interface Implementations ---
// This function registers models with the ModelFactory.
// It's good to call this once at library initialization (e.g., first call to init_pose_detection_pipeline).
// Since it's static in ModelFactory, it will only register once.
void register_detection_models()
{
    // Register YOLOv8 Pose model
    ModelFactory::registerModel("YoloV8_Pose", [](const std::string& engine_path, const std::map<std::string, std::any>& params) {
        int maximum_batch = get_param_safe<int>(params, "maximum_batch", 1);
        int maximum_items = get_param_safe<int>(params, "maximum_items", 100);
        int infer_features = get_param_safe<int>(params, "infer_features", 56);
        int infer_samples = get_param_safe<int>(params, "infer_samples", 8400);

        std::vector<TensorDefinition> output_tensor_defs = {{"output0", {maximum_batch, infer_features, infer_samples}}};

        // YOLOv8 Pose converter function
        auto pose_converter = [](const std::vector<float>& input, std::vector<YoloPose>& output, int features, int results) {
            cvtXYWHCoordsToYoloPose(input, output, features, results);
        };

        return std::make_unique<InferYoloV8<YoloPose, decltype(pose_converter)>>(
            engine_path, maximum_batch, maximum_items, infer_features, output_tensor_defs, pose_converter
        );
    });

    // Register EfficientNet model
    ModelFactory::registerModel("EfficientNet", [](const std::string& engine_path, const std::map<std::string, std::any>& params) {
        int maximum_batch = GET_PARAM(params, "maximum_batch", int);
        return std::make_unique<EfficientFeats>(engine_path, maximum_batch);
    });
}


bool init_pose_detection_pipeline(const char* yolo_engine_path, const char* efficient_engine_path, int max_items, float cls, float iou)
{
    // Ensure ModelFactory is initialized/models registered
    static bool models_registered = false;
    if (!models_registered) {
        register_detection_models();
        models_registered = true;
    }

    // Set YoloPose model creation and post-processing parameters
    g_pose_pp_params["maximum_batch"] = 8;
    g_pose_pp_params["maximum_items"] = max_items;
    g_pose_pp_params["infer_features"] = 56;
    g_pose_pp_params["infer_samples"] = 8400;
    g_pose_pp_params["cls"] = cls;
    g_pose_pp_params["iou"] = iou;

    // Set EfficientNet model creation and post-processing parameters
    g_efficient_pp_params["maximum_batch"] = 32;

    // Create YOLOv8 Pose Estimation model
    g_pose_model = ModelFactory::createModel(
        "YoloV8_Pose", yolo_engine_path, g_pose_pp_params
    );
    if (!g_pose_model) {
        LOG_ERROR("C_API", "Create YoloV8 Pose model failed.");
        return false;
    }

    // Create EfficientNet model
    g_efficient_model = ModelFactory::createModel(
        "EfficientNet", efficient_engine_path, g_efficient_pp_params
    );
    if (!g_efficient_model) {
        LOG_ERROR("C_API", "Create EfficientNet model failed.");
        g_pose_model.reset(); // Clean up if EfficientNet fails
        return false;
    }

    LOG_INFO("C_API", "Pose detection pipeline initialized successfully.");
    return true;
}


void add_image_to_pose_detection_pipeline(const unsigned char* image_data_in_bgr, int width, int height)
{
    if (!g_pose_model) {
        LOG_ERROR("C_API", "Pose model not initialized. Call init_pose_detection_pipeline first.");
        return;
    }
    if (!image_data_in_bgr) {
        LOG_ERROR("C_API", "Invalid image data pointer in add_image_to_pose_detection_pipeline.");
        return;
    }
    if (width <= 0 || height <= 0) {
        LOG_ERROR("C_API", "Invalid image dimensions (width=" + std::to_string(width) + ", height=" + std::to_string(height) + ") in add_image_to_pose_detection_pipeline.");
        return;
    }

    // Convert raw image data to cv::Mat
    // Make sure to create a deep copy if original_image might go out of scope or change
    cv::Mat original_image(height, width, CV_8UC3, (void*)image_data_in_bgr, cv::Mat::AUTO_STEP);

    // 由于 efficientnet 需要基于 yolo 的结果进行二次分析, 所以最好是将图片统一处理为 640x640 的大小
    if (width != 640 || height != 640) {
        cv::Mat resized_image;
        cv::resize(original_image, resized_image, cv::Size(640, 640));
        original_image = resized_image; // Update original_image to the resized version
    }

    // Store the image in the global queue
    g_image_queue.push_back(original_image); // Clone to ensure the data is owned by the queue
    LOG_VERBOSE_TOPIC("C_API", "ImageQueue", "Image added to queue. Current queue size: " + std::to_string(g_image_queue.size()));
}


bool run_pose_detection_pipeline(C_Inference_Result** out_results, int *out_num_results)
{
    // Initialize output pointers to null/zero in case of early return
    *out_results = nullptr;
    *out_num_results = 0;

    if (!g_pose_model || !g_efficient_model) {
        LOG_ERROR("C_API", "Models not initialized. Call init_pose_detection_pipeline first.");
        return false;
    }

    // If g_image_queue is empty, nothing to process
    if (g_image_queue.empty()) {
        LOG_WARNING("C_API", "No images in the queue to process.");
        // This is not an error, just no data. Return success with 0 results.
        return true;
    }

    // --- Stage 1: Run YOLOv8 Pose Detection Batch Processing ---
    // `g_image_queue` will be consumed by run_pose_detection_stage
    std::vector<InferenceResult> pose_stage_results = run_pose_detection_stage(
        g_image_queue, // This vector will be emptied by the call
        g_pose_model,
        g_pose_pp_params
    );

    if (g_image_queue.empty()) { // Verify images were consumed (optional check)
        LOG_VERBOSE_TOPIC("C_API", "RunPipeline", "Image queue consumed after pose_extend stage.");
    } else {
        LOG_WARNING("C_API", "Image queue not fully consumed by pose_extend stage. Remaining: " + std::to_string(g_image_queue.size()));
    }


    // --- Stage 2: Run EfficientNet Classification/Feature Extraction ---
    // `pose_stage_results` is passed by const reference, so it's not modified.
    std::vector<InferenceResult> final_cpp_results = run_efficientnet_stage(
        pose_stage_results,
        g_efficient_model,
        g_efficient_pp_params
    );

    // --- Convert C++ results (std::vector<InferenceResult>) to C-compatible output (C_Inference_Result**) ---
    *out_num_results = static_cast<int>(final_cpp_results.size());

    // Allocate memory for the array of C_Inference_Result pointers
    *out_results = (C_Inference_Result*)malloc(sizeof(C_Inference_Result) * (*out_num_results));
    if (!*out_results) {
        LOG_ERROR("C_API", "Failed to allocate memory for C_Inference_Result array.");
        *out_num_results = 0;
        return false;
    }

    // Populate each C_Inference_Result
    for (int i = 0; i < *out_num_results; ++i) {
        const InferenceResult& cpp_result = final_cpp_results[i];
        C_Inference_Result& c_result = (*out_results)[i];

        c_result.num_detected = cpp_result.num_detected; // Direct copy of detection count

        if (cpp_result.num_detected > 0 && !cpp_result.detections.empty()) {
            // Allocate memory for the detections array within this C_Inference_Result
            c_result.detections = (C_Extended_Person_Feats*)malloc(sizeof(C_Extended_Person_Feats) * cpp_result.detections.size());
            if (!c_result.detections) {
                LOG_ERROR("C_API", "Failed to allocate memory for detections in C_Inference_Result " + std::to_string(i));
                // Clean up previously allocated detections and the main array
                for (int j = 0; j < i; ++j) {
                    free((*out_results)[j].detections);
                }
                free(*out_results);
                *out_results = nullptr;
                *out_num_results = 0;
                return false; // Critical memory allocation failure
            }
            // Copy the actual C_Extended_Person_Feats data
            // Use memcpy for raw bytes copy, or loop for element-wise copy if C++ objects inside (not here)
            std::memcpy(c_result.detections, cpp_result.detections.data(), sizeof(C_Extended_Person_Feats) * cpp_result.detections.size());
            LOG_VERBOSE_TOPIC("C_API", "RunPipeline", "Copied " + std::to_string(cpp_result.detections.size()) + " detections for image " + std::to_string(i));
        } else {
            // No detections or error, set detections pointer to null
            c_result.detections = nullptr;
            LOG_VERBOSE_TOPIC("C_API", "RunPipeline", "No detections for image " + std::to_string(i) + ", setting detections to nullptr.");
        }
    }

    LOG_INFO("C_API", "Pipeline executed successfully and results converted to C structs.");
    return true;
}

void deinit_pose_detection_pipeline() {
    g_pose_model.reset();
    g_efficient_model.reset();
    g_image_queue.clear(); // Clear any remaining images in the queue
    LOG_INFO("C_API", "Pose detection pipeline models and queue deinitialized.");
}

// Corrected memory release function for C_Inference_Result output
void release_inference_result(C_Inference_Result* result_array, int count) {
    if (result_array) {
        for (int i = 0; i < count; ++i) {
            if (result_array[i].detections) {
                free(result_array[i].detections); // Free detections array for each C_Inference_Result
                result_array[i].detections = nullptr;
            }
        }
        free(result_array); // Free the main C_Inference_Result array
        LOG_INFO("C_API", "Released " + std::to_string(count) + " C_Inference_Result objects and their nested detections.");
    } else {
        LOG_WARNING("C_API", "Attempted to release a null C_Inference_Result array.");
    }
}