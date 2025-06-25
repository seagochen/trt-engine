// benchmark_api_impl.cpp
// Implementation of the C-compatible API for YoloPose and EfficientNet inference.

#include "trtengine/serverlet/models/inference/model_init_helper.hpp" // For ModelFactory and YoloPose
#include "trtengine/c_apis/c_pose_pipeline.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <any>
#include <map>
#include <chrono>
#include <memory> // For std::unique_ptr

#ifdef __cplusplus
extern "C" {
#endif

// Internal C++ context structure to hold model instances and parameters
struct YoloEfficientContextImpl {
    std::unique_ptr<InferModelBaseMulti> pose_model;
    std::unique_ptr<InferModelBaseMulti> efficient_model;
    std::map<std::string, std::any> yolo_params;
    std::map<std::string, std::any> efficient_params;
};

// Forward declaration for the helper function
static void free_batched_pose_results_single_image(C_ImagePoseResults* single_image_results);

// Helper function to convert std::vector<YoloPose> to C_YoloPose*
// This function is still responsible for C-style memory allocation, which the caller must free.
static C_YoloPose* convert_yolo_poses_to_c(const std::vector<YoloPose>& cpp_poses) {
    if (cpp_poses.empty()) {
        return nullptr;
    }

    // Use a unique_ptr to manage the initial allocation,
    // so if subsequent allocations fail, the main array is freed.
    auto* c_poses_raw = static_cast<C_YoloPose*>(malloc(cpp_poses.size() * sizeof(C_YoloPose)));
    if (!c_poses_raw) {
        std::cerr << "Memory allocation failed for C_YoloPose array." << std::endl;
        return nullptr;
    }
    // Custom deleter for unique_ptr to handle arrays allocated with malloc
    std::unique_ptr<C_YoloPose, decltype(&free)> c_poses(c_poses_raw, &free);

    for (size_t i = 0; i < cpp_poses.size(); ++i) {
        c_poses.get()[i].lx = cpp_poses[i].lx;
        c_poses.get()[i].ly = cpp_poses[i].ly;
        c_poses.get()[i].rx = cpp_poses[i].rx;
        c_poses.get()[i].ry = cpp_poses[i].ry;
        c_poses.get()[i].cls = static_cast<float>(cpp_poses[i].cls);
        c_poses.get()[i].num_pts = static_cast<int>(cpp_poses[i].pts.size());

        // Allocate memory for keypoints
        if (cpp_poses[i].pts.empty()) {
            c_poses.get()[i].pts = nullptr;
        } else {
            c_poses.get()[i].pts = static_cast<C_KeyPoint*>(malloc(cpp_poses[i].pts.size() * sizeof(C_KeyPoint)));
            if (!c_poses.get()[i].pts) {
                std::cerr << "Memory allocation failed for C_KeyPoint array for pose " << i << "." << std::endl;
                // unique_ptr will handle c_poses cleanup, but we also need to free any previously allocated pts.
                for (size_t j = 0; j < i; ++j) {
                    free(c_poses.get()[j].pts); // Free keypoints for poses already allocated
                }
                return nullptr; // c_poses unique_ptr will free c_poses_raw
            }
            for (size_t k = 0; k < cpp_poses[i].pts.size(); ++k) {
                c_poses.get()[i].pts[k].x = static_cast<float>(cpp_poses[i].pts[k].x);
                c_poses.get()[i].pts[k].y = static_cast<float>(cpp_poses[i].pts[k].y);
                c_poses.get()[i].pts[k].conf = 0.0f; // Default to 0 if not available from YoloPose
            }
        }

        // Allocate memory for features if they exist
        if (cpp_poses[i].feats.empty()) {
            c_poses.get()[i].feats = nullptr;
        } else {
            c_poses.get()[i].feats = static_cast<float*>(malloc(cpp_poses[i].feats.size() * sizeof(float)));
            if (!c_poses.get()[i].feats) {
                std::cerr << "Memory allocation failed for C_YoloPose feats array for pose " << i << "." << std::endl;
                // Cleanup previously allocated pts and feats
                for (size_t j = 0; j <= i; ++j) { // Include current pose's pts if allocated
                    free(c_poses.get()[j].pts);
                    free(c_poses.get()[j].feats);
                }
                return nullptr; // c_poses unique_ptr will free c_poses_raw
            }
            std::memcpy(c_poses.get()[i].feats, cpp_poses[i].feats.data(), cpp_poses[i].feats.size() * sizeof(float));
        }
    }
    return c_poses.release(); // Release ownership to the caller
}

// ------------------------------------------ C API Functions ------------------------------------------
void c_register_models() {
    ModelFactory::registerModel("YoloV8_Pose",
        [](const std::string& engine_path, const std::map<std::string, std::any>& params) {
            int maximum_batch = GET_PARAM(params, "maximum_batch", int);
            int maximum_items = GET_PARAM(params, "maximum_items", int);
            int infer_features = GET_PARAM(params, "infer_features", int);
            int infer_samples = GET_PARAM(params, "infer_samples", int);

            std::vector<TensorDefinition> output_tensor_defs = {{"output0",
                {maximum_batch, infer_features, infer_samples}}};

            auto pose_converter = [](const std::vector<float>& input, std::vector<YoloPose>& output, int features, int results) {
                cvtXYWHCoordsToYoloPose(input, output, features, results);
            };

            return std::make_unique<InferYoloV8<YoloPose, decltype(pose_converter)>>(
                engine_path, maximum_batch, maximum_items, infer_features, output_tensor_defs, pose_converter
            );
        });

    ModelFactory::registerModel("EfficientNet",
        [](const std::string& engine_path, const std::map<std::string, std::any>& params) {
            int maximum_batch = GET_PARAM(params, "maximum_batch", int);
            return std::make_unique<EfficientFeats>(engine_path, maximum_batch);
        });
}


YoloEfficientContext* c_create_pose_pipeline(
    const char* yolo_engine_path,
    const char* efficient_engine_path,
    int yolo_max_batch,
    int efficient_max_batch,
    float yolo_cls_thresh,
    float yolo_iou_thresh) {

    auto* context = new (std::nothrow) YoloEfficientContextImpl();
    if (!context) {
        std::cerr << "Failed to allocate YoloEfficientContextImpl." << std::endl;
        return nullptr;
    }

    context->yolo_params = {
        {"maximum_batch", yolo_max_batch},
        {"maximum_items", 100},
        {"infer_features", 56},
        {"infer_samples", 8400},
        {"cls", yolo_cls_thresh},
        {"iou", yolo_iou_thresh}
    };

    context->efficient_params = {
        {"maximum_batch", efficient_max_batch}
    };

    context->pose_model = ModelFactory::createModel("YoloV8_Pose", yolo_engine_path, context->yolo_params);
    if (!context->pose_model) {
        std::cerr << "Failed to create YoloV8_Pose model." << std::endl;
        delete context;
        return nullptr;
    }

    context->efficient_model = ModelFactory::createModel("EfficientNet", efficient_engine_path, context->efficient_params);
    if (!context->efficient_model) {
        std::cerr << "Failed to create EfficientNet model." << std::endl;
        delete context;
        return nullptr;
    }

    return reinterpret_cast<YoloEfficientContext*>(context);
}

// Internal helper for pose and crop data
struct FlattenedPoseData {
    int original_image_idx_in_batch; // Original index in the input_images_data array
    size_t original_pose_idx_in_image; // Index of pose within its image's detections
    YoloPose pose;
    cv::Mat crop_image; // The actual cropped image data
};

// Helper function to preprocess input images for YoloPose
static std::pair<std::vector<cv::Mat>, std::map<int, int>>
preprocess_images_for_yolo(const unsigned char* const* input_images_data,
                           const int* widths,
                           const int* heights,
                           const int* channels,
                           int num_images) {

    std::vector<cv::Mat> valid_original_images_for_cropping;
    std::map<int, int> original_to_processed_idx_map;
    int valid_count = 0;

    for (int i = 0; i < num_images; ++i) {
        if (!input_images_data[i] || widths[i] <= 0 || heights[i] <= 0 || channels[i] <= 0) {
            std::cerr << "Invalid image data provided for image index " << i << ". Skipping." << std::endl;
            continue;
        }

        cv::Mat original_img;
        if (channels[i] == 3) {
            original_img = cv::Mat(heights[i], widths[i], CV_8UC3, (void*)input_images_data[i]);
        } else if (channels[i] == 1) {
            original_img = cv::Mat(heights[i], widths[i], CV_8UC1, (void*)input_images_data[i]);
        } else {
            std::cerr << "Unsupported number of channels: " << channels[i] << " for image index " << i << ". Skipping." << std::endl;
            continue;
        }

        original_img = original_img.clone(); // Ensure data ownership
        valid_original_images_for_cropping.push_back(original_img);
        original_to_processed_idx_map[i] = valid_count++;
    }

    return {valid_original_images_for_cropping, original_to_processed_idx_map};
}

// Helper function to perform YoloPose inference
static std::map<int, std::vector<YoloPose>>
perform_yolo_inference(YoloEfficientContextImpl* context,
                       const std::vector<cv::Mat>& original_images,
                       const std::map<int, int>& original_to_processed_idx_map) {

    std::vector<cv::Mat> yolo_input_images;
    yolo_input_images.reserve(original_images.size());

    // Create resized images for YoloPose
    for (const auto& img : original_images) {
        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(640, 640)); // YoloPose input size
        yolo_input_images.push_back(resized_img);
    }

    size_t yolo_batch_size = std::min(static_cast<size_t>(std::any_cast<int>(context->yolo_params["maximum_batch"])), yolo_input_images.size());

    for (int i = 0; i < yolo_batch_size; ++i) {
        context->pose_model->preprocess(yolo_input_images[i], i);
    }
    context->pose_model->inference();

    std::map<int, std::vector<YoloPose>> cpp_batched_pose_detections;
    for (int i = 0; i < yolo_batch_size; ++i) {
        std::any single_image_pose_results_any;
        context->pose_model->postprocess(i, context->yolo_params, single_image_pose_results_any);
        try {
            auto single_image_detections = std::any_cast<std::vector<YoloPose>>(single_image_pose_results_any);
            cpp_batched_pose_detections[i] = single_image_detections;
        } catch (const std::bad_any_cast& e) {
            std::cerr << "YoloPose API: Error casting single image postprocess results for batch index " << i << ": " << e.what() << std::endl;
            cpp_batched_pose_detections[i] = {};
        }
    }
    return cpp_batched_pose_detections;
}

// Helper function to crop images based on YoloPose detections for EfficientNet
static std::vector<FlattenedPoseData>
crop_images_for_efficientnet(const std::map<int, std::vector<YoloPose>>& yolo_detections,
                             const std::vector<cv::Mat>& original_images,
                             const std::map<int, int>& processed_to_original_idx_map) {

    std::vector<FlattenedPoseData> all_flattened_poses_with_crops;
    constexpr float scale_factor = 1.2f;

    for (auto const& [processed_img_idx, poses_in_image] : yolo_detections) {
        if (!processed_to_original_idx_map.contains(processed_img_idx)) {
            std::cerr << "Error: Processed image index " << processed_img_idx << " not found in original_to_processed_idx_map." << std::endl;
            continue;
        }
        int original_img_idx_from_input = processed_to_original_idx_map.at(processed_img_idx);

        // Ensure processed_img_idx is valid for original_images
        if (processed_img_idx >= original_images.size()) {
            std::cerr << "Error: Processed image index " << processed_img_idx << " out of bounds for original_images." << std::endl;
            continue;
        }
        const cv::Mat& source_image = original_images[processed_img_idx];

        if (source_image.empty()) continue; // Skip if source image invalid

        for (size_t i = 0; i < poses_in_image.size(); ++i) {
            const auto& pose = poses_in_image[i];
            if (pose.pts.empty()) continue; // Skip if no keypoints

            auto min_x = static_cast<float>(std::min(pose.lx, pose.rx));
            auto min_y = static_cast<float>(std::min(pose.ly, pose.ry));
            auto max_x = static_cast<float>(std::max(pose.lx, pose.rx));
            auto max_y = static_cast<float>(std::max(pose.ly, pose.ry));
            auto width = max_x - min_x;
            auto height = max_y - min_y;

            int crop_x = std::max(0, static_cast<int>(min_x - width * (scale_factor - 1) / 2));
            int crop_y = std::max(0, static_cast<int>(min_y - height * (scale_factor - 1) / 2));
            int crop_width = std::min(source_image.cols - crop_x, static_cast<int>(width * scale_factor));
            int crop_height = std::min(source_image.rows - crop_y, static_cast<int>(height * scale_factor));

            if (crop_width > 0 && crop_height > 0) {
                cv::Mat cropped_img = source_image(cv::Rect(crop_x, crop_y, crop_width, crop_height));
                all_flattened_poses_with_crops.push_back({original_img_idx_from_input, i, pose, cropped_img});
            }
        }
    }
    return all_flattened_poses_with_crops;
}

// Helper function to perform EfficientNet inference on cropped images
static std::vector<std::vector<float>>
perform_efficientnet_inference(YoloEfficientContextImpl* context,
                               const std::vector<FlattenedPoseData>& flattened_poses_with_crops) {

    std::vector<std::vector<float>> efficient_net_all_results;
    size_t efficient_max_batch = std::any_cast<int>(context->efficient_params["maximum_batch"]);

    for (size_t batch_start_idx = 0; batch_start_idx < flattened_poses_with_crops.size(); batch_start_idx += efficient_max_batch) {
        size_t batch_end_idx = std::min(batch_start_idx + efficient_max_batch, flattened_poses_with_crops.size());

        std::vector<cv::Mat> current_efficient_batch_images;
        current_efficient_batch_images.reserve(efficient_max_batch);

        for (size_t k = batch_start_idx; k < batch_end_idx; ++k) {
            current_efficient_batch_images.push_back(flattened_poses_with_crops[k].crop_image);
        }

        if (current_efficient_batch_images.empty()) {
            continue;
        }

        for (int i = 0; i < current_efficient_batch_images.size(); ++i) {
            context->efficient_model->preprocess(current_efficient_batch_images[i], i);
        }
        context->efficient_model->inference();

        std::vector<std::vector<float>> cls_results_sub_batch;
        for(int i = 0; i < current_efficient_batch_images.size(); ++i) {
            std::any single_efficient_result_any;
            context->efficient_model->postprocess(i, context->efficient_params, single_efficient_result_any);
            try {
                auto cls_result = std::any_cast<std::vector<float>>(single_efficient_result_any);
                cls_results_sub_batch.push_back(cls_result);
            } catch (const std::bad_any_cast& e) {
                std::cerr << "EfficientNet API: Error casting single crop result for batch index " << i << ": " << e.what() << std::endl;
                cls_results_sub_batch.emplace_back();
            }
        }
        efficient_net_all_results.insert(efficient_net_all_results.end(), cls_results_sub_batch.begin(), cls_results_sub_batch.end());
    }
    return efficient_net_all_results;
}

// Helper function to merge EfficientNet results back into YoloPose detections
static std::map<int, std::vector<YoloPose>>
merge_efficientnet_results(int num_original_images,
                           const std::map<int, std::vector<YoloPose>>& initial_yolo_detections,
                           const std::vector<FlattenedPoseData>& flattened_poses_with_crops,
                           const std::vector<std::vector<float>>& efficient_net_results) {

    std::map<int, std::vector<YoloPose>> final_cpp_detections_map;

    // Initialize with all original image indices, potentially empty or with initial YoloPose detections
    for(int i = 0; i < num_original_images; ++i) {
        if (initial_yolo_detections.count(i)) {
            final_cpp_detections_map[i] = initial_yolo_detections.at(i);
        } else {
            final_cpp_detections_map[i] = {}; // No YoloPose detections for this image
        }
    }

    // Iterate through flattened poses and update with EfficientNet results
    for (size_t i = 0; i < flattened_poses_with_crops.size(); ++i) {
        const auto& flat_pose_with_crop = flattened_poses_with_crops[i];
        int original_image_idx = flat_pose_with_crop.original_image_idx_in_batch;
        size_t original_pose_idx = flat_pose_with_crop.original_pose_idx_in_image;

        if (i < efficient_net_results.size() && efficient_net_results[i].size() >= 257) {
            // Update class and features
            if (final_cpp_detections_map.count(original_image_idx) &&
                original_pose_idx < final_cpp_detections_map.at(original_image_idx).size()) {

                YoloPose& updated_pose = final_cpp_detections_map.at(original_image_idx)[original_pose_idx];
                updated_pose.cls = static_cast<int>(efficient_net_results[i][0]);
                updated_pose.feats.assign(efficient_net_results[i].begin() + 1, efficient_net_results[i].end());
            }
        }
    }
    return final_cpp_detections_map;
}


C_BatchedPoseResults c_process_batched_images(
    YoloEfficientContext* context_handle,
    const unsigned char* const* input_images_data,
    const int* widths,
    const int* heights,
    const int* channels,
    int num_images) {

    C_BatchedPoseResults c_results = {0, nullptr};

    if (!context_handle) {
        std::cerr << "Invalid YoloEfficientContext provided." << std::endl;
        return c_results;
    }
    auto context = reinterpret_cast<YoloEfficientContextImpl*>(context_handle);

    if (num_images <= 0) {
        return c_results;
    }

    // 1. Preprocess input images for YoloPose
    auto [valid_original_images_for_cropping, original_to_processed_idx_map] =
        preprocess_images_for_yolo(input_images_data, widths, heights, channels, num_images);

    if (valid_original_images_for_cropping.empty()) {
        std::cerr << "No valid images to process in the batch." << std::endl;
        return c_results;
    }

    // Create a map from processed index back to original index
    std::map<int, int> processed_to_original_idx_map;
    for (auto const& [orig_idx, proc_idx] : original_to_processed_idx_map) {
        processed_to_original_idx_map[proc_idx] = orig_idx;
    }

    // 2. Perform YoloPose inference
    std::map<int, std::vector<YoloPose>> cpp_batched_pose_detections =
        perform_yolo_inference(context, valid_original_images_for_cropping, original_to_processed_idx_map);

    // 3. Flatten detections and crop images for EfficientNet
    std::vector<FlattenedPoseData> all_flattened_poses_with_crops =
        crop_images_for_efficientnet(cpp_batched_pose_detections, valid_original_images_for_cropping, processed_to_original_idx_map);

    // 4. Perform EfficientNet inference on cropped images
    std::vector<std::vector<float>> efficient_net_all_results =
        perform_efficientnet_inference(context, all_flattened_poses_with_crops);

    // 5. Merge EfficientNet results back into pose detections
    std::map<int, std::vector<YoloPose>> final_cpp_detections_map =
        merge_efficientnet_results(num_images, cpp_batched_pose_detections,
                                   all_flattened_poses_with_crops, efficient_net_all_results);

    // 6. Convert C++ results to C-compatible structure
    c_results.num_images = num_images;
    c_results.results = static_cast<C_ImagePoseResults*>(malloc(num_images * sizeof(C_ImagePoseResults)));
    if (!c_results.results) {
        std::cerr << "Memory allocation failed for C_ImagePoseResults array." << std::endl;
        c_results.num_images = 0;
        return c_results;
    }

    for (int i = 0; i < num_images; ++i) {
        c_results.results[i].image_idx = i;
        if (final_cpp_detections_map.count(i)) {
            const auto& cpp_poses = final_cpp_detections_map[i];
            c_results.results[i].num_detections = static_cast<int>(cpp_poses.size());
            c_results.results[i].detections = convert_yolo_poses_to_c(cpp_poses);
            if (c_results.results[i].detections == nullptr && !cpp_poses.empty()) {
                // If conversion of C_YoloPose* failed, clean up previous C_ImagePoseResults
                for (int j = 0; j < i; ++j) {
                    free_batched_pose_results_single_image(&c_results.results[j]);
                }
                free(c_results.results);
                c_results.results = nullptr;
                c_results.num_images = 0;
                std::cerr << "Partial conversion to C_YoloPose failed, returning empty results." << std::endl;
                return c_results;
            }
        } else {
            c_results.results[i].num_detections = 0;
            c_results.results[i].detections = nullptr;
        }
    }
    return c_results;
}

// Helper to free a single C_ImagePoseResults structure's internal memory
static void free_batched_pose_results_single_image(C_ImagePoseResults* single_image_results) {
    if (single_image_results && single_image_results->detections) {
        for (int i = 0; i < single_image_results->num_detections; ++i) {
            if (single_image_results->detections[i].pts) {
                free(single_image_results->detections[i].pts);
                single_image_results->detections[i].pts = nullptr;
            }
            if (single_image_results->detections[i].feats) {
                free(single_image_results->detections[i].feats);
                single_image_results->detections[i].feats = nullptr;
            }
        }
        free(single_image_results->detections);
        single_image_results->detections = nullptr;
    }
}

void c_free_batched_pose_results(C_BatchedPoseResults* results) {
    if (!results || !results->results) return;

    for (int i = 0; i < results->num_images; ++i) {
        free_batched_pose_results_single_image(&results->results[i]);
    }

    free(results->results);
    results->results = nullptr;
    results->num_images = 0;
}

void c_destroy_pose_pipeline(YoloEfficientContext* context_handle) {
    if (context_handle) {
        auto* context = reinterpret_cast<YoloEfficientContextImpl*>(context_handle);
        delete context;
    }
}

#ifdef __cplusplus
}
#endif