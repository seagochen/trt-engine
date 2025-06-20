#include "trtengine/c_apis/c_pose_detection.h"
#include "trtengine/c_apis/aux_batch_process.h"
#include "trtengine/serverlet/models/inference/model_init_helper.hpp"
#include "trtengine/utils/logger.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <any>

// --- Global Model Pointers ---
// 使用全局智能指针管理模型生命周期
static std::unique_ptr<InferModelBaseMulti> g_pose_model = nullptr;
static std::unique_ptr<InferModelBaseMulti> g_efficient_model = nullptr;

// C++ 内部参数，用于 postprocess 调用
static std::map<std::string, std::any> g_pose_pp_params;
static std::map<std::string, std::any> g_efficient_pp_params;

// 批次数据
int g_current_batch_idx = 0;

// 全局用的queue，来存储待处理的图片数据
static std::vector<cv::Mat> g_image_queue;


// --- Helper for safe parameter retrieval (copied from previous suggestions) ---
template<typename T>
T get_param_safe(const std::map<std::string, std::any>& params, const std::string& key, const T& default_value) {
    auto it = params.find(key);
    if (it != params.end()) {
        if (const T* val_ptr = std::any_cast<T>(&(it->second))) {
            return *val_ptr;
        } else {
            LOG_WARNING("C_API", "Parameter '" + key + "' type mismatch. Expected type: " + typeid(T).name() + ". Using default value.");
        }
    } else {
        LOG_WARNING("C_API", "Parameter '" + key + "' not found. Using default value.");
    }
    return default_value;
}


// --- C Interface Implementations ---
void register_detection_models()
{
    // 注册 YOLOv8 姿态估计模型
    ModelFactory::registerModel("YoloV8_Pose", [](const std::string& engine_path, const std::map<std::string, std::any>& params) {

        // 从参数中获取必要的配置
        int maximum_batch = GET_PARAM(params, "maximum_batch", int);
        int maximum_items = GET_PARAM(params, "maximum_items", int);
        int infer_features = GET_PARAM(params, "infer_features", int);
        int infer_samples = GET_PARAM(params, "infer_samples", int);

        std::vector<TensorDefinition> output_tensor_defs = std::vector<TensorDefinition>{{"output0", {maximum_batch, infer_features, infer_samples}}};

        // YOLOv8 姿态估计的转换函数
        auto pose_converter = [](const std::vector<float>& input, std::vector<YoloPose>& output, int features, int results) {
            cvtXYWHCoordsToYoloPose(input, output, features, results);
        };

        return std::make_unique<InferYoloV8<YoloPose, decltype(pose_converter)>>(
            engine_path, maximum_batch, maximum_items, infer_features, output_tensor_defs, pose_converter
        );
    });

    // 注册 EfficientNet 模型
    ModelFactory::registerModel("EfficientNet", [](const std::string& engine_path, const std::map<std::string, std::any>& params) {
        int maximum_batch = GET_PARAM(params, "maximum_batch", int);
        return std::make_unique<EfficientFeats>(engine_path, maximum_batch);
    });
}


bool init_pose_detection_pipeline(const char* yolo_engine_path, const char* efficient_engine_path, int max_items, float cls, float iou)
{
    // 确保 ModelFactory 已初始化/模型已注册
    register_detection_models();

    // 为 YoloPose 模型创建参数
    g_pose_pp_params["maximum_batch"] = 8;          // TODO: 硬编码为8， 后续从config中读取
    g_pose_pp_params["maximum_items"] = max_items;
    g_pose_pp_params["infer_features"] = 56;        // TODO: 硬编码为56， 后续从config中读取
    g_pose_pp_params["infer_samples"] = 8400;       // TODO: 硬编码为8400， 后续从config中读取
    g_pose_pp_params["cls"] = cls;
    g_pose_pp_params["iou"] = iou;

    // EfficientNet 后处理不需要自定义参数，仅设置最大 batch
    g_efficient_pp_params["maximum_batch"] = 32;    // TODO: 硬编码为32， 后续从config中读取

    // 创建 YOLOv8 姿态估计模型
    g_pose_model = ModelFactory::createModel(
        "YoloV8_Pose", yolo_engine_path, g_pose_pp_params
    );
    if (!g_pose_model) {
        LOG_ERROR("C_API", "Create YoloV8 Pose model failed.");
        return false;
    }

    // 创建 EfficientNet 模型
    g_efficient_model = ModelFactory::createModel(
        "EfficientNet", efficient_engine_path, g_efficient_pp_params
    );
    if (!g_efficient_model) {
        LOG_ERROR("C_API", "Create EfficientNet model failed.");
        g_pose_model.reset(); // 如果 EfficientNet 创建失败则清理
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

    // 1. Convert raw image data to cv::Mat
    cv::Mat original_image(height, width, CV_8UC3, (void*)image_data_in_bgr);
    if (original_image.empty()) {
        LOG_ERROR("C_API", "Failed to create cv::Mat from input image data.");
        return;
    }

    // 2. Store the preprocessed image in the global queue
    g_image_queue.push_back(original_image.clone()); // Clone to ensure the data is owned by the queue
}


bool run_pose_detection_pipeline(void **out_results, int *out_num_results)
{
    if (!g_pose_model || !g_efficient_model) {
        LOG_ERROR("C_API", "Models not initialized. Call init_pose_detection_pipeline first.");
        return false;
    }

    // 确认 g_image_queue 不为空
    if (g_image_queue.empty()) {
        LOG_WARNING("C_API", "No images in the queue to process.");
        return true; // 没有图片处理，但也算成功完成
    }

    // 启动管道，执行yolo姿态检测
    auto yolo_processed_batch_results = run_pose_detection_stage(g_image_queue, g_pose_model, g_pose_pp_params);

    // TODO 首先，对所有的照片都执行 process_batch_images_by_pose_engine 获取初步的检测结果
    // 注意 g_image_queue 中的图片在这里会被消耗掉，因此执行克隆任务

    // TODO 然后，对每张图片单独使用 process_batch_images_by_efficient_engine 以获取完整的检测结果
    //

    return true;
}

void deinit_pose_detection_pipeline() {
    g_pose_model.reset();
    g_efficient_model.reset();
    LOG_INFO("C_API", "Pose detection pipeline models deinitialized.");
}


void release_inference_result(void* result) {
    if (result) {

        // 将 void* 转换为 C_InferenceResult*
        C_InferenceResult* inference_result = static_cast<C_InferenceResult*>(result);

        // 释放 detections 数组
        if (inference_result->detections) {
            delete[] inference_result->detections; // 释放 detections 数组
            inference_result->detections = nullptr; // 避免悬空指针
        }

        // 释放 C_InferenceResult 结构体本身
        delete inference_result; // 释放 C_InferenceResult 结构体
        inference_result = nullptr; // 避免悬空指针
    }
}