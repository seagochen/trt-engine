//
// Created by xtcjj on 2025/08/14.
//

#include "trtengine/servlet/models/inference/model_init_helper.hpp" // For ModelFactory and YoloPose
#include "trtengine/c_apis/c_common.h"
#include "trtengine/c_apis/c_yolopose_detection.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <any>
#include <map>
#include <chrono>
#include <memory> // For std::unique_ptr

// 原子操作和 OpenMP 头文件
#include <atomic>
#include <omp.h> // 目前暂时关闭这部分的优化


#ifdef __cplusplus
extern "C" {
#endif


// 内部 C++ 上下文结构，用于保存模型实例和参数
struct YoloPoseContextImpl {
    std::unique_ptr<InferModelBaseMulti> pose_model;
    std::map<std::string, std::any> yolo_params;
};


void c_register_yolopose_model() {

    // 注册 YoloV8_Pose 模型
    ModelFactory::registerModel("YoloV8_Pose",
        [](const std::string& engine_path, const std::map<std::string, std::any>& params) {

            // 从参数中获取配置
            int maximum_batch = GET_PARAM(params, "maximum_batch", int);
            int maximum_items = GET_PARAM(params, "maximum_items", int);
            int infer_features = GET_PARAM(params, "infer_features", int);
            int infer_samples = GET_PARAM(params, "infer_samples", int);

            // 定义输出张量
            std::vector<TensorDefinition> output_tensor_defs = {{"output0",
                {maximum_batch, infer_features, infer_samples}}};

            // 定义转换器
            auto pose_converter = [](const std::vector<float>& input, std::vector<YoloPose>& output, int features, int results) {
                cvtXYWHCoordsToYoloPose(input, output, features, results);
            };

            // 定义推理器
            return std::make_unique<InferYoloV8<YoloPose, decltype(pose_converter)>>(
                engine_path, maximum_batch, maximum_items, infer_features, output_tensor_defs, pose_converter
            );
        });
}


 void* c_create_yolopose_context(const char* model_path, int yolo_max_batch, float yolo_cls_threshold, float yolo_iou_threshold) {

    // 创建YoloPose的TrtContext
    auto* context = new (std::nothrow) YoloPoseContextImpl();
    if (!context) {
        std::cerr << "Failed to allocate YoloPoseContextImpl." << std::endl;
        return nullptr;
    }

    // 设置YoloPose的参数
    context->yolo_params = {
        {"maximum_batch", yolo_max_batch},
        {"maximum_items", 100},
        {"infer_features", 56},
        {"infer_samples", 8400},
        {"cls", yolo_cls_threshold},
        {"iou", yolo_iou_threshold}
    };

    // 创建YoloPose模型
    context->pose_model = ModelFactory::createModel("YoloV8_Pose", model_path, context->yolo_params);
    if (!context->pose_model) {
        std::cerr << "Failed to create YoloV8_Pose model." << std::endl;
        delete context;
        return nullptr;
    }

    // 返回上下文指针
    return reinterpret_cast<void*>(context);
}


// Helper function to perform YoloPose inference
static std::map<int, std::vector<YoloPose>>
perform_yolo_inference(YoloPoseContextImpl* context,
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

    // 使用 YoloPose 模型的 preprocess 方法处理每个图像
    for (int i = 0; i < yolo_batch_size; ++i) {
        context->pose_model->preprocess(yolo_input_images[i], i);
    }

    // 执行 YoloPose 模型的推理
    context->pose_model->inference();

    // 使用 YoloPose 模型的 postprocess 方法获取每个图像的检测结果
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


C_BatchedPoseResults c_process_batched_images_with_yolopose(void* context_handle, const unsigned char* const* input_images_data, 
    const int* widths, const int* heights, const int* channels, int num_images) {

    // 0) 初始化返回
    C_BatchedPoseResults c_results = {0, nullptr};

    // 1) 检查上下文
    if (!context_handle) {
        std::cerr << "Invalid YoloPoseContext provided." << std::endl;
        return c_results;
    }
    auto* ctx = reinterpret_cast<YoloPoseContextImpl*>(context_handle);

    if (num_images <= 0) return c_results;

    // 2) 预处理输入图像（把有效图像装进 vector，并建立原始->处理后索引映射）
    auto [valid_original_images_for_cropping, original_to_processed_idx_map] =
        preprocess_images_for_yolo(input_images_data, widths, heights, channels, num_images);

    if (valid_original_images_for_cropping.empty()) {
        std::cerr << "No valid images to process in the batch." << std::endl;
        return c_results;
    }

    // 处理后->原始 的逆映射（用于把 YOLO 结果放回原始图像下标）
    std::map<int, int> processed_to_original_idx_map;
    for (auto const& [orig_idx, proc_idx] : original_to_processed_idx_map) {
        processed_to_original_idx_map[proc_idx] = orig_idx;
    }

    // 3) 仅用 YOLO 做推理
    std::map<int, std::vector<YoloPose>> cpp_batched_pose_detections =
        perform_yolo_inference(ctx, valid_original_images_for_cropping, processed_to_original_idx_map);

    // 4) 构造 “按原始下标” 的最终 map（没有 EfficientNet，直接把 YOLO 结果映射回去）
    std::map<int, std::vector<YoloPose>> final_cpp_detections_map;
    // 先为所有原始图像建空槽，保证每个 i 都有结果容器
    for (int i = 0; i < num_images; ++i) final_cpp_detections_map[i] = {};

    for (const auto& [proc_idx, poses] : cpp_batched_pose_detections) {
        auto it = processed_to_original_idx_map.find(proc_idx);
        if (it == processed_to_original_idx_map.end()) {
            std::cerr << "Warning: processed index " << proc_idx
                      << " missing in processed_to_original_idx_map." << std::endl;
            continue;
        }
        int orig_idx = it->second;
        // 直接把该处理图像的检测复制回原始下标
        final_cpp_detections_map[orig_idx].insert(
            final_cpp_detections_map[orig_idx].end(), poses.begin(), poses.end());
    }

    // 5) 分配 C 结构数组内存，并初始化（若后续步骤失败可安全清理）
    c_results.num_images = num_images;
    c_results.results = static_cast<C_ImagePoseResults*>(
        malloc(num_images * sizeof(C_ImagePoseResults)));
    if (!c_results.results) {
        std::cerr << "Memory allocation failed for C_ImagePoseResults array." << std::endl;
        c_results.num_images = 0;
        return c_results;
    }

    for (int i = 0; i < num_images; ++i) {
        c_results.results[i].image_idx = i;
        c_results.results[i].num_detections = 0;
        c_results.results[i].detections = nullptr;
    }

    // 6) 把 C++ 的 vector<YoloPose> 转为 C 数组（带失败清理）
    std::atomic<bool> conversion_failed = false;

    // #pragma omp parallel for
    for (int i = 0; i < num_images; ++i) {
        if (conversion_failed.load(std::memory_order_relaxed)) {
            continue;
        }

        const auto& cpp_poses = final_cpp_detections_map.at(i); // 有可能为空
        c_results.results[i].num_detections = static_cast<int>(cpp_poses.size());

        C_YoloPose* current_image_detections = convert_yolo_poses_to_c(cpp_poses);
        // 注意：当 cpp_poses 为空时，convert_yolo_poses_to_c 按设计返回 nullptr 是合理的
        //（见实现），我们不当作失败；只有“不空却返回 nullptr”才视为失败。
        if (current_image_detections == nullptr && !cpp_poses.empty()) {
            conversion_failed.store(true, std::memory_order_relaxed);
            std::cerr << "Partial conversion to C_YoloPose failed for image " << i << "." << std::endl;
        }
        c_results.results[i].detections = current_image_detections;
    }

    if (conversion_failed.load(std::memory_order_relaxed)) {
        // 统一清理，返回空结果
        for (int i = 0; i < num_images; ++i) {
            free_batched_pose_results_single_image(&c_results.results[i]);
        }
        free(c_results.results);
        c_results.results = nullptr;
        c_results.num_images = 0;
        return c_results;
    }

    return c_results;
};


void c_destroy_yolopose_context(void* context) {
    if (context) {
        auto* ctx = reinterpret_cast<YoloPoseContextImpl*>(context);
        delete ctx;
    }
}


#ifdef __cplusplus
};
#endif