#include <vector>
#include <opencv2/opencv.hpp>
#include <any>
#include <map>
#include <cstddef>     // For size_t
#include <algorithm>   // For std::min
#include <numeric>     // For std::iota (if needed)

#include <omp.h> // For OpenMP parallel processing

#include "trtengine/c_apis/aux_batch_process.h"
#include "trtengine/c_apis/c_pose_detection.h"
#include "trtengine/serverlet/models/infer_model_multi.h"
#include "trtengine/serverlet/models/common/yolo_dstruct.h"
#include "trtengine/utils/logger.h"

// Re-defining GET_PARAM here for clarity, assuming it's available in your context
#ifndef GET_PARAM
#define GET_PARAM(map, key, type) std::any_cast<type>(map.at(key))
#endif

#define DEBUG 1


// Helper function: Converts C++ YoloPose structs to C-friendly C_Extended_Pose_Feats structs.
// In this stage, class_id is initialized to -1, to be filled by EfficientNet later.
void convert_pose_to_c_struct(
    const std::vector<YoloPose>& cpp_poses,
    std::vector<C_Extended_Pose_Feats>& c_feats_out) // Output vector of C-style structs
{
    c_feats_out.clear();
    c_feats_out.resize(cpp_poses.size()); // Resize to exact size for direct assignment

    // 启动OpenMP
    #pragma omp parallel for
    for (size_t i = 0; i < cpp_poses.size(); ++i) {
        const auto& cpp_pose = cpp_poses[i];
        C_Extended_Pose_Feats c_feat_local = {}; // Initialize with zeros (important for fixed-size arrays)

        // Copy bounding box information (lx, ly, rx, ry)
        c_feat_local.box.x1 = cpp_pose.lx;
        c_feat_local.box.y1 = cpp_pose.ly;
        c_feat_local.box.x2 = cpp_pose.rx;
        c_feat_local.box.y2 = cpp_pose.ry;
        c_feat_local.confidence = cpp_pose.conf;
        c_feat_local.class_id = -1.0f; // Default value, to be filled by EfficientNet stage (now float)

        // Copy keypoint information
        size_t num_kps_to_copy = std::min(cpp_pose.pts.size(), (size_t)17);
        for (size_t j = 0; j < num_kps_to_copy; ++j) {
            c_feat_local.pts[j].x = cpp_pose.pts[j].x;
            c_feat_local.pts[j].y = cpp_pose.pts[j].y;
            c_feat_local.pts[j].score = cpp_pose.pts[j].conf; // Conf for keypoint score
        }
        // No need to explicitly fill remaining keypoints with 0.0f if c_feat_local is zero-initialized.

        // Features are not available at this stage, leave them as zero-initialized.
        // For C_Extended_Pose_Feats.features[256], it's already zeroed by ={}.

        // Assign the locally created struct to the pre-allocated spot in c_feats_out
        c_feats_out[i] = c_feat_local;
    }
}

// Helper function to process a single batch of images through the pose model
// This handles preprocess, inference, and postprocess for one model's max_batch_size
std::vector<InferenceResult> process_single_batch_internal(
    const std::vector<cv::Mat>& current_batch_input_images,     // Images for this batch
    const std::unique_ptr<InferModelBaseMulti>& pose_model,     // Pose model instance
    const std::map<std::string, std::any>& pose_pp_params       // Pose post-processing parameters
) {
    std::vector<InferenceResult> batch_results_output; // Results for images in this single batch
    batch_results_output.reserve(current_batch_input_images.size()); // Pre-allocate

    // Preprocess images for this batch
    for (size_t i = 0; i < current_batch_input_images.size(); ++i) {
        pose_model->preprocess(current_batch_input_images[i], i);
    }

#if DEBUG
    LOG_VERBOSE_TOPIC("BatchProcess", "Pose_Internal", "Preprocess completed for " + std::to_string(current_batch_input_images.size()) + " images.");
#endif

    // Execute model inference for the current batch
    if (!pose_model->inference()) {

#if DEBUG
        LOG_ERROR("BatchProcess", "Pose model inference failed for current batch.");
#endif

        // Fill batch_results_output with error indications for each image
        for (size_t i = 0; i < current_batch_input_images.size(); ++i) {
            InferenceResult error_result;
            error_result.num_detected = -1; // Indicate error
            error_result.processed_image = current_batch_input_images[i].clone();
            error_result.detections.clear(); // Empty detections
            batch_results_output.push_back(std::move(error_result));
        }
        return batch_results_output;
    }

#if DEBUG
    LOG_VERBOSE_TOPIC("BatchProcess", "Pose_Internal", "Inference completed for current batch.");
#endif

    // Postprocess and collect detection results for each image in the current batch
    for (size_t i = 0; i < current_batch_input_images.size(); ++i) {

        // 后处理 ith 的结果
        std::any pose_raw_results;
        pose_model->postprocess(i, pose_pp_params, pose_raw_results);

        // 准备一个 InferenceResult 对象来存储当前图像的结果
        InferenceResult current_image_result;

        // 克隆图片
        current_image_result.processed_image = current_batch_input_images[i].clone();

        try {
            // 尝试将 std::any 转换为 std::vector<YoloPose>
            std::vector<YoloPose> cpp_pose_detections = std::any_cast<std::vector<YoloPose>>(pose_raw_results);
            current_image_result.num_detected = cpp_pose_detections.size();

            // 将 C++ 的 YoloPose 转换为 C 风格的 C_Extended_Pose_Feats
            convert_pose_to_c_struct(cpp_pose_detections, current_image_result.detections);

        } catch (const std::bad_any_cast& e) {
            LOG_ERROR("BatchProcess", "Error casting pose results for batch item " + std::to_string(i) + ": " + std::string(e.what()));
            current_image_result.num_detected = -1; // Indicate error
            current_image_result.detections.clear();
        } catch (const std::exception& e) {
            LOG_ERROR("BatchProcess", "An unexpected error during pose postprocessing for batch item " + std::to_string(i) + ": " + std::string(e.what()));
            current_image_result.num_detected = -1; // Indicate error
            current_image_result.detections.clear();
        }

        // 将当前图像的结果添加到输出结果中，使用 std::move 的方式来避免不必要的拷贝
        batch_results_output.push_back(std::move(current_image_result));
    }

#if DEBUG
    LOG_VERBOSE_TOPIC("BatchProcess", "Pose_Internal", "Postprocess completed for current batch.");
#endif

    return batch_results_output;
}


// Main function for the pose detection stage, handling multiple batches
std::vector<InferenceResult> run_pose_detection_stage(
    std::vector<cv::Mat>& images, // Input images (will be consumed/popped)
    const std::unique_ptr<InferModelBaseMulti>& pose_model,
    const std::map<std::string, std::any>& pose_pp_params)
{
    std::vector<InferenceResult> final_output;

    // Check if pose model is initialized
    if (!pose_model) {
        LOG_ERROR("BatchProcess", "Pose model not initialized. Cannot process images.");
        return final_output; // Return empty output
    }

    // Check if input images vector is valid
    if (images.empty()) {
        LOG_WARNING("BatchProcess", "Input images vector is empty. No images to process.");
        return final_output; // Return empty output, which is a 'success' for no input
    }

    // 获得模型最大图像处理批次大小
    int pose_model_max_batch_size = 0;
    try {
        pose_model_max_batch_size = GET_PARAM(pose_pp_params, "maximum_batch", int);
        if (pose_model_max_batch_size <= 0) {
             LOG_ERROR("BatchProcess", "Pose model configuration reports invalid maximum batch size: " + std::to_string(pose_model_max_batch_size));
             return final_output;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("BatchProcess", "Failed to get 'maximum_batch' from pose_pp_params: " + std::string(e.what()) + ". Ensure it is set and of type int.");
        return final_output;
    }

    // 剩余多少图片待处理
    size_t remaining_images_count = images.size();
#if DEBUG
    LOG_INFO("BatchProcess", "Starting batch processing with pose engine for " + std::to_string(remaining_images_count) + " images.");
#endif

    // 循环，处理每个批次的图像，直到没有剩余的图像需要处理
    while (remaining_images_count > 0) {
        // 计算当前应当处理的图片批次大小
        size_t current_batch_count = std::min(remaining_images_count, (size_t)pose_model_max_batch_size);

        // 通过从末尾复制子向量的方式提取当前批次的图像。
        // 然后从原始 `images` 向量中擦除这些图像。
        std::vector<cv::Mat> current_batch_images_for_processing;
        current_batch_images_for_processing.reserve(current_batch_count);

        // 从输入向量的末尾复制图像。
        // 如果需要保留原始顺序，可以从前面复制并从前面擦除。
        // 对于 `pop_back` 策略，处理顺序与输入相反，但对于批处理来说没有影响。
        for(size_t i = 0; i < current_batch_count; ++i) {
            current_batch_images_for_processing.push_back(images[images.size() - current_batch_count + i].clone());
        }

        // 将当前批次的图像从原始 `images` 向量中擦除
        images.erase(images.end() - current_batch_count, images.end());
        remaining_images_count = images.size();

        LOG_VERBOSE_TOPIC("BatchProcess", "Pose", "Processing a batch of " + std::to_string(current_batch_count) + " images.");

        // 处理当前批次的图像
        std::vector<InferenceResult> batch_results_from_internal = process_single_batch_internal(
            current_batch_images_for_processing, // Pass images for this batch
            pose_model,
            pose_pp_params
        );

        // 将当前批次的结果合并到最终输出中
        final_output.insert(final_output.end(),
                            std::make_move_iterator(batch_results_from_internal.begin()),
                            std::make_move_iterator(batch_results_from_internal.end()));
    }

#if DEBUG
    LOG_DEBUG_V3("BatchProcess", "Finished processing all images by pose engine. Total images processed: " + std::to_string(final_output.size()));
#endif
    return final_output;
}


std::vector<InferenceResult> run_efficientnet_stage(
    const std::vector<InferenceResult>& pose_results, // Input results from the pose detection stage
    const std::unique_ptr<InferModelBaseMulti>& efficient_model,
    const std::map<std::string, std::any>& efficient_pp_params
) 
{
    std::vector<InferenceResult> final_output;

    // Check if efficient model is initialized
    if (!efficient_model) {
        LOG_ERROR("BatchProcess", "Efficient model not initialized. Cannot process pose results.");
        return final_output; // Return empty output
    }

    // Check if pose results vector is valid
    if (pose_results.empty()) {
        LOG_WARNING("BatchProcess", "Pose results vector is empty. No results to process.");
        return final_output; // Return empty output, which is a 'success' for no input
    }

    // 处理每个姿态检测结果
    // TODO: 对每个画面的姿态检测结果进行处理，追加新的信息（包括类型分类，以及人的特征向量）

    return final_output;
};
