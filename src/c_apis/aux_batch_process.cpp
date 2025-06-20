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

#define DEBUG 0


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

    // Loop, processing images batch by batch until none remain
    while (remaining_images_count > 0) {
        size_t current_batch_count = std::min(remaining_images_count, (size_t)pose_model_max_batch_size);

        // --- MODIFICATION START ---
        // Copy images from the BEGINNING of the input vector for the current batch.
        std::vector<cv::Mat> current_batch_images_for_processing;
        current_batch_images_for_processing.reserve(current_batch_count);

        for(size_t i = 0; i < current_batch_count; ++i) {
            // Copy from the beginning of the 'images' vector
            current_batch_images_for_processing.push_back(images[i].clone());
        }

        // Erase processed images from the BEGINNING of the original `images` vector.
        images.erase(images.begin(), images.begin() + current_batch_count);
        // --- MODIFICATION END ---

        remaining_images_count = images.size(); // Update remaining count

        LOG_VERBOSE_TOPIC("BatchProcess", "Pose", "Processing a batch of " + std::to_string(current_batch_count) + " images.");

        // Process the current batch of images
        std::vector<InferenceResult> batch_results_from_internal = process_single_batch_internal(
            current_batch_images_for_processing, // Pass images for this batch
            pose_model,
            pose_pp_params
        );

        // Append results from this batch to the final output
        final_output.insert(final_output.end(),
                            std::make_move_iterator(batch_results_from_internal.begin()),
                            std::make_move_iterator(batch_results_from_internal.end()));
    }

#if DEBUG
    LOG_DEBUG_V3("BatchProcess", "Finished processing all images by pose engine. Total images processed: " + std::to_string(final_output.size()));
#endif
    return final_output;
}


// NEW HELPER: Prepares cropped and resized images for EfficientNet processing
EfficientNetCrops prepare_efficientnet_crops(const InferenceResult& image_result, float scale_factor) {
    EfficientNetCrops crops_output;

    // Check for valid detections for this image
    if (image_result.num_detected <= 0 || image_result.detections.empty()) {
        LOG_WARNING("BatchProcess", "No valid detections to crop for EfficientNet in this image.");
        return crops_output; // Return empty crops
    }

    const cv::Mat& source_image = image_result.processed_image; // This is the 640x640 image

    for (size_t i = 0; i < image_result.detections.size(); ++i) {
        const auto& person_pose = image_result.detections[i]; // C_Extended_Pose_Feats

        // Calculate bounding box (x1, y1, x2, y2)
        int min_x = static_cast<int>(person_pose.box.x1);
        int min_y = static_cast<int>(person_pose.box.y1);
        int max_x = static_cast<int>(person_pose.box.x2);
        int max_y = static_cast<int>(person_pose.box.y2);

        int width_bb = max_x - min_x;
        int height_bb = max_y - min_y;

        // Calculate crop region with scale factor and clamp to image boundaries
        int crop_x = std::max(0, static_cast<int>(min_x - width_bb * (scale_factor - 1) / 2));
        int crop_y = std::max(0, static_cast<int>(min_y - height_bb * (scale_factor - 1) / 2));
        int crop_width = static_cast<int>(width_bb * scale_factor);
        int crop_height = static_cast<int>(height_bb * scale_factor);

        // Adjust crop_width and crop_height to not exceed image bounds
        crop_width = std::min(source_image.cols - crop_x, crop_width);
        crop_height = std::min(source_image.rows - crop_y, crop_height);

        // Ensure crop dimensions are positive
        if (crop_width <= 0 || crop_height <= 0) {
            LOG_WARNING("BatchProcess", "Invalid crop dimensions for person " + std::to_string(i) + " (w=" + std::to_string(crop_width) + ", h=" + std::to_string(crop_height) + "), skipping.");
            continue; // Skip if invalid crop
        }

        cv::Rect crop_rect(crop_x, crop_y, crop_width, crop_height);
        cv::Mat cropped_image_from_640 = source_image(crop_rect).clone();

        // Resize the cropped image to EfficientNet's expected input size (224x224)
        cv::Mat final_efficientnet_input;
        cv::resize(cropped_image_from_640, final_efficientnet_input, cv::Size(224, 224));

        crops_output.cropped_images.push_back(final_efficientnet_input);
        crops_output.original_indices.push_back(i); // Store original index for mapping back
    }

#if DEBUG
    LOG_VERBOSE_TOPIC("BatchProcess", "EfficientNet_Crops", "Prepared " + std::to_string(crops_output.cropped_images.size()) + " crops.");
#endif
    return crops_output;
}

// Helper function to process a single batch of cropped persons through EfficientNet
std::vector<std::pair<int, std::vector<float>>> process_single_batch_feats(
    const std::vector<cv::Mat>& cropped_persons_batch, // Batch of cropped person images (224x224)
    const std::unique_ptr<InferModelBaseMulti>& efficient_model,
    const std::map<std::string, std::any>& efficient_pp_params // EfficientNet post-processing params
) {
    std::vector<std::pair<int, std::vector<float>>> efficientnet_results_output; // Pair: <original_index, [class_id, features...]>
    efficientnet_results_output.reserve(cropped_persons_batch.size());

    // Preprocess images for this batch
    for (size_t i = 0; i < cropped_persons_batch.size(); ++i) {
        efficient_model->preprocess(cropped_persons_batch[i], i);
    }

#if DEBUG
    LOG_VERBOSE_TOPIC("BatchProcess", "EfficientNet_Internal", "Preprocess completed for " + std::to_string(cropped_persons_batch.size()) + " persons.");
#endif

    // Execute EfficientNet inference
    if (!efficient_model->inference()) {
        LOG_ERROR("BatchProcess", "EfficientNet inference failed for current batch of persons.");
        // Fill output with error indications
        for (size_t i = 0; i < cropped_persons_batch.size(); ++i) {
            efficientnet_results_output.push_back({static_cast<int>(i), {}}); // Original index, empty vector for error
        }
        return efficientnet_results_output;
    }

#if DEBUG
    LOG_VERBOSE_TOPIC("BatchProcess", "EfficientNet_Internal", "Inference completed for current batch.");
#endif

    // Postprocess EfficientNet results
    for (size_t i = 0; i < cropped_persons_batch.size(); ++i) {
        std::any efficient_raw_results;
        efficient_model->postprocess(i, efficient_pp_params, efficient_raw_results);

        std::vector<float> result_vec; // [class_id, features...]
        try {
            result_vec = std::any_cast<std::vector<float>>(efficient_raw_results);
        } catch (const std::bad_any_cast& e) {
            LOG_ERROR("BatchProcess", "Error casting EfficientNet results for batch item " + std::to_string(i) + ": " + std::string(e.what()));
        } catch (const std::exception& e) {
            LOG_ERROR("BatchProcess", "An unexpected error during EfficientNet postprocessing for batch item " + std::to_string(i) + ": " + std::string(e.what()));
        }
        // The original index is implicitly `i` within this batch's context.
        efficientnet_results_output.push_back({static_cast<int>(i), std::move(result_vec)});
    }

#if DEBUG
    LOG_VERBOSE_TOPIC("BatchProcess", "EfficientNet_Internal", "Postprocess completed for current batch.");
#endif

    return efficientnet_results_output;
}


// Main function for the EfficientNet stage, handling classification and feature extraction
std::vector<InferenceResult> run_efficientnet_stage(
    const std::vector<InferenceResult>& pose_results, // Input results from the pose detection stage
    const std::unique_ptr<InferModelBaseMulti>& efficient_model,
    const std::map<std::string, std::any>& efficient_pp_params // EfficientNet post-processing parameters
)
{
    std::vector<InferenceResult> final_output_efficient; // Output results for the EfficientNet stage

    // Check if efficient model is initialized
    if (!efficient_model) {
        LOG_ERROR("BatchProcess", "Efficient model not initialized. Cannot process pose results.");
        return final_output_efficient;
    }

    // Check if pose results vector is valid
    if (pose_results.empty()) {
        LOG_WARNING("BatchProcess", "Pose results vector is empty. No pose detections to process.");
        return final_output_efficient; // Return empty output, which is a 'success' for no input
    }

    // Get EfficientNet's maximum batch size from its parameters
    int efficient_model_max_batch_size = 0;
    try {
        efficient_model_max_batch_size = GET_PARAM(efficient_pp_params, "maximum_batch", int);
        if (efficient_model_max_batch_size <= 0) {
             LOG_ERROR("BatchProcess", "EfficientNet model configuration reports invalid maximum batch size: " + std::to_string(efficient_model_max_batch_size));
             return final_output_efficient;
        }
    } catch (const std::exception& e) {
        LOG_ERROR("BatchProcess", "Failed to get 'maximum_batch' from efficient_pp_params: " + std::string(e.what()) + ". Ensure it is set and of type int.");
        return final_output_efficient;
    }

#if DEBUG
    LOG_INFO("BatchProcess", "Starting EfficientNet stage for " + std::to_string(pose_results.size()) + " images' pose results.");
#endif

    // Loop through each image's pose detection results from the previous stage
    for (const auto& single_image_result : pose_results) { // single_image_result is of type InferenceResult

        // Create a copy of the InferenceResult for this image to modify and add to final_output
        // This ensures the original pose_results vector is not modified.
        InferenceResult current_image_output = single_image_result;

        // Check if the current image has valid detections
        // num_detected = -1 means an error in previous stage for this image
        if (current_image_output.num_detected <= 0 || current_image_output.detections.empty()) {
            LOG_VERBOSE_TOPIC("BatchProcess", "Efficient", "Skipping image with num_detected_persons: " + std::to_string(current_image_output.num_detected));
            final_output_efficient.push_back(std::move(current_image_output)); // Add the skipped result as is
            continue; // Skip to the next image
        }

        // --- NEW HELPER CALL: Prepare all cropped images for EfficientNet inference for THIS SINGLE IMAGE ---
        EfficientNetCrops person_crops = prepare_efficientnet_crops(current_image_output, 1.2f); // 1.2f is scale_factor

        // If no valid persons were cropped for this image, add the original result and continue
        if (person_crops.cropped_images.empty()) {
            LOG_INFO("BatchProcess", "No valid persons cropped for EfficientNet processing in this image.");
            final_output_efficient.push_back(std::move(current_image_output));
            continue;
        }

        // --- Now, process `person_crops.cropped_images` in batches using EfficientNet ---
        size_t persons_remaining_for_efficientnet = person_crops.cropped_images.size();
        size_t current_batch_start_idx = 0; // Index into `person_crops.cropped_images`

        while (current_batch_start_idx < persons_remaining_for_efficientnet) { // Correct loop condition
            size_t current_efficient_batch_count = std::min(persons_remaining_for_efficientnet - current_batch_start_idx, (size_t)efficient_model_max_batch_size);

            // Create a sub-vector for the current EfficientNet batch
            std::vector<cv::Mat> current_efficient_batch_images;
            current_efficient_batch_images.reserve(current_efficient_batch_count);
            for (size_t i = 0; i < current_efficient_batch_count; ++i) {
                // No need to clone here if process_single_batch_feats doesn't modify its input
                // and if the images are already clones from prepare_efficientnet_crops.
                current_efficient_batch_images.push_back(
                    person_crops.cropped_images[current_batch_start_idx + i]
                );
            }

            // Call the internal helper function for EfficientNet batch processing
            // It returns a vector of pairs: <batch_local_index, [class_id, features...]>
            std::vector<std::pair<int, std::vector<float>>> efficientnet_batch_results_raw =
                process_single_batch_feats(
                    current_efficient_batch_images,
                    efficient_model,
                    efficient_pp_params
                );

            // --- Update the `current_image_output.detections` with results from this batch ---
            for (size_t i = 0; i < efficientnet_batch_results_raw.size(); ++i) {
                // `i` is the local batch index (0 to current_efficient_batch_count - 1)
                const auto& raw_efficient_result = efficientnet_batch_results_raw[i];
                // raw_efficient_result.first is the local batch index, which is `i`
                const std::vector<float>& efficient_feats_and_cls = raw_efficient_result.second;

                // Map back to the original detection index within `current_image_output.detections`
                size_t original_idx_in_detections =
                    person_crops.original_indices[current_batch_start_idx + i];

                if (original_idx_in_detections < current_image_output.detections.size()) {
                    C_Extended_Pose_Feats& person_feat = current_image_output.detections[original_idx_in_detections];

                    if (!efficient_feats_and_cls.empty()) {
                        float predicted_class = efficient_feats_and_cls[0];
                        person_feat.class_id = predicted_class;

                        if (efficient_feats_and_cls.size() > 1) {
                            size_t feature_vec_size = efficient_feats_and_cls.size() - 1;
                            if (feature_vec_size > 256) {
                                LOG_WARNING("BatchProcess", "EfficientNet feature vector size (" + std::to_string(feature_vec_size) + ") exceeds C_Extended_Pose_Feats.features[256]. Truncating.");
                                feature_vec_size = 256;
                            }
                            // Using std::copy to safely copy features
                            std::copy(efficient_feats_and_cls.begin() + 1,
                                      efficient_feats_and_cls.begin() + 1 + feature_vec_size,
                                      person_feat.features);
                        }
                    } else {
                        // No classification result from EfficientNet
                        person_feat.class_id = -1.0f;
                        // Features remain zero-initialized or previous state
                    }
                } else {
                    LOG_ERROR("BatchProcess", "Logic error: original_idx_in_detections (" + std::to_string(original_idx_in_detections) + ") out of bounds for current_image_output.detections (size " + std::to_string(current_image_output.detections.size()) + ").");
                    // Optionally, set error class_id here if it's a critical error
                }
            } // End for (size_t i = 0; i < efficientnet_batch_results_raw.size(); ++i)

            // Move to the next set of persons for EfficientNet processing
            current_batch_start_idx += current_efficient_batch_count;
            persons_remaining_for_efficientnet -= current_efficient_batch_count;
        } // End while (persons_remaining_for_efficientnet > 0)

        final_output_efficient.push_back(std::move(current_image_output)); // Add the modified image result to final output
    } // End for (const auto& single_image_result : pose_results)

#if DEBUG
    LOG_INFO("BatchProcess", "EfficientNet stage completed. Total images processed: " + std::to_string(final_output_efficient.size()));
#endif
    return final_output_efficient;
}