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
    static C_YoloPose* convert_yolo_poses_to_c(const std::vector<YoloPose>& cpp_poses) {
        if (cpp_poses.empty()) {
            return nullptr;
        }
        auto* c_poses = static_cast<C_YoloPose*>(malloc(cpp_poses.size() * sizeof(C_YoloPose)));
        if (!c_poses) {
            std::cerr << "Memory allocation failed for C_YoloPose array." << std::endl;
            return nullptr;
        }

        for (size_t i = 0; i < cpp_poses.size(); ++i) {
            c_poses[i].lx = cpp_poses[i].lx;
            c_poses[i].ly = cpp_poses[i].ly;
            c_poses[i].rx = cpp_poses[i].rx;
            c_poses[i].ry = cpp_poses[i].ry;
            c_poses[i].cls = static_cast<float>(cpp_poses[i].cls);
            c_poses[i].num_pts = static_cast<int>(cpp_poses[i].pts.size()); // Use size_t to int conversion

            // Allocate memory for keypoints
            if (cpp_poses[i].pts.empty()) {
                c_poses[i].pts = nullptr;
            } else {
                // c_poses[i].pts = (C_KeyPoint*)malloc(cpp_poses[i].pts.size() * sizeof(C_KeyPoint));
                c_poses[i].pts = static_cast<C_KeyPoint*>(malloc(cpp_poses[i].pts.size() * sizeof(C_KeyPoint)));
                if (!c_poses[i].pts) {
                    std::cerr << "Memory allocation failed for C_KeyPoint array." << std::endl;
                    // Clean up already allocated keypoint arrays for this C_YoloPose array
                    for(size_t j = 0; j < i; ++j) {
                        free(c_poses[j].pts); // Free keypoints for poses already allocated
                    }
                    free(c_poses); // Free the main poses array
                    return nullptr;
                }
                for (size_t k = 0; k < cpp_poses[i].pts.size(); ++k) {
                    c_poses[i].pts[k].x = static_cast<float>(cpp_poses[i].pts[k].x);
                    c_poses[i].pts[k].y = static_cast<float>(cpp_poses[i].pts[k].y);
                    // Removed 'prob' assignment as YoloPose::KeyPoint might not have it.
                    // If it should have a default, set it here.
                    c_poses[i].pts[k].conf = 0.0f; // Default to 0 if not available from YoloPose
                    // Original error: 'const value_type' {aka 'const struct YoloPoint'} has no member named 'prob'
                    // c_poses[i].pts[k].prob = cpp_poses[i].pts[k].prob;
                }
            }

            // Allocate memory for features if they exist
            if (cpp_poses[i].feats.empty()) {
                c_poses[i].feats = nullptr;
            } else {
                c_poses[i].feats = static_cast<float*>(malloc(cpp_poses[i].feats.size() * sizeof(float)));
                if (!c_poses[i].feats) {
                    std::cerr << "Memory allocation failed for C_YoloPose feats array." << std::endl;
                    // Clean up already allocated feats for this C_YoloPose array
                    for(size_t j = 0; j < i; ++j) {
                        free(c_poses[j].feats); // Free feats for poses already allocated
                    }
                    free(c_poses); // Free the main poses array
                    return nullptr;
                }
                // std::ranges::copy(cpp_poses[i].feats, c_poses[i].feats); // This requires C++20, use an alternative in C++17
                // for (size_t k = 0; k < cpp_poses[i].feats.size(); ++k) {
                    // c_poses[i].feats[k] = cpp_poses[i].feats[k];
                // }
                std::memcpy(c_poses[i].feats, cpp_poses[i].feats.data(), cpp_poses[i].feats.size() * sizeof(float));
            }
        }
        return c_poses;
    }

    // ------------------------------------------ C API Functions ------------------------------------------
    void c_register_models()
    {
        // 注册 YOLOv8 姿态估计模型
        ModelFactory::registerModel("YoloV8_Pose",
            [](const std::string& engine_path, const std::map<std::string, std::any>& params) {

            // 从参数中获取必要的配置
            int maximum_batch = GET_PARAM(params, "maximum_batch", int);
            int maximum_items = GET_PARAM(params, "maximum_items", int);
            int infer_features = GET_PARAM(params, "infer_features", int);
            int infer_samples = GET_PARAM(params, "infer_samples", int);

            std::vector<TensorDefinition> output_tensor_defs = std::vector<TensorDefinition>{{"output0",
                {maximum_batch, infer_features, infer_samples}}};

            // YOLOv8 姿态估计的转换函数
            auto pose_converter = [](const std::vector<float>& input, std::vector<YoloPose>& output, int features, int results) {
                cvtXYWHCoordsToYoloPose(input, output, features, results);
            };

            return std::make_unique<InferYoloV8<YoloPose, decltype(pose_converter)>>(
                engine_path, maximum_batch, maximum_items, infer_features, output_tensor_defs, pose_converter
            );
        });

        // 注册 EfficientNet 模型
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

        // Initialize YoloPose parameters
        context->yolo_params = {
            {"maximum_batch", yolo_max_batch},
            {"maximum_items", 100}, // Assuming a default reasonable value
            {"infer_features", 56}, // Assuming default
            {"infer_samples", 8400}, // Assuming default
            {"cls", yolo_cls_thresh},
            {"iou", yolo_iou_thresh}
        };

        // Initialize EfficientNet parameters
        context->efficient_params = {
            {"maximum_batch", efficient_max_batch}
        };

        // Create models
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

        return reinterpret_cast<YoloEfficientContext*>(context); // Cast to YoloEfficientContext*
    }

    C_BatchedPoseResults c_process_batched_images(
        YoloEfficientContext* context_handle,
        const unsigned char* const* input_images_data,
        const int* widths,
        const int* heights,
        const int* channels,
        int num_images) {

        C_BatchedPoseResults c_results = {0, nullptr}; // Initialize to empty results

        if (!context_handle) {
            std::cerr << "Invalid YoloEfficientContext provided." << std::endl;
            return c_results;
        }

        // auto* context = static_cast<YoloEfficientContextImpl*>(*context_handle); // This cast is now valid
        auto context = reinterpret_cast<YoloEfficientContextImpl*>(context_handle);

        if (num_images <= 0) {
            return c_results; // Nothing to process
        }

        // --- Convert raw input images to cv::Mat and resize for YoloPose ---
        std::vector<cv::Mat> batched_resized_images;
        std::vector<cv::Mat> batched_original_images_for_cropping; // Keep original aspect ratio images for cropping
        batched_resized_images.reserve(num_images);
        batched_original_images_for_cropping.reserve(num_images);

        for (int i = 0; i < num_images; ++i) {
            if (!input_images_data[i] || widths[i] <= 0 || heights[i] <= 0 || channels[i] <= 0) {
                std::cerr << "Invalid image data provided for image index " << i << ". Skipping." << std::endl;
                batched_resized_images.emplace_back(); // Add empty mat
                batched_original_images_for_cropping.emplace_back();
                continue;
            }

            cv::Mat original_img;
            if (channels[i] == 3) {
                original_img = cv::Mat(heights[i], widths[i], CV_8UC3, (void*)input_images_data[i]);
            } else if (channels[i] == 1) {
                original_img = cv::Mat(heights[i], widths[i], CV_8UC1, (void*)input_images_data[i]);
            } else {
                std::cerr << "Unsupported number of channels: " << channels[i] << " for image index " << i << ". Skipping." << std::endl;
                batched_resized_images.emplace_back();
                batched_original_images_for_cropping.emplace_back();
                continue;
            }

            // Clone to ensure data ownership by cv::Mat, especially if input_images_data is temporary
            original_img = original_img.clone();
            batched_original_images_for_cropping.push_back(original_img);

            cv::Mat resized_img;
            cv::resize(original_img, resized_img, cv::Size(640, 640)); // YoloPose input size
            batched_resized_images.push_back(resized_img);
        }

        // Filter out invalid images for actual processing
        std::vector<cv::Mat> valid_resized_images;
        std::vector<cv::Mat> valid_original_images_for_cropping;
        std::map<int, int> original_to_processed_idx_map; // Maps original input index to index in valid_resized_images
        int valid_count = 0;
        for(int i = 0; i < num_images; ++i) {
            if (!batched_resized_images[i].empty()) {
                valid_resized_images.push_back(batched_resized_images[i]);
                valid_original_images_for_cropping.push_back(batched_original_images_for_cropping[i]);
                original_to_processed_idx_map[i] = valid_count++;
            }
        }

        if (valid_resized_images.empty()) {
            std::cerr << "No valid images to process in the batch." << std::endl;
            return c_results;
        }


        // --- YoloPose Processing ---
        // Ensure yolo_max_batch doesn't exceed the number of available valid input images
        size_t yolo_batch_size = std::min(static_cast<size_t>(std::any_cast<int>(context->yolo_params["maximum_batch"])), valid_resized_images.size());

        for (int i = 0; i < yolo_batch_size; ++i) {
            context->pose_model->preprocess(valid_resized_images[i], i);
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
            } catch (...) {
                std::cerr << "YoloPose API: Unknown error during single image postprocessing or cast for batch index " << i << "." << std::endl;
                cpp_batched_pose_detections[i] = {};
            }
        }

        // --- Flatten and Crop for EfficientNet ---
        struct FlattenedPose {
            int original_image_idx_in_batch; // Original index in the input_images_data array
            size_t original_pose_idx_in_image; // Index of pose within its image's detections
            YoloPose pose;
            cv::Mat crop_image; // The actual cropped image data
        };
        std::vector<FlattenedPose> all_flattened_poses_with_crops;

        size_t efficient_max_batch = std::any_cast<int>(context->efficient_params["maximum_batch"]);

        // This map helps in re-associating results if original_image_idx changes due to filtering
        // Key: index in valid_resized_images, Value: original_image_idx from input_images_data
        std::map<int, int> processed_to_original_idx_map;
        for (auto const& [orig_idx, proc_idx] : original_to_processed_idx_map) {
            processed_to_original_idx_map[proc_idx] = orig_idx;
        }


        for (auto const& [processed_img_idx, poses_in_image] : cpp_batched_pose_detections) {
            // Ensure processed_img_idx is valid for processed_to_original_idx_map
            if (!processed_to_original_idx_map.contains(processed_img_idx)) {
                std::cerr << "Error: Processed image index " << processed_img_idx << " not found in original_to_processed_idx_map." << std::endl;
                continue;
            }
            // int original_img_idx_from_input = processed_to_original_idx_map[processed_img_idx];

            for (size_t i = 0; i < poses_in_image.size(); ++i) {
                constexpr float scale_factor = 1.2f;
                const auto& pose = poses_in_image[i];

                // Ensure processed_img_idx is valid for valid_original_images_for_cropping
                if (processed_img_idx >= valid_original_images_for_cropping.size()) {
                    std::cerr << "Error: Processed image index " << processed_img_idx << " out of bounds for valid_original_images_for_cropping." << std::endl;
                    continue;
                }
                const cv::Mat& source_image = valid_original_images_for_cropping[processed_img_idx];

                if (source_image.empty() || pose.pts.empty()) continue; // Skip if source image invalid or no keypoints

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
                    all_flattened_poses_with_crops.push_back({processed_to_original_idx_map[processed_img_idx], i, pose, cropped_img});
                }
            }
        }


        // --- EfficientNet Processing in Batches ---
        std::vector<std::vector<float>> efficient_net_all_results; // Stores all classification results

        for (size_t batch_start_idx = 0; batch_start_idx < all_flattened_poses_with_crops.size(); batch_start_idx += efficient_max_batch) {
            size_t batch_end_idx = std::min(batch_start_idx + efficient_max_batch, all_flattened_poses_with_crops.size());

            std::vector<cv::Mat> current_efficient_batch_images;
            current_efficient_batch_images.reserve(efficient_max_batch);

            for (size_t k = batch_start_idx; k < batch_end_idx; ++k) {
                current_efficient_batch_images.push_back(all_flattened_poses_with_crops[k].crop_image);
            }

            if (current_efficient_batch_images.empty()) {
                continue;
            }

            // EfficientNet Preprocess for this sub-batch
            for (int i = 0; i < current_efficient_batch_images.size(); ++i) {
                context->efficient_model->preprocess(current_efficient_batch_images[i], i);
            }

            // EfficientNet Inference for this sub-batch
            context->efficient_model->inference();

            // EfficientNet Postprocess for this sub-batch
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
                } catch (...) {
                    std::cerr << "EfficientNet API: Unknown error during single crop postprocessing or cast for batch index " << i << "." << std::endl;
                    cls_results_sub_batch.emplace_back();
                }
            }
            efficient_net_all_results.insert(efficient_net_all_results.end(), cls_results_sub_batch.begin(), cls_results_sub_batch.end());
        }

        // --- Re-associate EfficientNet results back to original YoloPose detections ---
        // Create a temporary map to store results, using original image indices
        std::map<int, std::vector<YoloPose>> final_cpp_detections_map;
        // Initialize with all original image indices, even if no poses found
        for(int i = 0; i < num_images; ++i) {
            final_cpp_detections_map[i] = {}; // Will be filled below
        }

        size_t result_idx_counter = 0;
        for (const auto& flat_pose_with_crop : all_flattened_poses_with_crops) {

            // 获取当前的 EfficientNet 结果
            if (result_idx_counter < efficient_net_all_results.size()
                && efficient_net_all_results[result_idx_counter].size() >= 257) {
                YoloPose updated_pose = flat_pose_with_crop.pose; // Start with the original pose

                // 1) 更新类别 cls（索引 0）
                updated_pose.cls = static_cast<int>(efficient_net_all_results[result_idx_counter][0]);

                // 2) 写入 EfficientNet 的 256 维特征向量（索引 1~256）
                updated_pose.feats.clear();
                updated_pose.feats.insert(
                    updated_pose.feats.end(),
                    efficient_net_all_results[result_idx_counter].begin() + 1,
                    efficient_net_all_results[result_idx_counter].end()
                );

                // 3) 然后按原逻辑把 updated_pose 放回 final_cpp_detections_ma
                if (final_cpp_detections_map.contains(flat_pose_with_crop.original_image_idx_in_batch)) {
                    // Ensure the vector for this image index is large enough
                    if (flat_pose_with_crop.original_pose_idx_in_image < final_cpp_detections_map[flat_pose_with_crop.original_image_idx_in_batch].size()) {
                        final_cpp_detections_map[flat_pose_with_crop.original_image_idx_in_batch][flat_pose_with_crop.original_pose_idx_in_image] = updated_pose;
                    } else {
                        // This case indicates that the pose index exceeds current size, means it was added
                        // during flattening but the initial map creation might not have pre-sized it correctly.
                        // For robustness, if it's new, add it.
                        final_cpp_detections_map[flat_pose_with_crop.original_image_idx_in_batch].push_back(updated_pose);
                    }
                } else {
                    // This original image index was not even initialized, which implies an issue
                    // with the initial loop filling final_cpp_detections_map.
                    // For robustness, add it.
                    final_cpp_detections_map[flat_pose_with_crop.original_image_idx_in_batch].push_back(updated_pose);
                }
            } else {
                // If EfficientNet didn't produce a result for this pose (e.g., crop failed, or result_idx_counter out of bounds)
                // Still add the original YoloPose, but without updated classification
                if (final_cpp_detections_map.contains(flat_pose_with_crop.original_image_idx_in_batch)) {
                    if (flat_pose_with_crop.original_pose_idx_in_image < final_cpp_detections_map[flat_pose_with_crop.original_image_idx_in_batch].size()) {
                        // Pose already exists (copied from YoloPose detections) and was not updated with classification, no action needed.
                    } else {
                        final_cpp_detections_map[flat_pose_with_crop.original_image_idx_in_batch].push_back(flat_pose_with_crop.pose);
                    }
                } else {
                    final_cpp_detections_map[flat_pose_with_crop.original_image_idx_in_batch].push_back(flat_pose_with_crop.pose);
                }
            }
            result_idx_counter++;
        }

        // Fill in poses for images that were processed by YoloPose but didn't have any detections
        // or were not handled by the flattened loop (e.g., if a YoloPose batch element had 0 detections)
        for (auto const& [img_idx, poses_vec] : cpp_batched_pose_detections) {
            // If this image index is not in the final map OR if it's in the map but has fewer detections
            // than the original YoloPose results (meaning some were skipped/filtered during cropping or EfficientNet),
            // we ensure the final map contains all original YoloPose detections (possibly without EfficientNet classification)
            if (final_cpp_detections_map.find(img_idx) == final_cpp_detections_map.end() ||
                final_cpp_detections_map[img_idx].size() < poses_vec.size()) {

                // Reinitialize with the original YoloPose detections to ensure all are present
                final_cpp_detections_map[img_idx] = poses_vec;

                // Now, iterate through the ones that *were* processed by EfficientNet and update their 'cls'
                size_t current_flat_pose_idx = 0;
                for(const auto& flat_pose_with_crop : all_flattened_poses_with_crops) {
                    if (flat_pose_with_crop.original_image_idx_in_batch == img_idx &&
                        current_flat_pose_idx < efficient_net_all_results.size() &&
                        !efficient_net_all_results[current_flat_pose_idx].empty()) {

                        if (flat_pose_with_crop.original_pose_idx_in_image < final_cpp_detections_map[img_idx].size()) {
                            final_cpp_detections_map[img_idx][flat_pose_with_crop.original_pose_idx_in_image].cls =
                                static_cast<int>(efficient_net_all_results[current_flat_pose_idx][0]);
                        }
                    }
                    current_flat_pose_idx++;
                }
            }
        }


        // --- Convert C++ results to C-compatible structure ---
        c_results.num_images = num_images; // Report results for all input images, even if empty
        c_results.results = static_cast<C_ImagePoseResults*>(malloc(num_images * sizeof(C_ImagePoseResults)));
        if (!c_results.results) {
            std::cerr << "Memory allocation failed for C_ImagePoseResults array." << std::endl;
            c_results.num_images = 0;
            return c_results;
        }

        for (int i = 0; i < num_images; ++i) {
            c_results.results[i].image_idx = i;
            if (final_cpp_detections_map.contains(i)) {
                const auto& cpp_poses = final_cpp_detections_map[i];
                c_results.results[i].num_detections = static_cast<int>(cpp_poses.size());
                c_results.results[i].detections = convert_yolo_poses_to_c(cpp_poses);
                if (c_results.results[i].detections == nullptr && !cpp_poses.empty()) {
                    // Failed to convert a sub-array, need to clean up previous allocations
                    for (int j = 0; j < i; ++j) {
                        free_batched_pose_results_single_image(&c_results.results[j]);
                    }
                    free(c_results.results);
                    c_results.results = nullptr;
                    c_results.num_images = 0;
                    std::cerr << "Partial conversion failed, returning empty results." << std::endl;
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
                if (single_image_results->detections[i].pts) { // Check if pts is not NULL before freeing
                    free(single_image_results->detections[i].pts);
                    single_image_results->detections[i].pts = nullptr;
                }
            }
            free(single_image_results->detections);
            single_image_results->detections = nullptr;
        }
    }

    void c_free_batched_pose_results(C_BatchedPoseResults* results) {
        if (!results || !results->results) return;

        for (int i = 0; i < results->num_images; ++i) {
            C_ImagePoseResults* img = &results->results[i];
            if (!img->detections) continue;

            // 对每个检测，先释放 pts，再释放 feats
            for (int j = 0; j < img->num_detections; ++j) {
                C_YoloPose* det = &img->detections[j];
                if (det->pts) {
                    free(det->pts);
                    det->pts = nullptr;
                }
                if (det->feats) {
                    free(det->feats);
                    det->feats = nullptr;
                }
            }
            // 再释放这一张图的 C_YoloPose 数组
            free(img->detections);
            img->detections = nullptr;
            img->num_detections = 0;
        }

        // 最后释放图片结果数组
        free(results->results);
        results->results = nullptr;
        results->num_images = 0;
    }

    void c_destroy_pose_pipeline(YoloEfficientContext* context_handle) {
        if (context_handle) {
            auto * context = reinterpret_cast<YoloEfficientContext*>(context_handle);
            delete context;
        }
    }

#ifdef __cplusplus
}
#endif