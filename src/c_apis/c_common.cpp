#include "trtengine/c_apis/c_common.h"

#include <iostream>
#include <vector>
#include <cstring>
#include <opencv2/opencv.hpp>



// Helper to free a single C_ImagePoseResults structure's internal memory
void free_batched_pose_results_single_image(C_ImagePoseResults* single_image_results) {
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


// Helper function to convert std::vector<YoloPose> to C_YoloPose*
// This function is still responsible for C-style memory allocation, which the caller must free.
C_YoloPose* convert_yolo_poses_to_c(const std::vector<YoloPose>& cpp_poses) {
    if (cpp_poses.empty()) {
        return nullptr;
    }

    // Allocate the main array. Use a raw pointer as ownership will be released.
    auto* c_poses_raw = static_cast<C_YoloPose*>(malloc(cpp_poses.size() * sizeof(C_YoloPose)));
    if (!c_poses_raw) {
        std::cerr << "Memory allocation failed for C_YoloPose array." << std::endl;
        return nullptr;
    }

    // Initialize pointers to nullptr to ensure safe free() calls later
    // 使用 OpenMP 加速
    // #pragma omp parallel for
    for (size_t i = 0; i < cpp_poses.size(); ++i) {
        c_poses_raw[i].pts = nullptr;
        c_poses_raw[i].feats = nullptr;
    }

    // Loop to populate the C_YoloPose array
    // 使用 OpenMP 并行化处理
    for (size_t i = 0; i < cpp_poses.size(); ++i) {
        c_poses_raw[i].lx = cpp_poses[i].lx;
        c_poses_raw[i].ly = cpp_poses[i].ly;
        c_poses_raw[i].rx = cpp_poses[i].rx;
        c_poses_raw[i].ry = cpp_poses[i].ry;
        c_poses_raw[i].cls = static_cast<float>(cpp_poses[i].cls);
        c_poses_raw[i].num_pts = static_cast<int>(cpp_poses[i].pts.size());
        c_poses_raw[i].conf = cpp_poses[i].conf;

        // Allocate memory for keypoints
        if (!cpp_poses[i].pts.empty()) {
            c_poses_raw[i].pts = static_cast<C_KeyPoint*>(malloc(cpp_poses[i].pts.size() * sizeof(C_KeyPoint)));
            if (!c_poses_raw[i].pts) {
                std::cerr << "Memory allocation failed for C_KeyPoint array for pose " << i << "." << std::endl;
                // Cleanup all previously allocated pts and feats, and the main c_poses_raw array
                // before returning nullptr.
                for (size_t j = 0; j <= i; ++j) { // Crucial: loop up to current 'i'
                    free(c_poses_raw[j].pts);
                    free(c_poses_raw[j].feats);
                }
                free(c_poses_raw);
                return nullptr;
            }

            for (size_t k = 0; k < cpp_poses[i].pts.size(); ++k) {
                c_poses_raw[i].pts[k].x = static_cast<float>(cpp_poses[i].pts[k].x);
                c_poses_raw[i].pts[k].y = static_cast<float>(cpp_poses[i].pts[k].y);
                c_poses_raw[i].pts[k].conf = cpp_poses[i].pts[k].conf;
            }
        }

        // Allocate memory for features if they exist
        if (!cpp_poses[i].feats.empty()) {
            c_poses_raw[i].feats = static_cast<float*>(malloc(cpp_poses[i].feats.size() * sizeof(float)));
            if (!c_poses_raw[i].feats) {
                std::cerr << "Memory allocation failed for C_YoloPose feats array for pose " << i << "." << std::endl;
                // Cleanup all previously allocated pts and feats, and the main c_poses_raw array
                // before returning nullptr.
                for (size_t j = 0; j <= i; ++j) { // Crucial: loop up to current 'i'
                    free(c_poses_raw[j].pts);
                    free(c_poses_raw[j].feats);
                }
                free(c_poses_raw);
                return nullptr;
            }
            std::memcpy(c_poses_raw[i].feats, cpp_poses[i].feats.data(), cpp_poses[i].feats.size() * sizeof(float));
        }
    }
    return c_poses_raw; // Return the raw pointer, ownership transferred to caller
}

// Helper function to preprocess input images for YoloPose
std::pair<std::vector<cv::Mat>, std::map<int, int>>
preprocess_images_for_yolo(const unsigned char* const* input_images_data, const int* widths, const int* heights,
                           const int* channels, int num_images) {

    std::vector<cv::Mat> valid_original_images_for_cropping;
    std::map<int, int> original_to_processed_idx_map;
    int valid_count = 0;

    // 使用 OpenMP 并行化处理图像预处理
    // #pragma omp parallel for
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
