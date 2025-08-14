#ifndef COMBINEDPROJECT_C_COMMON_H
#define COMBINEDPROJECT_C_COMMON_H

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <map>
#include <any>
#include <utility>            // 为了 std::pair
#include <opencv2/core.hpp>   // 或者 <opencv2/opencv.hpp>

#include "trtengine/c_apis/c_structs.h"
#include "trtengine/servlet/models/common/yolo_dstruct.h"


void free_batched_pose_results_single_image(C_ImagePoseResults* single_image_results);


C_YoloPose* convert_yolo_poses_to_c(const std::vector<YoloPose>& cpp_poses);


std::pair<std::vector<cv::Mat>, std::map<int, int>>
preprocess_images_for_yolo(const unsigned char* const* input_images_data, const int* widths, const int* heights,
                           const int* channels, int num_images);


#endif // COMBINEDPROJECT_C_COMMON_H