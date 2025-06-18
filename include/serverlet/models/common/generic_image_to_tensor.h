//
// Created by user on 6/13/25.
//

#ifndef IMAGE_TO_TENSOR_H
#define IMAGE_TO_TENSOR_H

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief 将 OpenCV 图像转换为 CUDA 张量格式，并可选地进行归一化。
 *
 * 该函数执行以下步骤：
 * 1. 调整图像大小到目标维度。
 * 2. 如果指定，将图像从 BGR 转换为 RGB。
 * 3. 将图像数据转换为 float 类型并归一化到 [0, 1] 范围。
 * 4. 如果提供了 mean 和 stdv，则使用它们对数据进行标准化。
 * 5. 将 HWC 格式的图像数据转换为 CHW 格式。
 * 6. 将最终结果拷贝到提供的 CUDA 设备指针。
 *
 * @param image 输入的 OpenCV 图像。
 * @param device_ptr CUDA 设备指针，用于存储转换后的张量数据，转换后数据格式为 [C, H, W]。
 * 此指针应指向已分配好的 CUDA 内存，其大小应足以容纳 target_dims 指定的张量。
 * @param target_height 目标高度，图像将被调整到此高度。
 * @param target_width 目标宽度，图像将被调整到此宽度
 * @param target_channels 目标通道数，通常为 3（RGB）。如果图像是单通道灰度图像，则可以设置为 1。
 * @param is_rgb 是否将图像转换为 RGB 格式（默认为 true，表示进行 BGR 到 RGB 转换）。
 * @param mean 每个通道的均值，用于归一化。如果提供空 vector 或不提供，则跳过均值归一化。
 * @param stdv 每个通道的标准差，用于归一化。如果提供空 vector 或不提供，则跳过标准差归一化。
 * 注意：如果提供了 mean 但未提供 stdv，或反之，可能会导致不完整的归一化。
 * 建议同时提供 mean 和 stdv，或同时不提供。
 */
void imageToCudaTensor(
    const cv::Mat& image,
    float* device_ptr,
    int target_height, // 目标高度
    int target_width,  // 目标宽度
    int target_channels, // 目标通道数，通常为 3（RGB）
    bool is_rgb = true, // Default to true for is_rgb
    const std::vector<float>& mean = {}, // Default empty vector means no mean normalization
    const std::vector<float>& stdv = {}  // Default empty vector means no stdv normalization
);

#endif //IMAGE_TO_TENSOR_H
