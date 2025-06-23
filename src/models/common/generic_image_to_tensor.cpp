//
// Created by user on 6/13/25.
//

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include <simple_cuda_toolkits/vision/image_permute.h>
#include <simple_cuda_toolkits/vision/image_ops.h>

#include "trtengine/serverlet/models/common/generic_image_to_tensor.h"
#include <stdexcept> // For std::runtime_error

void imageToCudaTensor(
    const cv::Mat& image,
    float* device_ptr,
    int target_height, // 目标高度
    int target_width,  // 目标宽度
    int target_channels, // 目标通道数，通常为 3（RGB）
    bool is_rgb,
    const std::vector<float>& mean,
    const std::vector<float>& stdv)
{
    // 检查输入参数
    if (image.empty()) {
        throw std::runtime_error("Input image is empty.");
    }
    if (device_ptr == nullptr) {
        throw std::runtime_error("Device pointer is null. Please ensure it is allocated before calling this function.");
    }

    // 总大小计算
    size_t total_size = static_cast<size_t>(target_height) * target_width * target_channels;

    // 分配 CUDA 内存
    float* cuda_temp_in = nullptr;
    float* cuda_temp_out = nullptr;

    cudaError_t err_in = cudaMalloc(&cuda_temp_in, total_size * sizeof(float));
    if (err_in != cudaSuccess) {
        throw std::runtime_error("Failed to allocate CUDA memory for cuda_temp_in: " + std::string(cudaGetErrorString(err_in)));
    }

    cudaError_t err_out = cudaMalloc(&cuda_temp_out, total_size * sizeof(float));
    if (err_out != cudaSuccess) {
        cudaFree(cuda_temp_in); // Clean up already allocated memory
        throw std::runtime_error("Failed to allocate CUDA memory for cuda_temp_out: " + std::string(cudaGetErrorString(err_out)));
    }

    // 用于后续处理的图像变量
    cv::Mat processedImage;

    // --- 图像尺寸处理逻辑 ---
    // 如果输入图像的大小与目标尺寸不相同，则需要调整图像大小
    if (image.size() != cv::Size(target_width, target_height)) {
        cv::resize(image, processedImage, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);
        if (processedImage.empty()) {
            cudaFree(cuda_temp_in);
            cudaFree(cuda_temp_out);
            throw std::runtime_error("Failed to resize image.");
        }
    } else {
        // 如果尺寸相同，直接使用原始图像的副本进行后续处理，避免修改原图
        image.copyTo(processedImage);
    }
    // --- 图像尺寸处理逻辑结束 ---

    // 2. 如果需要，将图像从 BGR 转换为 RGB
    if (is_rgb && target_channels == 3) {
        cv::cvtColor(processedImage, processedImage, cv::COLOR_BGR2RGB);
    }

    // 3. 将图像转换为浮点格式并归一化到 [0, 1] 范围
    cv::Mat floatImg;
    processedImage.convertTo(floatImg, CV_32FC3, 1.0f / 255.0f);

    // 4. 将图像数据从 OpenCV Mat 拷贝到 CUDA 设备内存
    cudaError_t err_memcpy = cudaMemcpy(cuda_temp_in, floatImg.data, total_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err_memcpy != cudaSuccess) {
        cudaFree(cuda_temp_in);
        cudaFree(cuda_temp_out);
        throw std::runtime_error("cudaMemcpy HostToDevice failed: " + std::string(cudaGetErrorString(err_memcpy)));
    }

    // 确定哪个张量持有“当前”数据以供后续操作使用
    // `current_data_dev` 指向下一步（归一化或转置）的输入
    // `next_output_dev` 指向下一步的输出
    float* current_data_dev = cuda_temp_in;
    float* next_output_dev = cuda_temp_out;

    // 5. 如果提供了 mean/stdv 参数则进行归一化
    if (!mean.empty() && !stdv.empty()) {
        // 检查 mean 和 stdv 的大小是否与通道数匹配
        if (mean.size() != target_channels || stdv.size() != target_channels) {
            cudaFree(cuda_temp_in);
            cudaFree(cuda_temp_out);
            throw std::runtime_error("Mean and standard deviation vectors must have size equal to channel count.");
        }

        // image[i] = (image[i] - mean[i]) / stdv[i];
        sctImageSubC3(current_data_dev, next_output_dev,
            target_height, target_width, mean[0], mean[1], mean[2]);
        std::swap(current_data_dev, next_output_dev);

        sctImageDivC3(current_data_dev, next_output_dev,
            target_height, target_width, stdv[0], stdv[1], stdv[2]);
        std::swap(current_data_dev, next_output_dev);

    } else if (!mean.empty() || !stdv.empty()) {
        // 警告：仅提供了 mean 或 stdv 其中之一，跳过完整归一化
        std::cerr << "Warning: Only one of mean or stdv is provided for normalization. Skipping full normalization." << std::endl;
    }

    // 6. 将 HWC 格式（current_data_dev）转置为 CHW 格式（next_output_dev）
    // sctPermute3D_v2 函数的参数为输入、输出、H、W、C 以及置换轴顺序。
    // HWC (0, 1, 2) -> CHW (2, 0, 1)
    sctImagePermute(current_data_dev, next_output_dev,
        target_height, target_width, target_channels,
        2, 0, 1);

    // 转置后，next_output_dev 现在持有最终的 CHW 数据。
    // 再次交换，使 current_data_dev 指向最终结果，便于后续统一清理
    std::swap(current_data_dev, next_output_dev); // current_data_dev 现在指向 CHW 结果

    // 7. 将最终结果从 `current_data_dev` 拷贝到用户提供的 `device_ptr`
    cudaError_t err_final_memcpy = cudaMemcpy(device_ptr, current_data_dev, total_size * sizeof(float), cudaMemcpyDeviceToDevice);
    if (err_final_memcpy != cudaSuccess) {
        cudaFree(cuda_temp_in); // 抛出异常前释放两个临时内存，因为它们可能已被交换
        cudaFree(cuda_temp_out);
        throw std::runtime_error("cudaMemcpy DeviceToDevice failed: " + std::string(cudaGetErrorString(err_final_memcpy)));
    }

    // 8. 释放临时 CUDA 内存
    cudaFree(cuda_temp_in);
    cudaFree(cuda_temp_out);
}
