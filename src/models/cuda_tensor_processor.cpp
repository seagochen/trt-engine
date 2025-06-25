//
// Created by user on 6/25/25.
//

#include <simple_cuda_toolkits/vision/image_permute.h> // 假设这些函数能够接受 cudaStream_t
#include <simple_cuda_toolkits/vision/image_ops.h>     // 假设这些函数能够接受 cudaStream_t

#include "trtengine/serverlet/models/cuda_tensor_processor.h"


// Implementations for CudaTensorProcessor
CudaTensorProcessor::CudaTensorProcessor(int target_height, int target_width, int target_channels)
    : target_height(target_height), target_width(target_width), target_channels(target_channels) {
    total_size_elements = static_cast<size_t>(target_height) * target_width * target_channels;
    size_t total_size_bytes = total_size_elements * sizeof(float);

    cudaError_t err_in = cudaMalloc(&ptr_temp_in, total_size_bytes);
    if (err_in != cudaSuccess) {
        throw std::runtime_error("Failed to allocate CUDA memory for d_temp_in_: " + std::string(cudaGetErrorString(err_in)));
    }

    cudaError_t err_out = cudaMalloc(&ptr_temp_out, total_size_bytes);
    if (err_out != cudaSuccess) {
        cudaFree(ptr_temp_in); // Clean up already allocated memory
        throw std::runtime_error("Failed to allocate CUDA memory for d_temp_out_: " + std::string(cudaGetErrorString(err_out)));
    }
}

CudaTensorProcessor::~CudaTensorProcessor() {
    // Synchronous free in destructor to ensure all operations on temporary buffers are complete.
    if (ptr_temp_in != nullptr) {
        cudaFree(ptr_temp_in);
    }
    if (ptr_temp_out != nullptr) {
        cudaFree(ptr_temp_out);
    }
}

void CudaTensorProcessor::transformImage(
    const cv::Mat& image,
    float* device_ptr_out,
    bool is_rgb,
    const std::vector<float>& mean,
    const std::vector<float>& stdv) const
{
    // Check input parameters
    if (image.empty()) {
        throw std::runtime_error("Input image is empty.");
    }
    if (device_ptr_out == nullptr) {
        throw std::runtime_error("Output device pointer is null. Please ensure it is allocated before calling this function.");
    }

    // Used for subsequent image processing variables (CPU side)
    cv::Mat processedImage;

    // --- Image size processing logic ---
    if (image.size() != cv::Size(target_width, target_height)) {
        cv::resize(image, processedImage, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);
        if (processedImage.empty()) {
            throw std::runtime_error("Failed to resize image.");
        }
    } else {
        image.copyTo(processedImage);
    }
    // --- Image size processing logic end ---

    // 2. If needed, convert image from BGR to RGB (CPU side)
    if (is_rgb && target_channels == 3) {
        cv::cvtColor(processedImage, processedImage, cv::COLOR_BGR2RGB);
    }

    // 3. Convert image to float format and normalize to [0, 1] range (CPU side)
    cv::Mat floatImg;
    processedImage.convertTo(floatImg, CV_32FC3, 1.0f / 255.0f);

    // 4. Copy image data from OpenCV Mat to CUDA device memory asynchronously
    // NOTE: For true non-blocking host-to-device copy, floatImg.data should ideally point to pinned host memory (cudaHostAlloc).
    // Assuming floatImg.data is pageable, cudaMemcpyAsync will still allow GPU kernel execution overlap.
    cudaError_t err_memcpy = cudaMemcpy(ptr_temp_in, floatImg.data, total_size_elements * sizeof(float), cudaMemcpyHostToDevice);
    if (err_memcpy != cudaSuccess) {
        throw std::runtime_error("cudaMemcpyHostToDeviceAsync failed: " + std::string(cudaGetErrorString(err_memcpy)));
    }

    float* current_data_dev = ptr_temp_in;
    float* next_output_dev = ptr_temp_out;

    // 5. If mean/stdv parameters are provided, perform normalization asynchronously
    if (!mean.empty() && !stdv.empty()) {
        if (mean.size() != target_channels || stdv.size() != target_channels) {
            throw std::runtime_error("Mean and standard deviation vectors must have size equal to channel count.");
        }

        sctImageSubC3(current_data_dev, next_output_dev,
    target_height, target_width, mean[0], mean[1], mean[2]);
        std::swap(current_data_dev, next_output_dev);

        // 假设 sctImageDivC3 函数支持传入 CUDA 流
        sctImageDivC3(current_data_dev, next_output_dev,
            target_height, target_width, stdv[0], stdv[1], stdv[2]);
        std::swap(current_data_dev, next_output_dev);

    } else if (!mean.empty() || !stdv.empty()) {
        std::cerr << "Warning: Only one of mean or stdv is provided for normalization. Skipping full normalization." << std::endl;
    }

    // 6. Transpose HWC format (current_data_dev) to CHW format (next_output_dev) asynchronously
    // The parameters for sctPermute3D_v2 function are input, output, H, W, C, and permutation axis order.
    // HWC (0, 1, 2) -> CHW (2, 0, 1)
    sctImagePermute(current_data_dev, next_output_dev,
        target_height, target_width, target_channels,
        2, 0, 1);
    std::swap(current_data_dev, next_output_dev);

    // 7. Copy the final result from `current_data_dev` to the user-provided `device_ptr_out` asynchronously
    cudaError_t err_final_memcpy = cudaMemcpy(device_ptr_out, current_data_dev, total_size_elements * sizeof(float), cudaMemcpyDeviceToDevice);
    if (err_final_memcpy != cudaSuccess) {
        throw std::runtime_error("cudaMemcpyDeviceToDeviceAsync failed: " + std::string(cudaGetErrorString(err_final_memcpy)));
    }
}
