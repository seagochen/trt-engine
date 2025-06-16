//
// Created by user on 6/13/25.
//

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include <simple_cuda_toolkits/tsutils/permute_3D.h>
#include <simple_cuda_toolkits/vision/normalization.h>

#include "serverlet/models/common/image_to_tensor.h"
#include <stdexcept> // For std::runtime_error

void sct_image_to_cuda_tensor(
    cv::Mat image,
    float* device_ptr,
    std::vector<int> target_dims,
    bool is_rgb,
    const std::vector<float>& mean,
    const std::vector<float>& stdv)
{
    // Basic input validation
    if (image.empty()) {
        throw std::runtime_error("Input image is empty.");
    }
    if (device_ptr == nullptr) {
        throw std::runtime_error("Device pointer is null. Please ensure it is allocated before calling this function.");
    }
    if (target_dims.size() != 3) {
        throw std::runtime_error("target_dims must have 3 elements: [H, W, C].");
    }

    auto height = target_dims[0];
    auto width = target_dims[1];
    auto channels = target_dims[2];
    size_t total_size = static_cast<size_t>(height) * width * channels;

    // Allocate temporary CUDA memory
    // We need two temporary tensors: one for input data to normalization/permute, one for output
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

    // 1. Resize image to target dimensions
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    if (resized.empty()) {
        cudaFree(cuda_temp_in);
        cudaFree(cuda_temp_out);
        throw std::runtime_error("Failed to resize image.");
    }
    // Check channel count match
    if (resized.channels() != channels) {
        cudaFree(cuda_temp_in);
        cudaFree(cuda_temp_out);
        throw std::runtime_error("Resized image channel count (" + std::to_string(resized.channels()) + 
                                 ") does not match target dimensions' channels (" + std::to_string(channels) + ").");
    }

    // 2. Convert BGR to RGB if required and channels is 3
    if (is_rgb && channels == 3) {
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    }

    // 3. Convert image to float type and normalize to [0, 1]
    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32FC3, 1.0f / 255.0f); // CV_32FC3 assumes 3 channels for output, but `resized.channels()` matters

    // 4. Copy image data to temporary CUDA memory (cuda_temp_in)
    cudaError_t err_memcpy = cudaMemcpy(cuda_temp_in, floatImg.data, total_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err_memcpy != cudaSuccess) {
        cudaFree(cuda_temp_in);
        cudaFree(cuda_temp_out);
        throw std::runtime_error("cudaMemcpy HostToDevice failed: " + std::string(cudaGetErrorString(err_memcpy)));
    }

    // Determine which tensor holds the "current" data for subsequent operations
    // `current_data_dev` will point to the input for the next step (normalization or permute)
    // `next_output_dev` will point to the output for the next step
    float* current_data_dev = cuda_temp_in;
    float* next_output_dev = cuda_temp_out;

    // 5. Apply mean/stdv normalization if parameters are provided
    // Check if both mean and stdv are provided and non-empty
    if (!mean.empty() && !stdv.empty()) {
        // Ensure mean and stdv vectors match channel count
        if (mean.size() != channels || stdv.size() != channels) {
            cudaFree(cuda_temp_in);
            cudaFree(cuda_temp_out);
            throw std::runtime_error("Mean and standard deviation vectors must have size equal to channel count.");
        }
        // Normalize: current_data_dev -> next_output_dev
        sctNormalizeMeanStd(current_data_dev, next_output_dev, height, width, channels, mean.data(), stdv.data());
        
        // Swap roles for the next step: next_output_dev becomes current_data_dev
        std::swap(current_data_dev, next_output_dev);
    } else if (!mean.empty() || !stdv.empty()) {
        // Warning if only one of mean/stdv is provided
        std::cerr << "Warning: Only one of mean or stdv is provided for normalization. Skipping full normalization." << std::endl;
        // In this case, `current_data_dev` remains `cuda_temp_in` and `next_output_dev` remains `cuda_temp_out`
        // as no normalization was effectively applied that swaps them.
    }
    // If mean and stdv are both empty, normalization is skipped, `current_data_dev` remains `cuda_temp_in`

    // 6. Permute HWC format (current_data_dev) to CHW format (next_output_dev)
    // The sctPermute3D_v2 function takes input, output, H, W, C, and permutation axes.
    // HWC (0, 1, 2) -> CHW (2, 0, 1)
    sctPermute3D_v2(current_data_dev, next_output_dev, height, width, channels, 2, 0, 1);
    
    // After permute, next_output_dev now holds the final CHW data.
    // Swap again so current_data_dev points to the final result, for consistent cleanup
    std::swap(current_data_dev, next_output_dev); // current_data_dev now points to the CHW result

    // 7. Copy the final result from `current_data_dev` to the user-provided `device_ptr`
    cudaError_t err_final_memcpy = cudaMemcpy(device_ptr, current_data_dev, total_size * sizeof(float), cudaMemcpyDeviceToDevice);
    if (err_final_memcpy != cudaSuccess) {
        cudaFree(cuda_temp_in); // Free both temps before throwing, as they might have been swapped
        cudaFree(cuda_temp_out);
        throw std::runtime_error("cudaMemcpy DeviceToDevice failed: " + std::string(cudaGetErrorString(err_final_memcpy)));
    }

    // 8. Free temporary CUDA memory
    cudaFree(cuda_temp_in); // This will free the one that current_data_dev (or next_output_dev) pointed to after the last swap
    cudaFree(cuda_temp_out); // This will free the other one
}