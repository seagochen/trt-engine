//
// Created by user on 6/25/25.
//

#ifndef CUDA_TENSOR_PROCESSOR_H
#define CUDA_TENSOR_PROCESSOR_H

#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @brief A class to encapsulate CUDA tensor processing, including image preprocessing,
 * data conversion, and asynchronous transfer to CUDA device memory.
 * This class pre-allocates temporary buffers for efficiency.
 */
class CudaTensorProcessor {
public:
    /**
     * @brief Constructs a CudaTensorProcessor and pre-allocates temporary CUDA memory.
     * @param target_height The target height of the image.
     * @param target_width The target width of the image.
     * @param target_channels The target number of channels (e.g., 3 for RGB).
     * @throws std::runtime_error if temporary CUDA memory allocation fails.
     */
    CudaTensorProcessor(int target_height, int target_width, int target_channels);

    /**
     * @brief Destructor: Frees temporary CUDA memory.
     */
    ~CudaTensorProcessor();

    /**
     * @brief Performs image preprocessing (resizing, color conversion), data conversion,
     * and copies data to a pre-allocated CUDA tensor asynchronously.
     * This method also handles normalization and HWC to CHW permutation on the GPU.
     *
     * @param image The input OpenCV Mat image (CPU).
     * @param device_ptr_out The pre-allocated CUDA device pointer where the final CHW tensor will be stored.
     * @param is_rgb A boolean indicating if BGR to RGB conversion is needed.
     * @param mean A vector of mean values for normalization (per channel).
     * @param std A vector of standard deviation values for normalization (per channel).
     * @throws std::runtime_error if input image is empty, output device pointer is null,
     */
    void transformImage(
        const cv::Mat& image,
        float* device_ptr_out,
        bool is_rgb,
        const std::vector<float>& mean,
        const std::vector<float>& std) const;

private:
    float* ptr_temp_in;  // Temporary CUDA memory for input to kernels (e.g., after H2D copy)
    float* ptr_temp_out; // Temporary CUDA memory for output from kernels (e.g., after normalization/permute)
    size_t total_size_elements; // Total number of elements (H * W * C)
    int target_height;
    int target_width;
    int target_channels;
};

#endif //CUDA_TENSOR_PROCESSOR_H
