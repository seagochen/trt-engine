// Created by ubuntu on 9/5/24.

#include "cores/cores.h"
#include "yolo/yolov8_utils.h"
#include <processing/yolov8.h>
#include <cuda_runtime.h>
#include "tensor_utils.hpp"
#include "vision/colorspace.h"
#include "vision/hwc2chw.h"
#include "vision/normalization.h"

// Hide this tensor to improve speed
int g_width = 0, g_height = 0, g_channels = 0;
void *ptrCudaRawData = nullptr, *ptrCudaFloatData = nullptr;

auto g_gpu_tensor = createZerosCudaTensor<float>(84, 8400);
auto g_cpu_tensor = createZerosCpuTensor<float>(84, 8400);

// Memory initialization function
void initCudaTemporaryBuffer(int width, int height, int channels) {
    releaseCudaTemporaryBuffer();  // Clean up old buffers if needed
    checkCudaError(cudaMalloc(&ptrCudaRawData, width * height * channels * sizeof(uchar)), "init ptrCudaRawData failed");
    checkCudaError(cudaMalloc(&ptrCudaFloatData, width * height * channels * sizeof(float)), "init ptrCudaFloatData failed");

    g_width = width;
    g_height = height;
    g_channels = channels;
}

// Memory release function
void releaseCudaTemporaryBuffer() {
    if (ptrCudaRawData) cudaFree(ptrCudaRawData);
    if (ptrCudaFloatData) cudaFree(ptrCudaFloatData);
}

// Preprocessing function for input image
void preprocess(cv::Mat &image, CudaTensor<float> &output) {
    cudaMemcpy(ptrCudaRawData, image.data, g_width * g_height * g_channels * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaConvertUInt8ToFloat(ptrCudaRawData, g_width, g_height, g_channels, ptrCudaFloatData);
    cudaBGR2RGB(ptrCudaFloatData, g_width, g_height, g_channels, ptrCudaFloatData);
    cudaNormalizeData(ptrCudaFloatData, g_width, g_height, g_channels, ptrCudaFloatData);
    cudaHWC2CHW(ptrCudaFloatData, g_width, g_height, g_channels, output.ptr());
}

// Postprocess helper to extract box and confidence information
YoloResult extractYoloResult(const std::vector<float> &data, int index, int features) {
    YoloResult result;
    result.lx = static_cast<int>(data[index * features]);
    result.ly = static_cast<int>(data[index * features + 1]);
    result.rx = static_cast<int>(data[index * features + 2]);
    result.ry = static_cast<int>(data[index * features + 3]);
    result.cls = static_cast<int>(data[index * features + 5]);
    result.conf = data[index * features + 4];
    return result;
}

// Object detection postprocessing
void obj_postprocess(CudaTensor<float> &output, float confidence, std::vector<YoloResult> &results) {
    cudaYolov8ObjectPostProcessing(output.ptr(), 84, 8400, confidence, g_gpu_tensor.ptr());
    g_cpu_tensor.copyFrom(g_gpu_tensor);
    const std::vector<float> &data = g_cpu_tensor.getData();

    results.clear();
    for (int i = 0; i < 8400; ++i) {
        if (data[i * 84 + 4] > confidence) {
            results.push_back(extractYoloResult(data, i, 84));
        }
    }
}

// Pose postprocessing with keypoints
void pose_postprocess(CudaTensor<float> &output, float confidence, std::vector<YoloResult> &results) {
    cudaYolov8PosePostProcessing(output.ptr(), 56, 8400, confidence, g_gpu_tensor.ptr());
    g_cpu_tensor.copyFrom(g_gpu_tensor);
    const std::vector<float> &data = g_cpu_tensor.getData();

    results.clear();
    for (int i = 0; i < 8400; ++i) {
        if (data[i * 56 + 4] > confidence) {
            YoloResult pose_result = extractYoloResult(data, i, 56);

            // Keypoints extraction
            for (int j = 0; j < 17; ++j) {
                YoloPoint keypoint{
                        static_cast<int>(data[i * 56 + 5 + j * 3]),
                        static_cast<int>(data[i * 56 + 5 + j * 3 + 1]),
                        data[i * 56 + 5 + j * 3 + 2]
                };
                pose_result.keypoints.push_back(keypoint);
            }
            results.push_back(pose_result);
        }
    }
}
