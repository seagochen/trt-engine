//
// Created by vipuser on 25-1-6.
//

#include <simple_cuda_toolkits/vision/normalization.h>
#include <simple_cuda_toolkits/tsutils/permute_3D.h>
#include <simple_cuda_toolkits/bbox/bbox.h>
#include <simple_cuda_toolkits/bbox/suppress.h>
#include <simple_cuda_toolkits/matrix/matrix.h>
#include <simple_cuda_toolkits/tensor_utils.hpp>

#include "serverlet/utils/logger.h"
#include "serverlet/models/yolo/infer_yolo_pose.h"

#define DEBUG 0


InferYoloV8Pose::InferYoloV8Pose(const std::string& engine_path,
                                 const std::string& input_name,
                                 const std::vector<int>& input_shape, // [batch, channels, height, width]
                                 const std::string& output_name,
                                 const std::vector<int>& output_shape): // [batch, features, samples]
        InferModelBase(engine_path,
                       input_name,
                       input_shape,
                       output_name,
                       output_shape) {
    // Allocate images for resizing and normalization
    g_int_inputWidth = input_shape[3];
    g_int_inputHeight = input_shape[2];
    g_int_inputChannels = input_shape[1];
    g_int_maximumBatch = input_shape[0];
    g_cv_resizedImg = cv::Mat(g_int_inputHeight, g_int_inputWidth, CV_8UC3);
    g_cv_floatImg = cv::Mat(g_int_inputHeight, g_int_inputWidth, CV_32FC3);
    LOG_VERBOSE_TOPIC("InferYolov8Pose", "InferYolov8Pose", "InferYoloObjectv8 object created successfully.");

    // Allocate local buffer for output
    g_int_outputFeatures = output_shape[1];
    g_int_outputSamples = output_shape[2];
    g_vec_outputData = std::vector<float>(g_int_outputFeatures * g_int_outputSamples);
    LOG_VERBOSE_TOPIC("InferYolov8Pose", "InferYolov8Pose", "Local buffer allocated successfully.");

    // Create the input and output buffer for CUDA computation
    g_vec_inputBuffer.push_back(createZerosTensor<TensorType::FLOAT32>(g_int_inputChannels, g_int_inputHeight, g_int_inputWidth));
    g_vec_inputBuffer.push_back(createZerosTensor<TensorType::FLOAT32>(g_int_inputChannels, g_int_inputHeight, g_int_inputWidth));
    g_vec_outputBuffer.push_back(createZerosTensor<TensorType::FLOAT32>(g_int_outputFeatures, g_int_outputSamples));
    g_vec_outputBuffer.push_back(createZerosTensor<TensorType::FLOAT32>(g_int_outputFeatures, g_int_outputSamples));
    LOG_VERBOSE_TOPIC("InferYolov8Pose", "InferYolov8Pose", "Temporary buffer for CUDA computation allocated successfully.");
}


InferYoloV8Pose::~InferYoloV8Pose() {

    // Release local buffer for output
    g_vec_outputData.clear();
    LOG_VERBOSE_TOPIC("InferYolov8Pose", "~InferYolov8Pose", "Local buffer released successfully.");

    // Release the images
    g_cv_resizedImg.release();
    g_cv_floatImg.release();
    LOG_VERBOSE_TOPIC("InferYolov8Pose", "~InferYolov8Pose", "InferYolov8Pose object destroyed successfully.");
}


// Preprocess the image
void InferYoloV8Pose::preprocess(const cv::Mat& image, const int batchIdx) {

    // Resize the image to the input dimensions
    cv::resize(image, g_cv_resizedImg, cv::Size(g_int_inputWidth, g_int_inputHeight));

    // Convert the image to float data
    g_cv_resizedImg.convertTo(g_cv_floatImg, CV_32FC3);

    // Copy the float data to the cuda input buffer
    checkCudaErrors(cudaMemcpy(g_vec_inputBuffer[0].ptr(),    // dst
                               g_cv_floatImg.data,            // src
                               g_cv_floatImg.total() * g_cv_floatImg.elemSize(), // size
                               cudaMemcpyHostToDevice),      // direction
                    "Failed to copy data to the cuda input buffer."); // error message

#if DEBUG
    dumpCudaMemoryToCSV(g_vec_inputBuffer[0].ptr(), "in_original", g_int_inputWidth, g_int_inputHeight, g_int_inputChannels);
#endif

    // Permute the input tensor to [batch, channels, height, width]
    sctPermute3D(g_vec_inputBuffer[0].ptr(), // src
                 g_vec_inputBuffer[1].ptr(), // dst
                 g_int_inputWidth, g_int_inputHeight, g_int_inputChannels, // dimensions
                 2, 0, 1); // permutation

#if DEBUG
    dumpCudaMemoryToCSV(g_vec_inputBuffer[1].ptr(), "in_permuted", g_int_inputWidth, g_int_inputHeight, g_int_inputChannels);
#endif

    // Normalize the input tensor
    sctNormalizeData(g_vec_inputBuffer[1].ptr(), // src
                     g_vec_inputBuffer[0].ptr(), // dst
                     g_int_inputWidth, g_int_inputHeight, g_int_inputChannels); // dimensions

#if DEBUG
    dumpCudaMemoryToCSV(g_vec_inputBuffer[0].ptr(), "in_normalized", g_int_inputWidth, g_int_inputHeight, g_int_inputChannels);
#endif

    // Copy the normalized data to the cuda input buffer
    copyCudaDataToInputBuffer(g_vec_inputBuffer[0], batchIdx);
}


// Postprocess the output
std::vector<YoloPose> InferYoloV8Pose::postprocess(const int batchIdx, const float cls,
                                                   const float alpha, const float beta) {

    // Copy the output data from the cuda output buffer
    copyCudaDataFromOutputBuffer(g_vec_outputBuffer[0], batchIdx);

#if DEBUG
    dumpCudaMemoryToCSV(g_vec_outputBuffer[0].ptr(), "out_original", g_int_outputFeatures, g_int_outputSamples);
#endif

    // Transpose the output tensor to [samples, features]
    sctMatrixTranspose(g_vec_outputBuffer[0].ptr(), // src
                       g_vec_outputBuffer[1].ptr(), // dst
                       g_int_outputFeatures, // rows
                       g_int_outputSamples);   // cols

#if DEBUG
    dumpCudaMemoryToCSV(g_vec_outputBuffer[1].ptr(), "out_transposed", g_int_outputSamples, g_int_outputFeatures);
#endif

    // Suppress the bounding boxes
    sctSuppressResults_closed(g_vec_outputBuffer[1].ptr(), // src
                              g_vec_outputBuffer[0].ptr(), // dst
                              cls, // confidence of classification
                              g_int_outputFeatures, // rows
                              g_int_outputSamples); // cols

#if DEBUG
    dumpCudaMemoryToCSV(g_vec_outputBuffer[0].ptr(), "out_suppressed", g_int_outputSamples, g_int_outputFeatures);
#endif

    // Convert the bounding boxes from XYWH to XYXY
    sctXYWH2XYXY(g_vec_outputBuffer[0].ptr(), // src
                 g_vec_outputBuffer[1].ptr(), // dst
                 g_int_outputFeatures, // rows
                 g_int_outputSamples, // cols
                 alpha, // minimum size
                 beta); // minimum size

#if DEBUG
    dumpCudaMemoryToCSV(g_vec_outputBuffer[1].ptr(), "out_xyxy", g_int_outputSamples, g_int_outputFeatures);
#endif

    // Copy the data from CUDA device to host
    g_vec_outputBuffer[1].copyToVector(g_vec_outputData);

    // Decode the output data
    return decode(g_vec_outputData, g_int_outputFeatures, g_int_outputSamples);
}


std::vector<YoloPose> InferYoloV8Pose::decode(const std::vector<float>& vec_data,
                                              const int features, const int samples) {
    std::vector<YoloPose> results;

    for (int i = 0; i < samples; i++) {
        // 检查置信度，过滤低置信度的检测
        if (vec_data[i * features + 4] > 0.0) {
            // 创建一个 YoloPose 实例
            YoloPose result {
                    .lx = static_cast<int>(vec_data[i * features + 0]),
                    .ly = static_cast<int>(vec_data[i * features + 1]),
                    .rx = static_cast<int>(vec_data[i * features + 2]),
                    .ry = static_cast<int>(vec_data[i * features + 3]),
                    .conf = vec_data[i * features + 4]
            };

            // 关键点数据从第 5 个位置开始
            const int keypoint_start_index = i * features + 5;

            // 遍历每个关键点
            for (int j = 0; j < 17; j++) { // 假设有 17 个关键点
                // 每个关键点包含 3 个值（x, y, conf）
                const int keypoint_offset = keypoint_start_index + j * 3;

                // 创建一个 YoloPoint 实例
                YoloPoint pt {
                        .x = static_cast<int>(vec_data[keypoint_offset]),
                        .y = static_cast<int>(vec_data[keypoint_offset + 1]),
                        .conf = vec_data[keypoint_offset + 2]
                };

                // std::cout << "x: " << pt.x << " y: " << pt.y << " conf: " << pt.conf << std::endl;

                // 添加关键点到结果
                result.pts.push_back(pt);
            }

            // 将结果添加到列表中
            results.push_back(result);
        }
    }

    return results;
}