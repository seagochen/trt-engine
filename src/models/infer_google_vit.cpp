//
// Created by vipuser on 25-1-6.
//


#include <simple_cuda_toolkits/vision/normalization.h>
#include <simple_cuda_toolkits/tsutils/permute_3D.h>
#include <simple_cuda_toolkits/tensor_utils.hpp>

#include "common/utils/logger.h"
#include "common/utils/load_labels.h"
#include "common/models/infer_google_vit.h"


#define DEBUG 0


InferGoogleVit::InferGoogleVit(const std::string& engine_path,
                             const std::string& input_name,
                             const std::vector<int>& input_shape, // [batch, channels, height, width]
                             const std::string& output_name,
                             const std::vector<int>& output_shape): // [batch, features]
                                 InferModelBase(engine_path, input_name, input_shape, output_name, output_shape) {
    // Allocate images for resizing and normalization
    g_int_inputWidth = input_shape[3];
    g_int_inputHeight = input_shape[2];
    g_int_inputChannels = input_shape[1];
    g_cv_resizedImg = cv::Mat(g_int_inputHeight, g_int_inputWidth, CV_8UC3);
    g_cv_floatImg = cv::Mat(g_int_inputHeight, g_int_inputWidth, CV_32FC3);
    LOG_VERBOSE_TOPIC("InferHumanActionVit", "InferHumanActionVit", "InferHumanActionVit object created successfully.");

    // Allocate local buffer for output
    g_int_outputFeatures = output_shape[1];
    g_vec_outputData = std::vector<float>(g_int_outputFeatures);
    LOG_VERBOSE_TOPIC("InferHumanActionVit", "InferHumanActionVit", "Local buffer allocated successfully.");

    // Create the input and output buffer for CUDA computation
    g_vec_inputBuffer.push_back(createZerosTensor<TensorType::FLOAT32>(g_int_inputChannels, g_int_inputHeight, g_int_inputWidth));
    g_vec_inputBuffer.push_back(createZerosTensor<TensorType::FLOAT32>(g_int_inputChannels, g_int_inputHeight, g_int_inputWidth));
    g_vec_outputBuffer.push_back(createZerosTensor<TensorType::FLOAT32>(g_int_outputFeatures));
    LOG_VERBOSE_TOPIC("InferHumanActionVit", "InferHumanActionVit", "Temporary buffer for CUDA computation allocated successfully.");
}


InferGoogleVit::~InferGoogleVit() {
    // Release local buffer for output
    g_vec_outputData.clear();
    LOG_VERBOSE_TOPIC("InferHumanActionVit", "~InferHumanActionVit", "Local buffer released successfully.");

    // Release the images
    g_cv_resizedImg.release();
    g_cv_floatImg.release();
    LOG_VERBOSE_TOPIC("InferHumanActionVit", "~InferHumanActionVit", "InferHumanActionVit object destroyed successfully.");
}


// Preprocess the image
void InferGoogleVit::preprocess(const cv::Mat& image, const int batchIdx) {

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
std::vector<std::tuple<int, float>> InferGoogleVit::postprocess(const int batchIdx, const float cls, const int topk) {
    // Copy the output data from the cuda output buffer
    copyCudaDataFromOutputBuffer(g_vec_outputBuffer[0], batchIdx);

 #if DEBUG
         dumpCudaMemoryToCSV(g_vec_outputBuffer[0].ptr(), "out_original", g_int_outputFeatures);
 #endif

    // Copy the data from CUDA device to host
    g_vec_outputBuffer[0].copyToVector(g_vec_outputData);

    // Decode the output data
    return decode(g_vec_outputData, cls, topk);
}


std::vector<std::tuple<int, float>> InferGoogleVit::decode(const std::vector<float>& vec_data, const float cls, const int topk) {
    // 存储结果的索引
    std::vector<std::tuple<int, float>> results;

    // 索引数组，初始为 0, 1, 2, ..., vec_data.size() - 1
    std::vector<int> indices(vec_data.size());
    std::iota(indices.begin(), indices.end(), 0);

    // 根据 vec_data 的值对索引进行排序
    std::sort(indices.begin(), indices.end(), [&vec_data](int i1, int i2) {
        return vec_data[i1] > vec_data[i2];
    });

    // 提取 topk 个符合条件的索引
    for (int i = 0; i < topk && i < static_cast<int>(indices.size()); ++i) {
        if (vec_data[indices[i]] > cls) { // 过滤掉低于 cls 的值
//            results.push_back(indices[i]);

            // 将索引和值存入结果中
            results.push_back(std::make_tuple(indices[i], vec_data[indices[i]]));

#if DEBUG
            std::cout << "Index: " << indices[i] << ", Value: " << vec_data[indices[i]] << std::endl;
#endif
        }
    }

    return results;
}

std::vector<std::string> InferGoogleVit::convertToLabels(const std::vector<std::tuple<int, float>>& vec_data,
                                                              const std::string& label_path) {
    // Load the labels from the json file
    auto vec_labels = loadLabelsFromJson(label_path);

    // Convert the indices to labels
    std::vector<std::string> vec_results;
//    for (const int i : vec_data) {
//        vec_results.push_back(vec_labels[i]);
//    }
    for (const auto& [i, _] : vec_data) {
        vec_results.push_back(vec_labels[i]);
    }

    return vec_results;
};