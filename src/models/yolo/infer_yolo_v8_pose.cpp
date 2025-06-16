//
// Created by vipuser on 25-1-6.
//

#include <simple_cuda_toolkits/vision/normalization.h>
#include <simple_cuda_toolkits/tsutils/permute_3D.h>
#include <simple_cuda_toolkits/tensor_utils.hpp>

#include <vector>
#include "serverlet/utils/logger.h"
#include "serverlet/models/yolo/infer_yolo_v8.h"
#include "serverlet/models/yolo/nms.hpp"
#include "serverlet/models/yolo/yolo_post_proc.h"
#include "../../../include/serverlet/models/common/image_to_tensor.h"


InferYoloV8Pose::InferYoloV8Pose(
        const std::string& engine_path,
        int maximum_batch): InferModelBaseMulti(engine_path,
            std::vector<TensorDefinition> {{"images", {maximum_batch, 3, 640, 640}}},
            std::vector<TensorDefinition> {{"output0", {maximum_batch, 56, 8400}}}) {

    g_int_inputWidth = 640;
    g_int_inputHeight = 640;
    g_int_inputChannels = 3;
    g_int_maximumBatch = maximum_batch;
    g_int_outputFeatures = 56;
    g_int_outputSamples = 8400;

    // Initialize the output buffer
    g_vec_output.resize(g_int_maximumBatch * g_int_outputFeatures * g_int_outputSamples, 0.0f); 
}


InferYoloV8Pose::~InferYoloV8Pose() {

    // Release local buffer for output
    g_vec_output.clear();
    LOG_VERBOSE_TOPIC("InferYolov8Pose", "deconstructor", "Local buffer released successfully.");
}


// Preprocess the image
void InferYoloV8Pose::preprocess(const cv::Mat& image, const int batchIdx) {
    // 1) 边界检查
    if (batchIdx >= g_int_maximumBatch) {
        LOG_ERROR("EfficientNet", "batchIdx >= g_int_maximumBatch");
        return;
    }

    // 2) 标准化参数
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> stdv = {0.229f, 0.224f, 0.225f};

    // 3) 查询本次要使用的 CUDA buffer
    const float* cuda_device_ptr = accessCudaBufByBatchIdx("images", batchIdx);

    // 4) 将 const float* cuda_buffer 转换为 float* 类型
    auto cuda_buffer_float = const_cast<float*>(cuda_device_ptr);
    if (cuda_buffer_float == nullptr) {
        LOG_ERROR("EfficientNet", "Failed to access CUDA buffer for input");
        return;
    }

    // 4) 转换图片并拷贝到CUDA设备中
    sct_image_to_cuda_tensor(
        image,
        cuda_buffer_float,
        {g_int_inputHeight, g_int_inputWidth, g_int_inputChannels},
        false);
}


// Postprocess the output
std::vector<YoloPose> InferYoloV8Pose::postprocess(const int batchIdx, const float cls) {

    // 1) 边界检查
    if (batchIdx >= g_int_maximumBatch) {
        LOG_ERROR("EfficientNet", "batchIdx >= g_int_maximumBatch");
        return {};
    }

    // 2) 查询本次要使用的 CUDA buffer
    const float* cuda_device_ptr = accessCudaBufByBatchIdx("output0", batchIdx);

    // 3) 使用 sct_yolo_post_proc 处理输出
    int results = sct_yolo_post_proc(cuda_device_ptr, g_vec_output, g_int_outputFeatures, g_int_outputSamples, cls, true);
    if (results < 0) {
        LOG_ERROR("InferYoloV8Pose", "No results found after post-processing");
        return {};
    }

    //TODO
    std::cout << "Post-processing results: " << results << " detected poses." << std::endl;

    return {};
}
