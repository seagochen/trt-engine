//
// Created by user on 4/22/25.
//
#include "serverlet/models/efficient_net/infer_efficient_net.h"
#include "serverlet/utils/logger.h"
#include "serverlet/utils/image_utils.h"

#include "simple_cuda_toolkits/tsutils/convert.h"

#include <vector>


EfficientNetForFeatAndClassification::EfficientNetForFeatAndClassification(
    const std::string& engine_path,
    int maximum_batch) : InferModelBaseMulti(
        engine_path,   // 引擎路径
        std::vector<TensorDefinition>{{"input", {maximum_batch, 3, 224, 224}}},
        std::vector<TensorDefinition>{
                {"logits", {maximum_batch, 2}},
                {"feat",   {maximum_batch, 256}}
        }) {

    // Allocate images for resizing and normalization
    g_int_maximumBatch = maximum_batch;
    g_int_inputWidth = 224;
    g_int_inputHeight = 224;
    g_int_inputChannels = 3;
    LOG_VERBOSE_TOPIC("EfficientNetForFeatAndClassification", "constructor", "EfficientNetForFeatAndClassification created");
}


void EfficientNetForFeatAndClassification::preprocess(const cv::Mat& image, int batchIdx) {
    // 1) 边界检查
    if (batchIdx >= g_int_maximumBatch) {
        LOG_ERROR("EfficientNet", "batchIdx >= g_int_maximumBatch");
        return;
    }

    // 2) 标准化参数
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> stdv = {0.229f, 0.224f, 0.225f};

    // 3) 查询本次要使用的 CUDA buffer
    auto cuda_buffer = accessCudaBufByBatchIdx("input", batchIdx);

    // 4) 将 const float* cuda_buffer 转换为 float* 类型
    auto cuda_buffer_float = const_cast<float*>(cuda_buffer);
    if (cuda_buffer_float == nullptr) {
        LOG_ERROR("EfficientNet", "Failed to access CUDA buffer for input");
        return;
    }

    // 4) 转换图片并拷贝到CUDA设备中
    cvtImgToCudaTensor(image, cuda_buffer_float, {g_int_inputHeight, g_int_inputWidth, g_int_inputChannels},
        mean, stdv, true);
}


std::vector<float> EfficientNetForFeatAndClassification::postprocess(int batchIdx) {
    // 如果batchIdx >= g_int_maximumBatch，则不处理数据
    if (batchIdx >= g_int_maximumBatch) {
        LOG_ERROR("EfficientNet", "batchIdx >= g_int_maximumBatch");
        return {};
    }

    // 存储计算结果
    std::vector<float> feats;   // 当计算完成后，模型的计算结果会拷贝到这里，该变量用于存储被识别的人物特征
    std::vector<float> types;   // 当计算完成后，模型的计算结果会拷贝到这里，该变量用于存储被识别的人物类别

    // 1) 读回特征和分类 logits
    copyCpuDataFromOutputBuffer("feat",   feats,  batchIdx);
    copyCpuDataFromOutputBuffer("logits", types, batchIdx);

    // 2) 找到 logits 中最大值的下标
    auto val1 = types[0];
    auto val2 = types[1];
    int maxIndex = (val1 > val2) ? 0 : 1;

    // 3) 构造最终返回： [maxIndex, feat0, feat1, ..., featN]
    std::vector<float> result;
    result.reserve(1 + feats.size());
    result.push_back(static_cast<float>(maxIndex));  // result[0]
    result.insert(result.end(),                             
                  feats.begin(),
                  feats.end());                  // 后面都是特征

    return result;
}

