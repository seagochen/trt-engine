// infer_efficientnet.hpp
#ifndef INFER_EFFICIENTNET_HPP
#define INFER_EFFICIENTNET_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "serverlet/models/infer_model_multi.h"
#include "serverlet/models/common/generic_image_to_tensor.h"
#include "serverlet/utils/logger.h"


class EfficientNetForFeatAndClassification final : public InferModelBaseMulti {
public:
    /**
     * @brief Constructor for EfficientNetForFeatAndClassification.
     * @param engine_path Path to the TensorRT engine file.
     * @param maximum_batch Maximum batch size (default is 1), must be between 1 and 8.
     */
    explicit EfficientNetForFeatAndClassification(const std::string& engine_path, int maximum_batch = 1);

    /**
     * @brief Preprocesses a single OpenCV image and uploads it to the GPU.
     * @param image Input image in cv::Mat format.
     * @param batchIdx Index of the batch to which this image belongs.
     */
    void preprocess(const cv::Mat& image, int batchIdx) override;

    /**
     * @brief Postprocesses the model's output, reading features and logits, and returning them combined.
     * The results are stored in the provided std::any container.
     * @param batchIdx Index of the batch to process.
     * @param args Additional arguments for postprocessing (not used here).
     * @param results_out Output container for the processed results.
     */
    void postprocess(int batchIdx, const std::map<std::string, std::any>& args, std::any& results_out) override;


private:
    // Using 'm_' prefix for member variables for clarity, common in C++ style guides
    int maximum_batch;
    int image_width;
    int image_height;
    int image_channels;
};


inline EfficientNetForFeatAndClassification::EfficientNetForFeatAndClassification(
    const std::string& engine_path,
    int maximum_batch)
    : InferModelBaseMulti(
        engine_path,
        std::vector<TensorDefinition>{{"input", {maximum_batch, 3, 224, 224}}},
        std::vector<TensorDefinition>{
            {"logits", {maximum_batch, 2}},
            {"feat",   {maximum_batch, 256}}
        }),
      maximum_batch(maximum_batch),
      image_width(224),
      image_height(224),
      image_channels(3)
{
    // 如果所有初始化都在初始化列表中完成，构造函数体可以为空
    LOG_VERBOSE_TOPIC("EfficientNetFeatCls", "constructor", "EfficientNetForFeatAndClassification 实例已创建。");
}

inline void EfficientNetForFeatAndClassification::preprocess(const cv::Mat& image, int batchIdx) {
    if (batchIdx >= maximum_batch) {
        LOG_ERROR("EfficientNetFeatCls", "Preprocess: batchIdx (" + std::to_string(batchIdx) + ") >= maximumBatch (" + std::to_string(maximum_batch) + ")");
        return;
    }

    // EfficientNet 的标准化参数
    // 如果始终相同，可以设为 static const 成员，避免每次调用都创建
    const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    const std::vector<float> stdv = {0.229f, 0.224f, 0.225f};

    // 访问当前 batch 索引的 CUDA 缓冲区
    auto cuda_device_ptr = const_cast<float*>(accessCudaBufByBatchIdx("input", batchIdx));
    if (cuda_device_ptr == nullptr) {
        LOG_ERROR("EfficientNetFeatCls", "Preprocess: 无法访问 batchIdx " + std::to_string(batchIdx) + " 的输入 CUDA 缓冲区");
        return;
    }

    // 转换并拷贝图像到 CUDA 设备，同时进行 BGR 到 RGB 转换和归一化
    imageToCudaTensor(
        image,              // 输入图像
        cuda_device_ptr,    // CUDA 设备指针
        image_height,       // 目标高度
        image_width,        // 目标宽度
        image_channels,     // 目标通道数
        true,               // 执行 BGR 到 RGB 转换
        mean,               // 归一化均值
        stdv                // 归一化标准差
    );
}


inline void EfficientNetForFeatAndClassification::postprocess(int batchIdx, const std::map<std::string, std::any>& args, std::any& results_out) {

    // 不需要 args 参数，这里可以忽略
    (void)args; // 避免未使用参数的警告

    // 输出类容
    std::vector<float> result;

    if (batchIdx >= maximum_batch) {
        LOG_ERROR("EfficientNetFeatCls", "Postprocess: batchIdx (" + std::to_string(batchIdx) + ") >= maximumBatch (" + std::to_string(maximum_batch) + ")");
        return;
    }

    std::vector<float> feats;
    std::vector<float> types;

    // 从 CUDA 输出缓冲区拷贝数据到 CPU 向量
    // 这些函数应自动处理 feats 和 types 的内存分配
    copyCpuDataFromOutputBuffer("feat", feats, batchIdx);
    copyCpuDataFromOutputBuffer("logits", types, batchIdx);

    // 确保 types 向量至少有 2 个元素以安全访问
    if (types.size() < 2) {
        LOG_ERROR("EfficientNetFeatCls", "Postprocess: batchIdx " + std::to_string(batchIdx) + " 的 logits 向量元素少于 2 个");
        return;
    }

    // 判断最大 logit 分数对应的类别
    int maxIndex = (types[0] > types[1]) ? 0 : 1;

    // 构造最终结果向量：[maxIndex, feat0, feat1, ..., featN]
    result.reserve(1 + feats.size()); // 预留空间避免多次分配
    result.push_back(static_cast<float>(maxIndex)); // 第一个元素为预测类别索引
    result.insert(result.end(),
                  feats.begin(),
                  feats.end()); // 追加所有特征元素


    // 将结果存储到 std::any 中
    results_out = result;
}

#endif //INFER_EFFICIENTNET_HPP