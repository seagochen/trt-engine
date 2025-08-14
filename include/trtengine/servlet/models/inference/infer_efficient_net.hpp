// infer_efficientnet.hpp
#ifndef COMBINEDPROJECT_INFER_EFFICIENTNET_HPP
#define COMBINEDPROJECT_INFER_EFFICIENTNET_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "trtengine/servlet/models/infer_model_multi.h"
#include "trtengine/servlet/models/cuda_tensor_processor.h"
#include "trtengine/utils/logger.h"


class EfficientFeats final : public InferModelBaseMulti {
public:
    /**
     * @brief Constructor for EfficientNetForFeatAndClassification.
     * @param engine_path Path to the TensorRT engine file.
     * @param maximum_batch Maximum batch size (default is 1), must be between 1 and 8.
     */
    explicit EfficientFeats(const std::string& engine_path, int maximum_batch = 1);

    /**
     * @brief Destructor for EfficientNetForFeatAndClassification.
     */
    ~EfficientFeats() override;

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

    std::vector<float> vec_types;  // 用于存储类别 logits
    std::vector<float> vec_feats;  // 用于存储特征向量

    CudaTensorProcessor tensor_processor; // 用于图像预处理和数据转换
};


inline EfficientFeats::EfficientFeats(
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
      image_channels(3),
      tensor_processor(image_height, image_width, image_channels)  // 初始化 CudaTensorProcessor
{
    // 初始化输出缓冲区
    vec_feats.resize(256);
    vec_types.resize(2);

    // 如果所有初始化都在初始化列表中完成，构造函数体可以为空
    LOG_VERBOSE_TOPIC("EfficientNetFeatCls", "constructor", "EfficientNetForFeatAndClassification instance created successfully.");
}


inline EfficientFeats::~EfficientFeats() {
    vec_feats.clear();
    vec_types.clear();
    LOG_VERBOSE_TOPIC("EfficientNetForFeatAndClassification", "deconstructor", "Local buffer released successfully.");
}


inline void EfficientFeats::preprocess(const cv::Mat& image, int batchIdx) {
    if (batchIdx >= maximum_batch) {
        LOG_ERROR("EfficientNetFeatCls", "Preprocess: batchIdx (" + std::to_string(batchIdx) + ") >= maximumBatch (" + std::to_string(maximum_batch) + ").");
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

    // 使用 CudaTensorProcessor 进行图像预处理和数据转换
    tensor_processor.transformImage(
        image,              // 输入图像 (cv::Mat)
        cuda_device_ptr,    // CUDA 设备输出指针
        true,               // 执行 BGR 到 RGB 的转换
        mean,               // 归一化均值
        stdv                // 归一化标准差
    );
}


inline void EfficientFeats::postprocess(int batchIdx, const std::map<std::string, std::any>& args, std::any& results_out) {

    // 不需要 args 参数，这里可以忽略
    (void)args; // 避免未使用参数的警告

    // 初始化结果容器
    // 如果处理失败，results_out 将保持为空
    results_out = std::vector<float>(); // 初始化结果容器为一个空的 std::vector<float>

    // 输出类容
    std::vector<float> result;

    if (batchIdx >= maximum_batch) {
        LOG_ERROR("EfficientNetFeatCls", "Postprocess: batchIdx (" + std::to_string(batchIdx) + ") >= maximumBatch (" + std::to_string(maximum_batch) + ")");
        return;
    }

    // 从 CUDA 输出缓冲区拷贝数据到 CPU 向量
    // 这些函数应自动处理 feats 和 types 的内存分配
    copyCpuDataFromOutputBuffer("feat", vec_feats, batchIdx);
    copyCpuDataFromOutputBuffer("logits", vec_types, batchIdx);

    // 判断最大 logit 分数对应的类别
    int maxIndex = (vec_types[0] > vec_types[1]) ? 0 : 1;

    // 构造最终结果向量：[maxIndex, feat0, feat1, ..., featN]
    result.reserve(1 + vec_feats.size()); // 预留空间避免多次分配
    result.push_back(static_cast<float>(maxIndex)); // 第一个元素为预测类别索引
    result.insert(
        result.end(),    // 插入位置为结果向量的末尾
        vec_feats.begin(),  // 追加特征向量
        vec_feats.end()     // 结束特征向量
        );

    // 将结果存储到 std::any 中
    results_out = result;
}

#endif // COMBINEDPROJECT_INFER_EFFICIENTNET_HPP