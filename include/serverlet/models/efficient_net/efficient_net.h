//
// Created by user on 4/22/25.
//

#ifndef INFER_EFFICIENTNET_H
#define INFER_EFFICIENTNET_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "serverlet/models/infer_model_multi.h"

class EfficientNetForFeatAndClassification final : public InferModelBaseMulti {
public:
    // engine_path: TensorRT 引擎文件路径
    // maximum_batch: 最大 batch 大小（>=1 && <=4）
    [[maybe_unused]] explicit EfficientNetForFeatAndClassification(
            const std::string& engine_path,
            int maximum_batch = 1);

    ~EfficientNetForFeatAndClassification() override;

    // 把单张 OpenCV 图像预处理并上传到 GPU
    void preprocess(const cv::Mat& image, int batchIdx) override;

    // 推理完成后读取 feature + logits，并合并返回
    std::vector<float> postprocess(int batchIdx);

private:
    int g_int_maximumBatch;
    int g_int_inputWidth;
    int g_int_inputHeight;
    int g_int_inputChannels;

    // Host 端临时存储
    std::vector<float> g_vec_inputData;   // 大小 = C*H*W
    std::vector<float> g_vec_featData;    // 大小 = feature_dim(256)
    std::vector<float> g_vec_classData;   // 大小 = class_dim(2)

    // // 合并 feature + class 向量
    // static std::vector<float> decode(
    //         const std::vector<float>& vec_feat,
    //         const std::vector<float>& vec_class);
};


#endif //INFER_EFFICIENTNET_H
