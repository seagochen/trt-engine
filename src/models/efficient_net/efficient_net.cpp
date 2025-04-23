//
// Created by user on 4/22/25.
//
#include "serverlet/models/efficient_net/efficient_net.h"


EfficientNetForFeatAndClassification::EfficientNetForFeatAndClassification(const std::string &engine_path,
                           const std::vector<TensorDefinition> &input_ts_def,
                           const std::vector<TensorDefinition> &output_ts_def,
                           const int maximum_batch):
                           InferModelBaseMulti(engine_path, input_ts_def, output_ts_def) {

    // 记录模型最多同时处理的batch数
    m_int_maximumBatch = maximum_batch;

    //

}

EfficientNetForFeatAndClassification::~EfficientNetForFeatAndClassification() {

}

void EfficientNetForFeatAndClassification::preprocess(const cv::Mat &image, int batchIdx) {

}

std::vector<float> EfficientNetForFeatAndClassification::postprocess(int batchIdx) {
    return std::vector<float>();
}

std::vector<float> EfficientNetForFeatAndClassification::decode(const std::vector<float> &vec_feat, const std::vector<float> &vec_class) {
    return std::vector<float>();
}


