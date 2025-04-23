//
// Created by user on 4/22/25.
//

#ifndef COMBINEDPROJECT_EFFICIENTNET_H
#define COMBINEDPROJECT_EFFICIENTNET_H

#include <opencv2/opencv.hpp>
#include "serverlet/models/infer_model_multi.h"


class EfficientNetForFeatAndClassification final: public InferModelBaseMulti {

    int m_int_maximumBatch;     // Maximum number of batch

    cv::Mat m_cv_resizedImg;    // Resized image for normalization
    cv::Mat m_cv_floatImg;      // Float data for normalization

    // vector of output data
    std::vector<float> m_vec_final_output_feat;     // for feature extraction
    std::vector<float> m_vec_final_output_class;    // for classification

    // vector of input temporary buffer
    std::vector<Tensor<float>> m_vec_input_buffs;

    // vector of output temporary buffer
    std::vector<Tensor<float>> m_vec_temp_output_feat;
    std::vector<Tensor<float>> m_vec_temp_output_class;

public:
    // Constructor
    explicit EfficientNetForFeatAndClassification(const std::string& engine_path,
                          const std::vector<TensorDefinition>& input_ts_def,
                          const std::vector<TensorDefinition>& output_ts_def,
                          const int maximum_batch = 1);

    // Destructor
    ~EfficientNetForFeatAndClassification() override;

    // Preprocess the input image
    void preprocess(const cv::Mat& image, int batchIdx) override;

    // Postprocess the output data
    std::vector<float> postprocess(int batchIdx);

private:
    static std::vector<float> decode(const std::vector<float>& vec_feat, const std::vector<float>& vec_class);
};


#endif //COMBINEDPROJECT_EFFICIENTNET_H
