//
// Created by vipuser on 25-1-6.
//

#ifndef INFER_HUMAN_ACTION_VIT_H
#define INFER_HUMAN_ACTION_VIT_H

#include "common/models/infer_model_base.h"

#include <opencv2/opencv.hpp>


class InferGoogleVit final: public InferModelBase {
    int g_int_inputWidth;       // Input width
    int g_int_inputHeight;      // Input height
    int g_int_inputChannels;    // Input channels

    int g_int_outputFeatures;   // Number of output features

    cv::Mat g_cv_resizedImg;    // Resized image for normalization
    cv::Mat g_cv_floatImg;      // Float data for normalization

    // vector of output data
    std::vector<float> g_vec_outputData;

    // vector of input temporary buffer
    std::vector<Tensor<float>> g_vec_inputBuffer;

    // vector of output temporary buffer
    std::vector<Tensor<float>> g_vec_outputBuffer;

public:
    // Constructor and destructor
    explicit InferGoogleVit(const std::string& engine_path,
                             const std::string& input_name,
                             const std::vector<int>& input_shape,
                             const std::string& output_name,
                             const std::vector<int>& output_shape);

    // Destructor
    ~InferGoogleVit() override;

    // Preprocess the image
    void preprocess(const cv::Mat& image, int batchIdx) override;

    // Postprocess the output
    std::vector<std::tuple<int, float>> postprocess(int batchIdx=0, float cls=0.4, int topk=1);

    // Convert the output to labels
    static std::vector<std::string> convertToLabels(const std::vector<std::tuple<int, float>>& vec_data,
        const std::string& label_path);

private: 
    static std::vector<std::tuple<int, float>> decode(const std::vector<float>& vec_data, float cls, int topk);
};


#endif //INFER_HUMAN_ACTION_VIT_H
