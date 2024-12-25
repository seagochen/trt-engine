//
// Created by vipuser on 25-1-6.
//

#ifndef INFER_YOLO_V8_OBJ_H
#define INFER_YOLO_V8_OBJ_H

#include "common/models/infer_model_base.h"
#include "common/yolo/yolo_def.h"

#include <opencv2/opencv.hpp>

class InferYoloV8Obj final: public InferModelBase {
    int g_int_maximumBatch;     // Maximum number of batch

    int g_int_inputWidth;       // Input width
    int g_int_inputHeight;      // Input height
    int g_int_inputChannels;    // Input channels
    int g_int_outputFeatures;   // Number of output features
    int g_int_outputSamples;    // Number of output samples

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
    explicit InferYoloV8Obj(const std::string& engine_path,
                             const std::string& input_name,
                             const std::vector<int>& input_shape,
                             const std::string& output_name,
                             const std::vector<int>& output_shape);

    // Destructor
    ~InferYoloV8Obj() override;

    // Preprocess the image
    void preprocess(const cv::Mat& image, int batchIdx) override;

    // Postprocess the output
    std::vector<Yolo> postprocess(int batchIdx=0,
        float cls=0.4, float alpha=0.f, float beta=640.f);

    // Convert the class id to class name
    static std::vector<std::string> convertToLabel(const std::vector<Yolo>& vec_data, const std::string& label_path);

private:
    static std::vector<Yolo> decode(const std::vector<float>& vec_data, int features, int samples);
};

#endif //INFER_YOLO_V8_OBJ_H
