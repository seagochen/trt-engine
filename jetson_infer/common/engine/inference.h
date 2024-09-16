//
// Created by vipuser on 9/13/24.
//

#ifndef JETSON_INFER_INFERENCE_H
#define JETSON_INFER_INFERENCE_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <simple_cuda_toolkits/tensor.hpp>

#include "common/nlohmann/json.hpp"


struct YoloPoint {
    int x, y;
    float conf;
};

struct YoloResult {
    int lx, ly, rx, ry, cls;
    float conf;
    std::vector<YoloPoint> key_pts;
};


/**
 * @brief Preprocess the image
 *
 * @param image
 * @param output
 * @param inputDim
 */
#include <map>
void preprocess(cv::Mat &image, Tensor<float> &output);


/**
 * @brief Postprocess the output
 *
 * @param input
 * @param outputA¥  1`
 * @param confidence
 * @param model
 */
void postprocess(const Tensor<float> &input, std::vector<YoloResult> &output, float confidence);



// YoloPoint のシリアライズ関数
void to_json(const YoloPoint& p, nlohmann::json& j);


// YoloResult のシリアライズ関数
void to_json(const YoloResult& r, nlohmann::json& j);


// YoloResult のデシリアライズ関数
void from_json(const std::string &str, YoloResult &r);


// YoloPoint のデシリアライズ関数
void from_json(const std::string &str, YoloPoint &p);


// vector<YoloResult> のシリアライズ関数
std::string to_json(const std::vector<YoloResult>& results);


// vector<YoloResult> のデシリアライズ関数
void from_json(const std::string &str, std::vector<YoloResult> &results);



#endif //JETSON_INFER_INFERENCE_H
