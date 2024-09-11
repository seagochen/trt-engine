//
// Created by ubuntu on 9/12/24.
//

#ifndef JETSON_INFER_YOLOV8_H
#define JETSON_INFER_YOLOV8_H

#include <vector>
#include <opencv2/opencv.hpp>

#include <tensor_cuda.hpp>
#include "common/nlohmann/json.hpp"


struct YoloPoint {
    int x, y;
    float conf;
};

struct YoloResult {
    int lx, ly, rx, ry, cls;
    float conf;
    std::vector<YoloPoint> keypoints;
};

/**
 * @brief Initialize the CUDA temporary buffer
 *
 * @param width
 * @param height
 * @param channels
 */
void initCudaTemporaryBuffer(int width, int height, int channels=3);


/**
 * @brief Release the CUDA temporary buffer
 *
 */
void releaseCudaTemporaryBuffer();


/**
 * @brief Preprocess the image
 *
 * @param image
 * @param output
 */
void preprocess(cv::Mat &image, CudaTensor<float> &output);


/**
 * @brief Postprocess the output
 *
 * @param input
 * @param output
 * @param confidence
 * @param model
 */
void postprocess(const CudaTensor<float> &input, std::vector<YoloResult> &output,
                 float confidence, std::string model="yolov8");



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


#endif //JETSON_INFER_YOLOV8_H
