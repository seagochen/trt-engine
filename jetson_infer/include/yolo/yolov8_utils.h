//
// Created by ubuntu on 9/5/24.
//

#ifndef JETSON_INFER_YOLOV8_H
#define JETSON_INFER_YOLOV8_H

#include <vector>
#include <opencv2/opencv.hpp>

#include "tensor_cuda.hpp"


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
 * @brief Postprocess the output tensor
 * 
 * @param output
 * @param confidence
 * @param results
 */
void obj_postprocess(CudaTensor<float> &output, float confidence, std::vector<YoloResult> &results);

/**
 * @brief Postprocess the output tensor
 * 
 * @param output
 * @param confidence
 * @param results
 */
void pose_postprocess(CudaTensor<float> &output, float confidence, std::vector<YoloResult> &results);


#endif //JETSON_INFER_YOLOV8_H