//
// Created by xtcjj on 2025/08/14.
//

#include "trtengine/servlet/models/inference/model_init_helper.hpp" // For ModelFactory and YoloPose
#include "trtengine/c_apis/c_yolopose_detection.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <any>
#include <map>
#include <chrono>
#include <memory> // For std::unique_ptr

// 原子操作和 OpenMP 头文件
#include <atomic>
#include <omp.h> // 目前暂时关闭这部分的优化


#ifdef __cplusplus
extern "C" {
#endif


// Internal C++ context structure to hold model instances and parameters
struct YoloPoseContextImpl {
    std::unique_ptr<InferModelBaseMulti> pose_model;
    std::map<std::string, std::any> yolo_params;
};


#ifdef __cplusplus
};
#endif