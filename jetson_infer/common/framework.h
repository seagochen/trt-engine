//
// Created by vipuser on 8/23/24.
//

#ifndef JETSON_INFER_FRAMEWORK_H
#define JETSON_INFER_FRAMEWORK_H

#include <NvInferRuntime.h>
#include <map>

#include "tensor_cuda.hpp"

std::map<std::string, CudaTensor<float>> loadTensorsFromModel(nvinfer1::ICudaEngine* engine);

void trackMemoryUsage();

#endif // JETSON_INFER_FRAMEWORK_H