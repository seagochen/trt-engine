#include <iostream>
#include <map>
#include <string>

#include <sys/resource.h>  // For memory usage tracking on Unix systems

#include "tensor_utils.hpp"
#include "common/engine_loader.h"
#include "common/framework.h"


// 从 TensorRT 引擎中加载张量
std::map<std::string, CudaTensor<float>> loadTensorsFromModel(nvinfer1::ICudaEngine* engine) {
    // 创建输入输出缓冲区
    std::map<std::string, CudaTensor<float>> buffers;

    // 获取模型的所有张量名字
    auto tensor_names = getTensorNamesFromModel(engine);

    for (const auto& name : tensor_names) {
        // 获取每个张量的大小
        TensorDimensions dims = getTensorDimsByName(engine, name, tensor_type::FLOAT32);

        // 创建一个 CudaTensor
        CudaTensor<float> tensor(dims);

        // 保存 Tensor 到缓冲区
        buffers[name] = std::move(tensor);

        // 输出一些信息
        std::cout << "Allocated buffer for Tensor: " << name
                  << " with size: " << buffers[name].memSize() << " bytes." << std::endl;
    }

    // 返回给上级调用者
    return buffers;
}


// Function to track memory usage on Unix systems
void trackMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    std::cout << "Memory usage: " << usage.ru_maxrss << " KB" << std::endl;
}