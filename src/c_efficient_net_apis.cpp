//
// Created by user on 4/23/25.
//

#include "serverlet/c_efficient_net_apis.h"
#include "serverlet/models/efficient_net/efficient_net.h"

#include "serverlet/utils/logger.h"


// EfficientNetForFeatAndClassification 对象的指针
EfficientNetForFeatAndClassification* vptr_efficient_net = nullptr;

// 存储数据结果
float* vptr_result = nullptr;
int result_size = 0;

#ifdef __cplusplus
extern "C" {
#endif

void c_efficient_net_init(const char* engine_file_path, int maximum_batch_size) {

    // 如果vptr_efficient_net为nullptr，说明没有初始化
    if (vptr_efficient_net == nullptr) {
        vptr_efficient_net = new EfficientNetForFeatAndClassification(engine_file_path, maximum_batch_size);
        LOG_INFO("c_efficient_net_init", "EfficientNet model initialized successfully.");
    } else {

        // 如果已经初始化，delete之前的对象
        delete vptr_efficient_net;

        // 创建新的 EfficientNetForFeatAndClassification 对象
        vptr_efficient_net = new EfficientNetForFeatAndClassification(engine_file_path, maximum_batch_size);
        LOG_INFO("c_efficient_net_init", "EfficientNet model re-initialized successfully.");
    }
}

void c_efficient_net_release() {
    // 释放 EfficientNetForFeatAndClassification 对象
    if (vptr_efficient_net != nullptr) {
        delete vptr_efficient_net;
        vptr_efficient_net = nullptr;
        LOG_INFO("c_efficient_net_release", "EfficientNet model released successfully.");
    } else {
        LOG_WARNING("c_efficient_net_release", "EfficientNet model is already released.");
    }

    // 释放结果数据
    if (vptr_result != nullptr) {
        delete[] vptr_result;
        vptr_result = nullptr;
        result_size = 0;
        LOG_INFO("c_efficient_net_release", "Result data released successfully.");
    } else {
        LOG_WARNING("c_efficient_net_release", "Result data is already released.");
    }
};

bool c_efficient_net_add_image(int n_index, byte* cstr, int n_channels, int n_width, int n_height) {

    // Check if the model is initialized
    if (vptr_efficient_net == nullptr) {
        LOG_ERROR("c_efficient_net_add_image", "EfficientNet model is not initialized.");
        return false;
    }

    // 将图像数据转换为 OpenCV Mat 对象
    cv::Mat image(n_height, n_width, CV_8UC3, cstr);

    // 将图像数据添加到模型中
    vptr_efficient_net->preprocess(image, n_index);

    return true;
};

bool c_efficient_net_inference() {
    // Check if the model is initialized
    if (vptr_efficient_net == nullptr) {
        LOG_ERROR("c_efficient_net_inference", "EfficientNet model is not initialized.");
        return false;
    }

    // 执行推理
    vptr_efficient_net->inference();
};

float* c_efficient_net_get_result(int n_index, int* n_size) {

    // Check if the model is initialized
    if (vptr_efficient_net == nullptr) {
        LOG_ERROR("c_efficient_net_get_result", "EfficientNet model is not initialized.");
        return nullptr;
    }

    // 获取推理结果
    auto vptr_result = vptr_efficient_net->postprocess(n_index);
    if (vptr_result.empty()) {
        LOG_ERROR("c_efficient_net_get_result", "EfficientNet model result is empty.");
        *n_size = -1;
        return nullptr;
    }

    // 获取结果大小
    *n_size = vptr_result.size();
  
    // 如果结果不为空，将数据复制到 vptr_result，并返回该指针
    if (vptr_result != nullptr && result_size != *n_size) {
        delete[] vptr_result;
        vptr_result = new float[*n_size];
        result_size = *n_size;
    }
    
    // 拷贝数据到 vptr_result
    for (int i = 0; i < *n_size; ++i) {
        vptr_result[i] = vptr_result[i];
    }
    return vptr_result;
};

#ifdef __cplusplus
};
#endif