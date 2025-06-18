//
// Created by user on 6/18/25.
//

#include <string>
#include <cuda_runtime.h>
#include <vector>

// Simple CUDA Toolkits 
#include <simple_cuda_toolkits/tsutils/filter.h>
#include <simple_cuda_toolkits/tsutils/sort.h>
#include <simple_cuda_toolkits/tsutils/maxmin.h>
#include <simple_cuda_toolkits/matrix/matrix.h>

// Serverlet headers
#include "serverlet/models/common/yolo_postprocess.h"
#include "serverlet/utils/logger.h"

// Define DEBUG for conditional compilation (consider moving to a global config)
#ifndef DEBUG
#define DEBUG 0
#endif


// --- Utility Function for YOLO Post-processing (CUDA-based) ---
int inferPostProcForYolo(
    const float* ptr_device,
    std::vector<float>& output,
    const int features,
    const int samples,
    const float cls,
    const int maximum,
    const bool use_pose
) {

    // 创建CUDA设备上的临时张量指针
    float* ptr_device_temp0 = nullptr;
    float* ptr_device_temp1 = nullptr;
    size_t total_size = (size_t)features * samples * sizeof(float);

    cudaError_t err0 = cudaMalloc(&ptr_device_temp0, total_size);
    if (err0 != cudaSuccess) {
        LOG_ERROR("sct_yolo_post_proc", "Failed to allocate CUDA memory for temp tensor 0: " + std::string(cudaGetErrorString(err0)));
        return -1; // Indicate failure
    }

    cudaError_t err1 = cudaMalloc(&ptr_device_temp1, total_size);
    if (err1 != cudaSuccess) {
        cudaFree(ptr_device_temp0); // Clean up previous allocation
        LOG_ERROR("sct_yolo_post_proc", "Failed to allocate CUDA memory for temp tensor 1: " + std::string(cudaGetErrorString(err1)));
        return -1; // Indicate failure
    }

    cudaMemcpy(ptr_device_temp0, ptr_device, total_size, cudaMemcpyDeviceToDevice);

#if DEBUG
    sctDumpCudaMemoryToCSV(ptr_device_temp0, "before_op.csv", features, samples);
#endif

    // 转置操作，将 [features, samples] 转换为 [samples, features]
    sctMatrixTranspose(ptr_device_temp0, ptr_device_temp1, features, samples);

#if DEBUG
    sctDumpCudaMemoryToCSV(ptr_device_temp1, "transpose.csv", samples, features);
#endif

    // 如果不是姿态模型，则执行分类处理（例如 argmax）
    if (!use_pose)
    {
        // 从分类概率中提取最大值的索引和对应的值
        sctArgmax_dim1(
            ptr_device_temp1,   // 输入的CUDA设备指针 [samples, features]，其中每一行内容为 [cx, cy, w, h, cls1_conf, cls2_conf, ...]
            ptr_device_temp0,   // 输出的CUDA设备指针 [samples, features]，其中每一行内容为 [cx, cy, w, h, conf, cls_idx]
            samples,            // dim0 - 样本数量
            features,           // dim1 - 特征数量
            4,                  // start_at_col_idx - 从第4列开始处理（通常是 [cx, cy, w, h]）
            features - 1,       // end_at_col_idx - 处理到最后一列
            4,                  // output_value_at_col_idx - 输出的值存储在第4列（通常是置信度）
            5                   // output_ind_at_col_idx - 输出的索引存储在第5列（通常是分类索引）
        );

        // 交换指针，ptr_device_temp1 现在包含 [cx, cy, w, h, conf, cls_idx] 的结果
        std::swap(ptr_device_temp0, ptr_device_temp1); 
    }

    // 根据置信度阈值过滤结果
    int results = sctFilterGreater_dim1(
        ptr_device_temp1, // 输入数据（已转置，可能包含分类信息）
        ptr_device_temp0, // 用于存放过滤后结果的输出缓冲区
        4,                // 置信度分数所在的维度索引（例如第4列，表示x,y,w,h,conf,...）
        cls,              // 置信度阈值
        samples,          // dim0 - 需要检查的样本总数
        features          // dim1 - 每个样本的特征数量
    ); // 过滤后，不符合要求的数据被全部置为0，但此时依然乱序

#if DEBUG
    sctDumpCudaMemoryToCSV(ptr_device_temp0, "filter.csv", samples, features);
#endif

    if (results > 0)
    {
        // 按置信度降序排序结果
        sctSortTensor_dim1_descending(ptr_device_temp0, ptr_device_temp1, samples, features, 4); 
        // 处理之后，有效结果排列靠前，之后可以使用 cvtXYWHCoordsToYolo / cvtXYWHCoordsToYoloPose 函数进行坐标转换

#if DEBUG
        sctDumpCudaMemoryToCSV(ptr_device_temp1, "sort.csv", samples, features);
#endif

        // 拷贝的时候，我们不需要拷贝全部的 sammples，而是只拷贝有效的 results 数量
        if (results > maximum) {
            results = maximum; // 限制结果数量不超过最大值
        }

        // 确保输出向量大小足够
        if (output.size() < results * features) {
            output.resize(results * features); 
        }

        // 将结果从设备内存拷贝到输出向量
        cudaMemcpy(output.data(), ptr_device_temp1, results * features * sizeof(float), cudaMemcpyDeviceToHost);

    } else {
        results = -1; // Indicate no valid results found
    }

    // Clean up CUDA memory
    cudaFree(ptr_device_temp0);
    cudaFree(ptr_device_temp1);

    return results; // Return number of valid results
}