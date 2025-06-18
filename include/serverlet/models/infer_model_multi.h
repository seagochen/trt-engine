//
// Created by user on 4/22/25.
//

#ifndef INFERENCE_INFER_MODEL_MULTI_H
#define INFERENCE_INFER_MODEL_MULTI_H

#include <string>
#include <vector>
#include <map>
#include <any>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "serverlet/trt_engine/trt_engine_multi.h"

// 前向声明：Tensor 模板类
template<typename T>
class Tensor;

/**
 * 定义输入/输出张量的名称与维度
 */
struct TensorDefinition {
    std::string name;
    std::vector<int> dims;  // e.g. {batch, C, H, W}
};

/**
 * 基于 TensorRT 的通用推理基类，支持多输入多输出
 */
class InferModelBaseMulti {
public:
    /**
     * @param engine_path  TensorRT 引擎文件路径
     * @param input_defs   多输入张量定义列表
     * @param output_defs  多输出张量定义列表
     */
    InferModelBaseMulti(
        const std::string& engine_path,
        const std::vector<TensorDefinition>& input_defs,
        const std::vector<TensorDefinition>& output_defs);

    virtual ~InferModelBaseMulti();

    /**
     * 用户需实现：将 cv::Mat 图像预处理并拷贝到指定输入张量
     * @param image    输入图像
     * @param batchIdx 批次索引
     */
    virtual void preprocess(const cv::Mat& image, int batchIdx) = 0;

    // 在 InferModelBaseMulti 中
    virtual void postprocess(int batchIdx, const std::map<std::string, std::any>& args, std::any& results_out) = 0;

    /**
     * 执行推理，多输入多输出统一处理
     * @return 推理是否成功
     */
    bool inference();

    /**
     * 拷贝 Host 数据到指定输入张量的 Device Buffer
     * @param tensor_name 输入张量名称
     * @param input_data  批次单样本数据
     * @param batch_idx   样本在批次中的索引
     */
    void copyCpuDataToInputBuffer(
        const std::string& tensor_name,
        const std::vector<float>& input_data,
        int batch_idx = 0);

    /**
     * 从指定输出张量的 Device Buffer 拷贝到 Host
     * @param tensor_name 输出张量名称
     * @param output_data 接收单样本数据的 vector（大小应匹配）
     * @param batch_idx   样本在批次中的索引
     */
    void copyCpuDataFromOutputBuffer(
        const std::string& tensor_name,
        std::vector<float>& output_data,
        int batch_idx = 0);

    /**
     * 获取指定输入张量的 Device 指针，可用于直接访问
     * @param tensor_name 输入张量名称
     * @param batch_idx   样本在批次中的索引
     * @return 返回 Device 指针
     */
    [[nodiscard]]
    const float* accessCudaBufByBatchIdx(
        const std::string& tensor_name,
        int batch_idx = 0) const;


protected:
    /**
     * 加载引擎并创建 ExecutionContext，设置动态输入形状
     */
    bool loadEngine(
        const std::string& engine_path,
        const std::vector<TensorDefinition>& input_defs,
        const std::vector<TensorDefinition>& output_defs);

    /**
     * 为所有输入和输出分配 Device Tensor Buffer
     */
    bool allocateBufForTrtEngine(
        const std::vector<TensorDefinition>& input_defs,
        const std::vector<TensorDefinition>& output_defs);

    TrtEngineMultiTs*                       g_ptr_engine      = nullptr;
    std::vector<TensorDefinition>           g_input_defs;         // 多输入定义
    std::vector<TensorDefinition>           g_output_defs;        // 多输出定义
    std::map<std::string, Tensor<float>>    g_map_trtTensors;     // TensorRT 张量映射, <name, Tensor>
    cudaStream_t                            g_stream           = nullptr;  // CUDA 流
};

#endif // INFERENCE_INFER_MODEL_MULTI_H

