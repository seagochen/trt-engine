//
// Created by user on 4/22/25.
//

#ifndef COMBINEDPROJECT_TRT_ENGINE_MULTI_TS_H
#define COMBINEDPROJECT_TRT_ENGINE_MULTI_TS_H

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <simple_cuda_toolkits/tensor.hpp>
#include <string>
#include <vector>

class TrtEngineMultiTs {
public:
    TrtEngineMultiTs();
    ~TrtEngineMultiTs();

    TrtEngineMultiTs(const TrtEngineMultiTs&) = delete;
    TrtEngineMultiTs& operator=(const TrtEngineMultiTs&) = delete;

    // 从文件或 ONNX 创建 engine
    bool loadFromFile(const std::string& engineFile);
    bool loadFromONNX(const std::string& onnxFile);

    /**
     * 创建 ExecutionContext 并设置所有输入的动态形状
     *
     * @param input_names  输入 tensor 名称列表
     * @param input_dims   与 input_names 对应的 nvinfer1::Dims（动态形状）
     * @param output_names 输出 tensor 名称列表
     */
    bool createContext(
            const std::vector<std::string>& input_names,
            const std::vector<nvinfer1::Dims4>&   input_dims,
            const std::vector<std::string>&       output_names);

    /**
     * 多输入多输出推理
     *
     * @param inputs   与 createContext 时 input_names 顺序一致的 device Tensor 列表
     * @param outputs  与 createContext 时 output_names 顺序一致的 device Tensor 列表
     * @param stream   用于 enqueueV3 的 CUDA 流
     */
    bool infer(
            const std::vector<Tensor<float>>& inputs,
            const std::vector<Tensor<float>>& outputs,
            cudaStream_t                      stream) const;

private:
    // TensorRT 核心对象
    nvinfer1::IRuntime*         g_ptr_runtime  = nullptr;
    nvinfer1::ICudaEngine*      g_ptr_engine   = nullptr;
    nvinfer1::IExecutionContext* g_ptr_context = nullptr;

    // 记录名字用于后续绑定
    std::vector<std::string>     m_inputNames;
    std::vector<std::string>     m_outputNames;

    // Logger
    class NvLogger : public nvinfer1::ILogger {
        void log(Severity sev, const char* msg) noexcept override;
    } g_logger;

    void cleanup();
};

#endif //COMBINEDPROJECT_TRT_ENGINE_MULTI_TS_H
