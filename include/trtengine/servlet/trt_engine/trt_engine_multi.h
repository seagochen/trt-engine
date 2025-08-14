//
// Created by user on 4/22/25.
//

#ifndef COMBINEDPROJECT_INFER_TRT_ENGINE_MULTI_TS_H
#define COMBINEDPROJECT_INFER_TRT_ENGINE_MULTI_TS_H

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
     */
    bool infer(
        const std::vector<Tensor<float>*>& inputs,
        const std::vector<Tensor<float>*>& outputs
    ) const;

    /**
     * 获取当前 CUDA 流
     * @return 返回当前的 CUDA 流
     */
    [[nodiscard]]
    cudaStream_t getCudaStream() const { return cuda_stream; }


private:
    // TensorRT 核心对象
    nvinfer1::IRuntime*             ptr_runtime     = nullptr;
    nvinfer1::ICudaEngine*          ptr_engine      = nullptr;
    nvinfer1::IExecutionContext*    ptr_context     = nullptr;
    cudaStream_t                    cuda_stream     = nullptr; // CUDA 流

    // 记录名字用于后续绑定
    std::vector<std::string>     vec_inputNames;
    std::vector<std::string>     vec_outputNames;

    // Logger
    class NvLogger : public nvinfer1::ILogger {
        void log(Severity sev, const char* msg) noexcept override;
    } cuda_logger;

    void cleanup();
};

#endif // COMBINEDPROJECT_INFER_TRT_ENGINE_MULTI_TS_H
