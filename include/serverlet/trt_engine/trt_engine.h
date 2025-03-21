//
// Created by vipuser on 24-12-25.
//

#ifndef TRT_ENGINE_H
#define TRT_ENGINE_H

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <simple_cuda_toolkits/tensor.hpp>


class TrtEngine {

public:
    TrtEngine();
    ~TrtEngine();

    // 禁用拷贝构造和赋值操作
    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    // 从文件加载引擎
    bool loadFromFile(const std::string& engineFile);
    
    // 从ONNX模型创建引擎
    bool loadFromONNX(const std::string& onnxFile);

    // 创建执行上下文
    bool createContext(const std::string& input_name, const nvinfer1::Dims4& input_dims);

    // 执行推理
    bool infer(Tensor<float>& input, Tensor<float>& output) const;


private:
    // TensorRT资源
    nvinfer1::IRuntime* g_ptr_runtime;
    nvinfer1::ICudaEngine* g_ptr_engine;
    nvinfer1::IExecutionContext* g_ptr_context;

    // 用于日志记录的logger
    class NvLogger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override;
    } g_logger;

    // 清理资源的辅助方法
    void cleanup();
};

#endif //TRT_ENGINE_H
