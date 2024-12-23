#ifndef _TRT_ENGINE_V8_H
#define _TRT_ENGINE_V8_H

// TrtEngineV8.h
#ifndef TRT_ENGINE_V8_H
#define TRT_ENGINE_V8_H

#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <string>
#include <vector>
#include <map>
#include <simple_cuda_toolkits/tensor.hpp>

class TrtEngineV8 {
public:
    TrtEngineV8();
    ~TrtEngineV8();

    // 禁用拷贝构造和赋值操作
    TrtEngineV8(const TrtEngineV8&) = delete;
    TrtEngineV8& operator=(const TrtEngineV8&) = delete;

    // 从文件加载引擎
    bool loadFromFile(const std::string& engineFile);
    
    // 从ONNX模型创建引擎
    bool loadFromONNX(const std::string& onnxFile);
    
    // 创建执行上下文
    bool createContext(const std::string& input_name, const nvinfer1::Dims4& input_dims);
    
    // 获取模型中所有张量的名称
    std::vector<std::string> getTensorNames() const;
    
    // 执行推理
    bool infer(Tensor<float>& input, Tensor<float>& output);
    
    // 分配CUDA张量
    std::map<std::string, Tensor<float>> allocateTensors(
        const std::map<std::string, std::vector<int>>& tensor_info);

private:
    // TensorRT资源
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    
    // 用于日志记录的logger
    class NvLogger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override;
    } logger_;

    // 清理资源的辅助方法
    void cleanup();
};

#endif // _TRT_ENGINE_V8_H