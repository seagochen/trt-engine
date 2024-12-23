#include <fstream>
#include <iostream>
#include "common/utils/logger.h"
#include "common/engine/trt_engine_v8.h"


TrtEngineV8::TrtEngineV8() 
    : runtime_(nullptr)
    , engine_(nullptr)
    , context_(nullptr) {
}

TrtEngineV8::~TrtEngineV8() {
    cleanup();
}

void TrtEngineV8::NvLogger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
    if (severity != Severity::kINFO) {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                LOG_ERROR_TOPIC("TensorRT", "INTERNAL_ERROR", msg);
                break;
            case Severity::kERROR:
                LOG_ERROR("TensorRT", msg);
                break;
            case Severity::kWARNING:
                LOG_WARNING("TensorRT", msg);
                break;
            default:
                LOG_VERBOSE("TensorRT", msg);
                break;
        }
    }
}

void TrtEngineV8::cleanup() {
    if (context_) {
        context_->destroy();
        context_ = nullptr;
    }
    if (engine_) {
        engine_->destroy();
        engine_ = nullptr;
    }
    if (runtime_) {
        runtime_->destroy();
        runtime_ = nullptr;
    }
}

bool TrtEngineV8::loadFromFile(const std::string& engineFile) {
    cleanup(); // 清理现有资源

    std::ifstream file(engineFile, std::ios::binary);
    if (!file) {
        LOG_ERROR("TrtEngineV8", "Error opening engine file: " + engineFile);
        return false;
    }

    // 读取引擎文件内容
    file.seekg(0, std::ios::end);
    size_t length = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(length);
    if (!file.read(data.data(), length)) {
        LOG_ERROR("TrtEngineV8", "Error reading engine file: " + engineFile);
        return false;
    }

    // 创建runtime和engine
    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) {
        LOG_ERROR("TrtEngineV8", "Failed to create runtime");
        return false;
    }

    engine_ = runtime_->deserializeCudaEngine(data.data(), length);
    if (!engine_) {
        LOG_ERROR("TrtEngineV8", "Failed to deserialize engine");
        cleanup();
        return false;
    }

    return true;
}

bool TrtEngineV8::loadFromONNX(const std::string& onnxFile) {
    cleanup(); // 清理现有资源

    // 创建builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger_);
    if (!builder) {
        LOG_ERROR("TrtEngineV8", "Failed to create builder");
        return false;
    }

    // 创建网络
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);
    if (!network) {
        LOG_ERROR("TrtEngineV8", "Failed to create network");
        builder->destroy();
        return false;
    }

    // 创建ONNX解析器
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger_);
    if (!parser) {
        LOG_ERROR("TrtEngineV8", "Failed to create parser");
        network->destroy();
        builder->destroy();
        return false;
    }

    // 读取并解析ONNX文件
    std::ifstream file(onnxFile, std::ios::binary | std::ios::ate);
    if (!file) {
        LOG_ERROR("TrtEngineV8", "Error opening ONNX file: " + onnxFile);
        parser->destroy();
        network->destroy();
        builder->destroy();
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        LOG_ERROR("TrtEngineV8", "Error reading ONNX file: " + onnxFile);
        parser->destroy();
        network->destroy();
        builder->destroy();
        return false;
    }

    // 解析ONNX模型
    if (!parser->parse(buffer.data(), buffer.size())) {
        LOG_ERROR("TrtEngineV8", "Failed to parse ONNX model");
        parser->destroy();
        network->destroy();
        builder->destroy();
        return false;
    }

    // 创建构建配置
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    if (!config) {
        LOG_ERROR("TrtEngineV8", "Failed to create builder config");
        parser->destroy();
        network->destroy();
        builder->destroy();
        return false;
    }

    // 设置工作空间大小
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    // 构建序列化网络
    LOG_VERBOSE("TrtEngineV8", "Building serialized TensorRT engine...");
    nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
    if (!serializedModel) {
        LOG_ERROR("TrtEngineV8", "Failed to build serialized network");
        config->destroy();
        parser->destroy();
        network->destroy();
        builder->destroy();
        return false;
    }

    // 创建runtime和反序列化engine
    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) {
        LOG_ERROR("TrtEngineV8", "Failed to create runtime");
        serializedModel->destroy();
        config->destroy();
        parser->destroy();
        network->destroy();
        builder->destroy();
        return false;
    }

    engine_ = runtime_->deserializeCudaEngine(serializedModel->data(), serializedModel->size());

    // 清理临时资源
    serializedModel->destroy();
    config->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();

    if (!engine_) {
        LOG_ERROR("TrtEngineV8", "Failed to deserialize engine");
        cleanup();
        return false;
    }

    return true;
}

bool TrtEngineV8::createContext(const std::string& input_name, const nvinfer1::Dims4& input_dims) {
    if (!engine_) {
        LOG_ERROR("TrtEngineV8", "Engine not initialized");
        return false;
    }

    if (context_) {
        context_->destroy();
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        LOG_ERROR("TrtEngineV8", "Failed to create execution context");
        return false;
    }

    context_->setInputShape(input_name.c_str(), input_dims);
    
    LOG_VERBOSE("TrtEngineV8", "Execution context created successfully");
    LOG_VERBOSE("TrtEngineV8", std::to_string(input_dims.d[0]) + "x" + 
                               std::to_string(input_dims.d[1]) + "x" +
                               std::to_string(input_dims.d[2]) + "x" + 
                               std::to_string(input_dims.d[3]));
    return true;
}

std::vector<std::string> TrtEngineV8::getTensorNames() const {
    std::vector<std::string> tensor_names;
    if (!engine_) {
        LOG_ERROR("TrtEngineV8", "Engine not initialized");
        return tensor_names;
    }

    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        tensor_names.push_back(engine_->getIOTensorName(i));
    }

    return tensor_names;
}

bool TrtEngineV8::infer(Tensor<float>& input, Tensor<float>& output) {
    if (!context_) {
        LOG_ERROR("TrtEngineV8", "Context not initialized");
        return false;
    }

    if (!input.ptr() || !output.ptr()) {
        LOG_ERROR("TrtEngineV8", "Invalid input or output tensor");
        return false;
    }

    void* buffers[2] = {input.ptr(), output.ptr()};
    if (!context_->executeV2(buffers)) {
        LOG_ERROR("TrtEngineV8", "Failed to execute inference");
        return false;
    }

    return true;
}

std::map<std::string, Tensor<float>> TrtEngineV8::allocateTensors(
    const std::map<std::string, std::vector<int>>& tensor_info) {
    
    std::map<std::string, Tensor<float>> allocated_tensors;
    
    for (const auto& [tensor_name, dims] : tensor_info) {
        Tensor<float> tensor = createZerosTensor<TensorType::FLOAT32>(dims);
        
        std::string dims_str;
        for (size_t i = 0; i < dims.size(); ++i) {
            dims_str += std::to_string(dims[i]);
            if (i < dims.size() - 1) {
                dims_str += "x";
            }
        }
        
        LOG_VERBOSE("TrtEngineV8", "Allocated tensor " + tensor_name + 
                   " with dimensions: " + dims_str);
        
        allocated_tensors.emplace(tensor_name, std::move(tensor));
    }
    
    return allocated_tensors;
}