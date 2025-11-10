//
// Created by user on 4/22/25.
//


#include <NvInferVersion.h>  // 一定要包含这个头，才能拿到 NV_TENSORRT_MAJOR 宏
#include <fstream>

#include "trtengine_v2/core/trt_engine_multi.h"
#include "trtengine_v2/utils/logger.h"

// Temporary debugging - Check macro values
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#pragma message("TRT Version Check: MAJOR=" TOSTRING(NV_TENSORRT_MAJOR) ", MINOR=" TOSTRING(NV_TENSORRT_MINOR))


// --------------------------------------
// NvLogger
// --------------------------------------
void TrtEngineMultiTs::NvLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kINFO) return;
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
            LOG_ERROR_TOPIC("TensorRT","INTERNAL_ERROR",msg); break;
        case Severity::kERROR:
            LOG_ERROR_TOPIC("TensorRT", "SYS", msg); break;
        case Severity::kWARNING:
            LOG_WARNING_TOPIC("TensorRT", "SYS", msg); break;
        default:
            LOG_VERBOSE_TOPIC("TensorRT", "SYS", msg); break;
    }
}

// --------------------------------------
// 构造 / 析构
// --------------------------------------
TrtEngineMultiTs::TrtEngineMultiTs()
{
    // 初始化 CUDA 流
    if (cudaStreamCreate(&cuda_stream) != cudaSuccess) {
        LOG_ERROR_TOPIC("TrtEngineMultiTs", "Constructor", "Failed to create CUDA stream.");
        cuda_stream = nullptr;  // 确保流创建失败时不使用无效指针
    }
}

TrtEngineMultiTs::~TrtEngineMultiTs()
{
    // 先销毁 CUDA 流
    if (cuda_stream) {
        cudaStreamDestroy(cuda_stream);
        cuda_stream = nullptr;  // 确保流销毁后指针不再指向无效内存
    }

    // 清理 TensorRT 相关资源
    cleanup();
}

void TrtEngineMultiTs::cleanup() {
    // 修复：对于这些 TensorRT 接口对象，应使用 delete 运算符释放，而不是 destroy()
    if (ptr_context)  { delete ptr_context;  ptr_context  = nullptr; }
    if (ptr_engine)   { delete ptr_engine;   ptr_engine   = nullptr; }
    if (ptr_runtime)  { delete ptr_runtime;  ptr_runtime  = nullptr; }
}

bool TrtEngineMultiTs::loadFromFile(const std::string& engineFile) {
    // Release the previous engine
    cleanup();

    // Check if the file exists and can be opened
    std::ifstream cls_file(engineFile, std::ios::binary);
    if (!cls_file.is_open()) {
        LOG_ERROR_TOPIC("TrtEngineMultiTs", "loadFromFile",
            "Failed to read the binary file: " + engineFile);
        return false;
    }

    // Seek to the end of the file to get the size
    cls_file.seekg(0, std::ios::end);
    size_t uint64_length = cls_file.tellg();
    cls_file.seekg(0, std::ios::beg);
    std::vector<char> vec_data(uint64_length);
    if (!cls_file.read(vec_data.data(), static_cast<std::streamsize>(uint64_length))) {
        LOG_ERROR_TOPIC("TrtEngineMultiTs", "loadFromFile",
            "Error reading engine file: " + engineFile);
        return false;
    }

    // Deserialize the engine
    ptr_runtime = nvinfer1::createInferRuntime(cuda_logger);
    if (!ptr_runtime) {
        LOG_ERROR_TOPIC("TrtEngineMultiTs", "loadFromFile",
            "Failed to create TensorRT runtime.");
        return false;
    }

    // Deserialize the engine
    ptr_engine = ptr_runtime->deserializeCudaEngine(vec_data.data(), uint64_length);
    if (!ptr_engine) {
        LOG_ERROR_TOPIC("TrtEngineMultiTs", "loadFromFile",
            "Failed to deserialize engine from file: " + engineFile);
        return false;
    }

    // Successfully loaded the engine
    return true;
}


bool TrtEngineMultiTs::loadFromONNX(const std::string& onnxFile) {
    // Release the previous engine
    cleanup();

    // Create the builder
    nvinfer1::IBuilder* ptr_builder = nvinfer1::createInferBuilder(cuda_logger);
    if (!ptr_builder) {
        LOG_ERROR_TOPIC("TrtEngineMultiTs", "loadFromONNX",
            "Failed to create TensorRT builder.");
        return false;
    }

#if NV_TENSORRT_MAJOR <= 8  // TensorRT 8.X and below

    // Create the network
    uint32_t uint32_flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* ptr_network = ptr_builder->createNetworkV2(uint32_flag);
    if (!ptr_network) {
        LOG_ERROR("TrtEngineMultiTs", "Failed to create network.");
        delete ptr_builder;
        return false;
    }

#else // Above TensorRT 8.X

    // Create the network - Do not explicitly set the batch size
    nvinfer1::INetworkDefinition* ptr_network = ptr_builder->createNetworkV2(0);  // Use 0 for the flag
    if (!ptr_network) {
        LOG_ERROR("TrtEngineMultiTs", "Failed to create network.");
        delete ptr_builder;
        return false;
    }

#endif

    // Create the parser
    auto* ptr_parser = nvonnxparser::createParser(*ptr_network, cuda_logger);
    if (!ptr_parser) {
        LOG_ERROR_TOPIC("TrtEngineMultiTs", "loadFromONNX", "Failed to create parser.");
        delete ptr_network;
        delete ptr_builder;
        return false;
    }

    // Parse the ONNX file
    std::ifstream cls_file(onnxFile, std::ios::binary | std::ios::ate);
    if (!cls_file) {
        LOG_ERROR_TOPIC("TrtEngineMultiTs", "loadFromONNX",
            "Failed to open ONNX file: " + onnxFile);
        delete ptr_parser;
        delete ptr_network;
        delete ptr_builder;
        return false;
    }

    // Get the size of the file
    size_t uint64_length = cls_file.tellg();
    cls_file.seekg(0, std::ios::beg);
    std::vector<char> vec_data(uint64_length);
    if (!cls_file.read(vec_data.data(), static_cast<std::streamsize>(uint64_length))) {
        LOG_ERROR_TOPIC("TrtEngineMultiTs", "loadFromONNX",
            "Error reading ONNX file: " + onnxFile);
        delete ptr_parser;
        delete ptr_network;
        delete ptr_builder;
        return false;
    }

    // Parse the ONNX file
    if (!ptr_parser->parse(vec_data.data(), uint64_length)) {
        LOG_ERROR_TOPIC("TrtEngineMultiTs", "loadFromONNX",
            "Failed to parse ONNX file: " + onnxFile);
        delete ptr_parser;
        delete ptr_network;
        delete ptr_builder;
        return false;
    }

    // Create the builder configuration
    nvinfer1::IBuilderConfig* ptr_config = ptr_builder->createBuilderConfig();
    if (!ptr_config) {
        LOG_ERROR_TOPIC("TrtEngineMultiTs", "loadFromONNX", "Failed to create builder config.");
        delete ptr_parser;
        delete ptr_network;
        delete ptr_builder;
        return false;
    }

    // Set the maximum workspace size
    ptr_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    // Build the serialized engine
    auto ptr_serialized = ptr_builder->buildSerializedNetwork(*ptr_network, *ptr_config);
    if (!ptr_serialized) {
        LOG_ERROR_TOPIC("TrtEngineMultiTs", "loadFromONNX", "Failed to build serialized engine.");
        delete ptr_parser;
        delete ptr_network;
        delete ptr_builder;
        delete ptr_config;
        return false;
    }

    // Create the runtime
    ptr_runtime = nvinfer1::createInferRuntime(cuda_logger);
    if (!ptr_runtime) {
        LOG_ERROR_TOPIC("TrtEngineMultiTs", "loadFromONNX", "Failed to create runtime.");
        delete ptr_parser;
        delete ptr_network;
        delete ptr_builder;
        delete ptr_config;
        return false;
    }

    // Deserialize the engine
    ptr_engine = ptr_runtime->deserializeCudaEngine(ptr_serialized->data(), ptr_serialized->size());
    if (!ptr_engine) {
        LOG_ERROR_TOPIC("TrtEngineMultiTs", "loadFromONNX", "Failed to deserialize engine.");
        delete ptr_parser;
        delete ptr_network;
        delete ptr_builder;
        delete ptr_config;
        return false;
    }

    // Clear the temporary objects
    delete ptr_serialized;
    delete ptr_parser;
    delete ptr_network;
    delete ptr_builder;
    delete ptr_config;

    // Successfully loaded the engine
    return true;
}

//-----------------------------------------------------------------------------
// createContext：替代旧的 getNbBindings/getBindingIndex/setBindingDimensions
// - 用 setInputShape 设置动态输入形状
// - 不再需要显式索引绑定，后续通过名字 setTensorAddress 绑定内存
//-----------------------------------------------------------------------------
bool TrtEngineMultiTs::createContext(
        const std::vector<std::string>& input_names,
        const std::vector<nvinfer1::Dims4>& input_dims,
        const std::vector<std::string>& output_names)
{
    if (!ptr_engine) {
        LOG_ERROR_TOPIC("TrtEngineMultiTs", "createContext", "Engine is not loaded.");
        return false;
    }

    // 销毁旧 context（如有）
    if (ptr_context) {
        delete ptr_context; // 修复：使用 delete
        ptr_context = nullptr;
    }

    // 创建新 context
    ptr_context = ptr_engine->createExecutionContext();
    if (!ptr_context) {
        LOG_ERROR_TOPIC("TrtEngineMultiTs", "createContext", "Failed to create execution context.");
        return false;
    }

    // 设置每个输入的动态形状
    if (input_names.size() != input_dims.size()) {
        LOG_ERROR_TOPIC("TrtEngineMultiTs", "createContext",
                        "Input names and dimensions size mismatch.");
        return false;
    }
    for (size_t i = 0; i < input_names.size(); ++i) {
        if (!ptr_context->setInputShape(
                input_names[i].c_str(), input_dims[i])) {
            LOG_ERROR_TOPIC("TrtEngineMultiTs", "createContext",
                            "Failed to set input shape for: " + input_names[i]);
            return false;
        }
    }

    // 记录名字，供 infer 时绑定内存
    vec_inputNames  = input_names;
    vec_outputNames = output_names;
    return true;
}

//-----------------------------------------------------------------------------
// infer：替代旧的 executeV2/enqueueV2 + getBindingIndex + bindings 数组
// - 用 setInputTensorAddress/setOutputTensorAddress 绑定内存
// - 使用 enqueueV3 执行推理（V2/V1 都已弃用）
//-----------------------------------------------------------------------------
bool TrtEngineMultiTs::infer(
        const std::vector<Tensor<float>*>& inputs,
        const std::vector<Tensor<float>*>& outputs) const
{
    if (!ptr_context) {
        LOG_ERROR_TOPIC("TrtEngineMultiTs", "infer", "ExecutionContext does not exist.");
        return false;
    }
    if (inputs.size()  != vec_inputNames.size()  ||
        outputs.size() != vec_outputNames.size()) {
        LOG_ERROR_TOPIC("TrtEngineMultiTs", "infer", "Input/Output size mismatch with context.");
        return false;
    }

    // 只有在 TensorRT 8.6+ 版本才支持新的 API
#if (NV_TENSORRT_MAJOR > 8) || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR > 6)
    /**
     * 8.6+ 版本之后，使用 setInputTensorAddress 和 setOutputTensorAddress 以及 enqueueV3 来执行推理
     */

    // 将输入张量的地址绑定到 context
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!ptr_context->setInputTensorAddress(
                vec_inputNames[i].c_str(),
                inputs[i]->ptr()))
        {
            LOG_ERROR_TOPIC("TrtEngineMultiTs", "infer",
                "setInputTensorAddress failed: " + vec_inputNames[i]);
            return false;
        }
    }

    // 将输出张量的地址绑定到 context
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto outPtr = outputs[i]->ptr();
        if (!ptr_context->setOutputTensorAddress(
                vec_outputNames[i].c_str(),
                outPtr))
        {
            LOG_ERROR_TOPIC("TrtEngineMultiTs", "infer",
                "setOutputTensorAddress failed: " + vec_outputNames[i]);
            return false;
        }
    }

    // 执行推理
    if (!ptr_context->enqueueV3(cuda_stream)) {
        LOG_DEBUG_TOPIC("TrtEngineMultiTs", "infer",
            "Failed to enqueueV3 with bindings: " +
            std::to_string(inputs.size() + outputs.size()));
        return false;
    }

    // 推理完成后同步 CUDA 流，确保结果可用且资源可以被安全访问
    cudaStreamSynchronize(cuda_stream);

#else
    // ---- TensorRT 8.5 及以下老 API 路径 ----
    int nb = ptr_engine->getNbBindings();
    std::vector<void*> bindings(nb, nullptr);

    // inputs
    for (size_t i = 0; i < inputs.size(); ++i) {
        int idx = ptr_engine->getBindingIndex(vec_inputNames[i].c_str());
        if (idx < 0 || idx >= nb) {
            LOG_ERROR_TOPIC("TrtEngineMultiTs", "infer", "Invalid input binding: " + vec_inputNames[i]);
            return false;
        }
        bindings[idx] = const_cast<float*>(inputs[i]->ptr());
    }
    // outputs
    for (size_t i = 0; i < outputs.size(); ++i) {
        int idx = ptr_engine->getBindingIndex(vec_outputNames[i].c_str());
        if (idx < 0 || idx >= nb) {
            LOG_ERROR_TOPIC("TrtEngineMultiTs", "infer", "Invalid output binding: " + vec_outputNames[i]);
            return false;
        }
        bindings[idx] = const_cast<float*>(outputs[i]->ptr());
    }
    // 旧版 executeV2，不带 stream
    if (!ptr_context->executeV2(bindings.data())) {
        LOG_DEBUG_TOPIC("TrtEngineMultiTs", "infer", "Bindings: " + std::to_string(bindings.size()));
        return false;
    }
#endif

    return true;
}
