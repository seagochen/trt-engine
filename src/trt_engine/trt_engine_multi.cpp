//
// Created by user on 4/22/25.
//

#include "serverlet/trt_engine/trt_engine_multi.h"
#include "serverlet/utils/logger.h"

#include <fstream>

// --------------------------------------
// NvLogger
// --------------------------------------
void TrtEngineMultiTs::NvLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kINFO) return;
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
            LOG_ERROR_TOPIC("TensorRT","INTERNAL_ERROR",msg); break;
        case Severity::kERROR:
            LOG_ERROR("TensorRT", msg); break;
        case Severity::kWARNING:
            LOG_WARNING("TensorRT", msg); break;
        default:
            LOG_VERBOSE("TensorRT", msg); break;
    }
}

// --------------------------------------
// 构造 / 析构
// --------------------------------------
TrtEngineMultiTs::TrtEngineMultiTs() {}
TrtEngineMultiTs::~TrtEngineMultiTs() { cleanup(); }

void TrtEngineMultiTs::cleanup() {
    if (g_ptr_context)  { delete g_ptr_context;  g_ptr_context  = nullptr; }
    if (g_ptr_engine)   { delete g_ptr_engine;   g_ptr_engine   = nullptr; }
    if (g_ptr_runtime)  { delete g_ptr_runtime;  g_ptr_runtime  = nullptr; }
//    m_inputBindings.clear();
//    m_outputBindings.clear();
}

bool TrtEngineMultiTs::loadFromFile(const std::string& engineFile) {
    // Release the previous engine
    cleanup();

    // Check if the file exists and can be opened
    std::ifstream cls_file(engineFile, std::ios::binary);
    if (!cls_file.is_open()) {
        LOG_ERROR("TrtEngine::loadFromFile", "Failed to read the binary file");
        return false;
    }

    // Seek to the end of the file to get the size
    cls_file.seekg(0, std::ios::end);
    size_t uint64_length = cls_file.tellg();
    cls_file.seekg(0, std::ios::beg);
    std::vector<char> vec_data(uint64_length);
    if (!cls_file.read(vec_data.data(), static_cast<std::streamsize>(uint64_length))) {
        LOG_ERROR("TrtEngine::loadFromFile", "Error reading engine file: " + engineFile);
        return false;
    }

    // Deserialize the engine
    g_ptr_runtime = nvinfer1::createInferRuntime(g_logger);
    if (!g_ptr_runtime) {
        LOG_ERROR("TrtEngine::loadFromFile", "Failed to create runtime");
        return false;
    }

    // Deserialize the engine
    g_ptr_engine = g_ptr_runtime->deserializeCudaEngine(vec_data.data(), uint64_length);
    if (!g_ptr_engine) {
        LOG_ERROR("TrtEngine::loadFromFile", "Failed to deserialize engine");
        return false;
    }

    // Successfully loaded the engine
    return true;
}


bool TrtEngineMultiTs::loadFromONNX(const std::string& onnxFile) {
    // Release the previous engine
    cleanup();

    // Create the builder
    nvinfer1::IBuilder* ptr_builder = nvinfer1::createInferBuilder(g_logger);
    if (!ptr_builder) {
        LOG_ERROR("TrtEngine::loadFromONNX", "Failed to create builder");
        return false;
    }

#if NV_TENSORRT_MAJOR <= 8  // TensorRT 8.X and below

    // Create the network
    uint32_t uint32_flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* ptr_network = ptr_builder->createNetworkV2(uint32_flag);
    if (!ptr_network) {
        LOG_ERROR("TrtEngineV8", "Failed to create network");
        delete ptr_builder;
        return false;
    }

#else // Above TensorRT 8.X

    // Create the network - Do not explicitly set the batch size
    nvinfer1::INetworkDefinition* ptr_network = ptr_builder->createNetworkV2(0);  // Use 0 for the flag
    if (!ptr_network) {
        LOG_ERROR("TrtEngineV8", "Failed to create network");
        delete ptr_builder;
        return false;
    }

#endif

    // Create the parser
    auto* ptr_parser = nvonnxparser::createParser(*ptr_network, g_logger);
    if (!ptr_parser) {
        LOG_ERROR("TrtEngine::loadFromONNX", "Failed to create parser");
        delete ptr_network;
        delete ptr_builder;
        return false;
    }

    // Parse the ONNX file
    std::ifstream cls_file(onnxFile, std::ios::binary | std::ios::ate);
    if (!cls_file) {
        LOG_ERROR("TrtEngine::loadFromONNX", "Failed to open ONNX file: " + onnxFile);
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
        LOG_ERROR("TrtEngine::loadFromONNX", "Error reading ONNX file: " + onnxFile);
        delete ptr_parser;
        delete ptr_network;
        delete ptr_builder;
        return false;
    }

    // Parse the ONNX file
    if (!ptr_parser->parse(vec_data.data(), uint64_length)) {
        LOG_ERROR("TrtEngine::loadFromONNX", "Failed to parse ONNX file: " + onnxFile);
        delete ptr_parser;
        delete ptr_network;
        delete ptr_builder;
        return false;
    }

    // Create the builder configuration
    nvinfer1::IBuilderConfig* ptr_config = ptr_builder->createBuilderConfig();
    if (!ptr_config) {
        LOG_ERROR("TrtEngine::loadFromONNX", "Failed to create builder config");
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
        LOG_ERROR("TrtEngine::loadFromONNX", "Failed to build serialized engine");
        delete ptr_parser;
        delete ptr_network;
        delete ptr_builder;
        delete ptr_config;
        return false;
    }

    // Create the runtime
    g_ptr_runtime = nvinfer1::createInferRuntime(g_logger);
    if (!g_ptr_runtime) {
        LOG_ERROR("TrtEngine::loadFromONNX", "Failed to create runtime");
        delete ptr_parser;
        delete ptr_network;
        delete ptr_builder;
        delete ptr_config;
        return false;
    }

    // Deserialize the engine
    g_ptr_engine = g_ptr_runtime->deserializeCudaEngine(ptr_serialized->data(), ptr_serialized->size());
    if (!g_ptr_engine) {
        LOG_ERROR("TrtEngine::loadFromONNX", "Failed to deserialize engine");
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
// - 用 setInputShape 设置动态输入形状  :contentReference[oaicite:0]{index=0}
// - 不再需要显式索引绑定，后续通过名字 setTensorAddress 绑定内存
//-----------------------------------------------------------------------------
bool TrtEngineMultiTs::createContext(
        const std::vector<std::string>& input_names,
        const std::vector<nvinfer1::Dims4>& input_dims,
        const std::vector<std::string>& output_names)
{
    if (!g_ptr_engine) {
        LOG_ERROR("TrtEngine::createContext", "Engine 未加载");
        return false;
    }

    // 销毁旧 context（如有）
    if (g_ptr_context) {
        delete g_ptr_context;
        g_ptr_context = nullptr;
    }

    // 创建新 context
    g_ptr_context = g_ptr_engine->createExecutionContext();
    if (!g_ptr_context) {
        LOG_ERROR("TrtEngine::createContext", "createExecutionContext 失败");
        return false;
    }

    // 设置每个输入的动态形状
    if (input_names.size() != input_dims.size()) {
        LOG_ERROR("TrtEngine::createContext", "输入名称和维度数量不匹配");
        return false;
    }
    for (size_t i = 0; i < input_names.size(); ++i) {
        if (!g_ptr_context->setInputShape(
                input_names[i].c_str(), input_dims[i])) {
            LOG_ERROR("TrtEngine::createContext",
                      "setInputShape 失败: " + input_names[i]);
            return false;
        }
    }

    // 记录名字，供 infer 时绑定内存
    m_inputNames  = input_names;
    m_outputNames = output_names;
    return true;
}

//-----------------------------------------------------------------------------
// infer：替代旧的 executeV2/enqueueV2 + getBindingIndex + bindings 数组
// - 用 setInputTensorAddress/setOutputTensorAddress 绑定内存  :contentReference[oaicite:1]{index=1}
// - 使用 enqueueV3 执行推理（V2/V1 都已弃用）    :contentReference[oaicite:2]{index=2}
//-----------------------------------------------------------------------------
bool TrtEngineMultiTs::infer(
        const std::vector<Tensor<float>>& inputs,
        const std::vector<Tensor<float>>& outputs,
        cudaStream_t                      stream) const
{
    if (!g_ptr_context) {
        LOG_ERROR("TrtEngine::infer", "ExecutionContext 未初始化");
        return false;
    }
    if (inputs.size()  != m_inputNames.size()  ||
        outputs.size() != m_outputNames.size()) {
        LOG_ERROR("TrtEngine::infer", "I/O 大小与 Context 不匹配");
        return false;
    }

    // 绑定所有输入 buffer
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& name = m_inputNames[i];
        if (!g_ptr_context->setInputTensorAddress(
                name.c_str(), inputs[i].ptr())) {
            LOG_ERROR("TrtEngine::infer",
                      "setInputTensorAddress 失败: " + name);
            return false;
        }
    }

    // 绑定所有输出 buffer
    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& name = m_outputNames[i];

        // outputs[i].ptr()返回的是一个智能指针，因此需要做一些转换
        auto ptr = static_cast<void*>(const_cast<float*>(outputs[i].ptr()));

        // 绑定输出 buffer
        if (!g_ptr_context->setOutputTensorAddress(
                name.c_str(), ptr)) {
            LOG_ERROR("TrtEngine::infer",
                      "setOutputTensorAddress 失败: " + name);
            return false;
        }
    }

    // 启动推理（显式 batch 网络使用 enqueueV3）
    if (!g_ptr_context->enqueueV3(stream)) {
        LOG_ERROR("TrtEngine::infer", "enqueueV3 失败");
        return false;
    }
    return true;
}