//
// Created by vipuser on 24-12-25.
//

#include "common/models/engine/trt_engine.h"
#include "common/utils/logger.h"

#include <fstream>


void TrtEngine::NvLogger::log(Severity severity, const char* msg) noexcept {
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

TrtEngine::TrtEngine() {
    g_ptr_runtime = nullptr;
    g_ptr_engine = nullptr;
    g_ptr_context = nullptr;
}

TrtEngine::~TrtEngine() {
    // Release the engine
    cleanup();
}

void TrtEngine::cleanup() {
    if (g_ptr_context) {
        delete g_ptr_context;
        g_ptr_context = nullptr;
    }
    if (g_ptr_engine) {
        delete g_ptr_engine;
        g_ptr_engine = nullptr;
    }
    if (g_ptr_runtime) {
        delete g_ptr_runtime;
        g_ptr_runtime = nullptr;
    }
}

bool TrtEngine::loadFromFile(const std::string& engineFile) {
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


bool TrtEngine::loadFromONNX(const std::string& onnxFile) {
    // Release the previous engine
    cleanup();

    // Create the builder
    nvinfer1::IBuilder* ptr_builder = nvinfer1::createInferBuilder(g_logger);
    if (!ptr_builder) {
        LOG_ERROR("TrtEngine::loadFromONNX", "Failed to create builder");
        return false;
    }

// #if NV_TENSORRT_MAJOR <= 8  // TensorRT 8.X and below

//     // Create the network
//     uint32_t uint32_flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
//     nvinfer1::INetworkDefinition* network = ptr_builder->createNetworkV2(uint32_flag);
//     if (!network) {
//         LOG_ERROR("TrtEngineV8", "Failed to create network");
//         delete ptr_builder;
//         return false;
//     }

// #else // Above TensorRT 8.X

//     // Create the network - Do not explicitly set the batch size
//     nvinfer1::INetworkDefinition* ptr_network = ptr_builder->createNetworkV2(0);  // Use 0 for the flag
//     if (!ptr_network) {
//         LOG_ERROR("TrtEngineV8", "Failed to create network");
//         delete ptr_builder;
//         return false;
//     }

// #endif

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


bool TrtEngine::createContext(const std::string& input_name, const nvinfer1::Dims4& input_dims) {

    if (!g_ptr_engine) {
        LOG_ERROR("TrtEngine::createContext", "Engine is not loaded");
        return false;
    }

    // Create the context
    if (g_ptr_context) {
        delete g_ptr_context;
        g_ptr_context = nullptr;
    }
    g_ptr_context = g_ptr_engine->createExecutionContext();
    if (!g_ptr_context) {
        LOG_ERROR("TrtEngine::createContext", "Failed to create context");
        return false;
    }

    // Set the input binding dimensions
    g_ptr_context->setInputShape(input_name.c_str(), input_dims);

    return true;
}


bool TrtEngine::infer(Tensor<float>& input, Tensor<float>& output) const {
    if (!g_ptr_engine) {
        LOG_ERROR("TrtEngine::infer", "Engine is not loaded");
        return false;
    }

    if (!input.ptr() || !output.ptr()) {
        LOG_ERROR("TrtEngine::infer", "Invalid input/output tensors");
        return false;
    }

    void* ptr_bindings[2] = {input.ptr(), output.ptr()};
    if (!g_ptr_context->executeV2(ptr_bindings)) {
        LOG_ERROR("TrtEngine::infer", "Failed to execute context");
        return false;
    }

    return true;
}