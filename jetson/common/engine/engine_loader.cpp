//
// Created by ubuntu on 9/13/24.
//

#include "engine_loader.h"
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <NvOnnxParser.h>
#include <simple_cuda_toolkits/tensor.hpp>

#include "common/utils/logger.h"

using namespace nvinfer1;
using namespace nvonnxparser;

// Custom logger class for TensorRT
class TensorLogger : public ILogger {
public:
    // Override log function for custom handling of log messages
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {  // Only log messages of severity higher than INFO
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

// private:
//     // Convert severity to string for easier logging
//     static const char* severityString(Severity severity) {
//         switch (severity) {
//             case Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
//             case Severity::kERROR:          return "ERROR";
//             case Severity::kWARNING:        return "WARNING";
//             case Severity::kVERBOSE:        return "VERBOSE";
//             default:                        return "UNKNOWN";
//         }
//     }
} gLogger;  // Global logger instance

// Load TensorRT engine from a serialized engine file
ICudaEngineUniquePtr loadEngineFromFile(const std::string& engineFile) {
    // Open the engine file in binary mode
    std::ifstream file(engineFile, std::ios::binary);
    if (!file) {
        LOG_ERROR("EngineLoader", "Error opening engine file: " + engineFile);
        exit(EXIT_FAILURE);
    }

    // Read the engine file into a buffer
    file.seekg(0, std::ifstream::end);
    size_t length = file.tellg();
    file.seekg(0, std::ifstream::beg);
    std::vector<char> data(length);
    file.read(data.data(), length);
    if (!file) {
        LOG_ERROR("EngineLoader", "Error reading engine file: " + engineFile);
        exit(EXIT_FAILURE);
    }

    // Create a runtime for deserialization
    auto runtime = std::unique_ptr<IRuntime, void(*)(IRuntime*)>(
            createInferRuntime(gLogger),
            [](IRuntime* r) { r->destroy(); }
    );

    // Deserialize the engine from the buffer
    auto engine = std::unique_ptr<ICudaEngine, void(*)(ICudaEngine*)>(
            runtime->deserializeCudaEngine(data.data(), length),
            [](ICudaEngine* e) { e->destroy(); }
    );

    // Ensure engine is valid
    if (!engine) {
        LOG_ERROR("EngineLoader", "Failed to deserialize engine.");
        exit(EXIT_FAILURE);
    }

    return engine;
}

// Load TensorRT engine from an ONNX model file
ICudaEngineUniquePtr loadEngineFromONNX(const std::string& onnxFilePath) {
    // Create a builder to build the engine
    auto builder = std::unique_ptr<IBuilder, void(*)(IBuilder*)>(
            createInferBuilder(gLogger),
            [](IBuilder* b) { b->destroy(); }
    );

    // Create network definition with explicit batch (required for TensorRT 8.x+)
    auto network = std::unique_ptr<INetworkDefinition, void(*)(INetworkDefinition*)>(
            builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)),
            [](INetworkDefinition* n) { n->destroy(); }
    );

    // Create the ONNX parser
    auto parser = std::unique_ptr<IParser, void(*)(IParser*)>(
            createParser(*network, gLogger),
            [](IParser* p) { p->destroy(); }
    );

    // Open the ONNX file and read its contents
    std::ifstream file(onnxFilePath, std::ios::binary | std::ios::ate);
    if (!file) {
        LOG_ERROR("EngineLoader", "Error opening ONNX file: " + onnxFilePath);
        exit(EXIT_FAILURE);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        LOG_ERROR("EngineLoader", "Error reading ONNX file: " + onnxFilePath);
        exit(EXIT_FAILURE);
    }

    // Parse the ONNX model into the network
    if (!parser->parse(buffer.data(), buffer.size())) {
        // throw std::runtime_error("Failed to parse ONNX model.");
        LOG_ERROR("EngineLoader", "Failed to parse ONNX model.");
        exit(EXIT_FAILURE);
    }

    // Create a configuration for building the engine
    auto config = std::unique_ptr<IBuilderConfig, void(*)(IBuilderConfig*)>(
            builder->createBuilderConfig(),
            [](IBuilderConfig* c) { c->destroy(); }
    );

    // Set memory pool limit for workspace (1MB in this case)
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 20);

    // Build the serialized engine from the network
    std::cout << "Building serialized TensorRT engine..." << std::endl;
    auto serializedModel = std::unique_ptr<IHostMemory, void(*)(IHostMemory*)>(
            builder->buildSerializedNetwork(*network, *config),
            [](IHostMemory* mem) { mem->destroy(); }
    );

    // Ensure the serialized model is valid
    if (!serializedModel) {
        // throw std::runtime_error("Failed to build serialized network.");
        LOG_ERROR("EngineLoader", "Failed to build serialized network.");
        exit(EXIT_FAILURE);
    }

    // Deserialize the serialized engine to get the ICudaEngine
    auto runtime = std::unique_ptr<IRuntime, void(*)(IRuntime*)>(
            createInferRuntime(gLogger),
            [](IRuntime* r) { r->destroy(); }
    );

    auto engine = std::unique_ptr<ICudaEngine, void(*)(ICudaEngine*)>(
            runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size()),
            [](ICudaEngine* e) { e->destroy(); }
    );

    // Ensure engine is valid
    if (!engine) {
        // throw std::runtime_error("Failed to deserialize engine.");
        LOG_ERROR("EngineLoader", "Failed to deserialize engine.");
        exit(EXIT_FAILURE);
    }

    return engine;
}

// Get all input/output tensor names from the TensorRT engine
std::vector<std::string> getTensorNamesFromModel(ICudaEngineUniquePtr& engine) {
    std::vector<std::string> tensor_names;
    std::shared_ptr<ICudaEngine> engine_ptr(engine.get(), [](ICudaEngine*){});

    // Retrieve names of all tensors in the engine
    for (int i = 0, e = engine_ptr->getNbIOTensors(); i < e; ++i) {
        tensor_names.emplace_back(engine_ptr->getIOTensorName(i));
    }

    return tensor_names;
}

// Create an execution context for inference
IExecutionContextUniquePtr createExecutionContext(ICudaEngineUniquePtr &engine, 
        const std::string& input_name, 
        const Dims4& input_dims) {
            
    if (!engine) {
        // throw std::runtime_error("Invalid engine pointer.");
        LOG_ERROR("EngineLoader", "Invalid engine pointer.");
        exit(EXIT_FAILURE);
    }

    // Create the execution context from the engine
    auto context = std::unique_ptr<IExecutionContext, void(*)(IExecutionContext*)>(
            engine->createExecutionContext(),
            [](IExecutionContext* c) { c->destroy(); }
    );

    // Ensure context is valid
    if (!context) {
        throw std::runtime_error("Failed to create execution context.");
    }

    // Set input tensor shape based on provided dimensions
    context->setInputShape(input_name.c_str(), input_dims);

    // Print out the information 
    // std::cout << "[EngineLoader] VERBOSE: Execution context created successfully." << std::endl;
    // std::cout << "[EngineLoader] VERBOSE: Input tensor shape set to: " 
        // << input_dims.d[0] << "x" << input_dims.d[1] << "x" << input_dims.d[2] << "x" << input_dims.d[3] << std::endl; 
    LOG_VERBOSE("EngineLoader", "Execution context created successfully.");
    LOG_VERBOSE("EngineLoader", std::to_string(input_dims.d[0]) + "x" + std::to_string(input_dims.d[1]) + "x" +
        std::to_string(input_dims.d[2]) + "x" + std::to_string(input_dims.d[3]));

    return context;
}

// Perform inference on input and output tensors using the execution context
void inference(IExecutionContextUniquePtr& context, Tensor<float>& input, Tensor<float>& output) {
    if (!input.ptr() || !output.ptr()) {
        // throw std::runtime_error("Invalid input or output tensor.");
        LOG_ERROR("EngineLoader", "Invalid input or output tensor.");
        exit(EXIT_FAILURE);
    }

    // Bind the input and output tensors for inference
    void* buffers[2] = {input.ptr(), output.ptr()};
    if (!context->executeV2(buffers)) {
        // throw std::runtime_error("Failed to execute inference.");
        LOG_ERROR("EngineLoader", "Failed to execute inference.");
        exit(EXIT_FAILURE);
    }
}


std::map<std::string, Tensor<float>> allocateCudaTensors(const std::map<std::string, std::vector<int>>& tensor_info) {
    std::map<std::string, Tensor<float>> allocated_tensors;

    // Iterate over each key-value pair in the map
    for (const auto& [tensor_name, dims] : tensor_info) {

        // Allocate a zero-initialized tensor with the provided dimensions
        Tensor<float> tensor = createZerosTensor<TensorType::FLOAT32>(dims);

        // Print out the information
        // std::cout << "[EngineLoader] VERBOSE: Allocated tensor " << tensor_name << " with dimensions: " 
            // << dims[0] << "x" << dims[1] << "x" << dims[2] << "x" << dims[3] << std::endl;
        LOG_VERBOSE("EngineLoader", "Allocated tensor " + tensor_name + " with dimensions: " +
            std::to_string(dims[0]) + "x" + std::to_string(dims[1]) + "x" + 
            std::to_string(dims[2]) + "x" + std::to_string(dims[3]));

        // Add the tensor to the map using its name as the key
        allocated_tensors.emplace(tensor_name, std::move(tensor));
    }

    return allocated_tensors;  // Return the map of allocated tensors
}