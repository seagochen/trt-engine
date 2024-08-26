#include "common/engine_loader.h"
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>

// Load ONNX headers if user uses ONNX models
#include <NvOnnxParser.h>

#if NV_TENSORRT_MAJOR >= 10
#define TENSORRT_VERSION_10
#elif NV_TENSORRT_MAJOR == 8
#define TENSORRT_VERSION_8
#else
#error "Unsupported TensorRT version"
#endif


using namespace nvinfer1;
using namespace nvonnxparser;  // Add this for ONNX parser


// Logger class for inference common
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;


std::unique_ptr<ICudaEngine, void(*)(ICudaEngine*)> loadEngineFromFile(const std::string& engineFile) {
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Error opening common file: " << engineFile << std::endl;
        exit(-1);
    }

    file.seekg(0, std::ifstream::end);
    size_t length = file.tellg();
    file.seekg(0, std::ifstream::beg);

    if (length > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        std::cerr << "File size is too large to be read into memory." << std::endl;
        exit(-1);
    }

    std::vector<char> data(static_cast<std::streamsize>(length));
    file.read(data.data(), static_cast<std::streamsize>(length));
    if (!file) {
        std::cerr << "Error reading common file: " << engineFile << std::endl;
        exit(-1);
    }

    std::unique_ptr<IRuntime, void(*)(IRuntime*)> runtime(createInferRuntime(gLogger), [](IRuntime* r) { delete r; });
    if (!runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        exit(-1);
    }

    std::unique_ptr<ICudaEngine, void(*)(ICudaEngine*)> engine(runtime->deserializeCudaEngine(data.data(), length), [](ICudaEngine* e) { delete e; });
    if (!engine) {
        std::cerr << "Failed to deserialize common" << std::endl;
        exit(-1);
    }

    return engine;
}

std::unique_ptr<ICudaEngine, void(*)(ICudaEngine*)> loadEngineFromONNX(const std::string& onnxFilePath) {
    // Initialize TensorRT components
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1U << (int)NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto parser = nvonnxparser::createParser(*network, gLogger);

    std::cout << "Parsing ONNX model..." << std::endl;

    // Parse ONNX model
    std::ifstream file(onnxFilePath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening ONNX file: " + onnxFilePath);
    }

    std::cout << "Reading ONNX file..." << std::endl;

    std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (!parser->parse(data.data(), data.size())) {
        throw std::runtime_error("Failed to parse ONNX model.");
    }

    std::cout << "Building TensorRT common..." << std::endl;

    // Build the common
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> engine(
        builder->buildEngineWithConfig(*network, *config),
        [](nvinfer1::ICudaEngine* e) { e->destroy(); }
    );

    std::cout << "Engine built successfully." << std::endl;

    if (!engine) {
        throw std::runtime_error("Failed to build TensorRT common.");
    }

    return engine;
}


std::vector<std::string> getTensorNamesFromModel(ICudaEngine* engine) {
    std::vector<std::string> tensor_names;

#ifdef TENSORRT_VERSION_10
    for (int i = 0, e = common->getNbIOTensors(); i < e; i++) {
        auto const name = common->getIOTensorName(i);
        tensor_names.emplace_back(name);
    }
#elif defined(TENSORRT_VERSION_8)
    int nbBindings = engine->getNbBindings();
    for (int i = 0; i < nbBindings; ++i) {
        const char* name = engine->getBindingName(i);
        tensor_names.emplace_back(name);
    }
#endif

    return tensor_names;
}


TensorDimensions getTensorDimsByName(ICudaEngine* engine, const std::string& tensor_name, tensor_type type) {
    TensorDimensions tensor_dims;

#ifdef TENSORRT_VERSION_10
    auto const dims = common->getTensorShape(tensor_name.c_str());
    int nbDims = dims.nbDims;

    std::vector<int> dim_sizes;
    for (int i = 0; i < nbDims; ++i) {
        dim_sizes.push_back(dims.d[i]);
    }

    tensor_dims = TensorDimensions(dim_sizes, type);  // Assume FLOAT32, adjust as necessary

#elif defined(TENSORRT_VERSION_8)
    int bindingIndex = engine->getBindingIndex(tensor_name.c_str());
    if (bindingIndex == -1) {
        std::cerr << "Tensor name not found: " << tensor_name << std::endl;
        exit(-1);
    }

    auto dims = engine->getBindingDimensions(bindingIndex);
    int nbDims = dims.nbDims;

    std::vector<int> dim_sizes;
    dim_sizes.reserve(nbDims);

    for (int i = 0; i < nbDims; ++i) {
        dim_sizes.push_back(dims.d[i]);
    }

    tensor_dims = TensorDimensions(dim_sizes, type);  // Assume FLOAT32, adjust as necessary

#endif

    return tensor_dims;
}


std::unique_ptr<IExecutionContext, void(*)(IExecutionContext*)> createExecutionContext(ICudaEngine* engine) {
    if (!engine) {
        std::cerr << "Invalid common pointer." << std::endl;
        exit(-1);
    }

    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context." << std::endl;
        exit(-1);
    }

    return {context, [](IExecutionContext* c) { delete c; }};
}


void inference(std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)>& context,
               void* ptr_input, void* ptr_output) {

    // Set the input tensor
    void* buffers[2] = {ptr_input, ptr_output};

    // Execute the inference
    context->executeV2(buffers);
};