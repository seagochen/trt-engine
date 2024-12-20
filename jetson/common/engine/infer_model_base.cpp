//
// Created by orlando on 12/19/24.
//

#include "infer_model_base.h"
#include "common/utils/logger.h"


// Custom deleter for ICudaEngine
auto engineDeleter = [](nvinfer1::ICudaEngine* engine) {
    if (engine) {
        engine->destroy();
    }
};


// Custom deleter for IExecutionContext
auto contextDeleter = [](nvinfer1::IExecutionContext* context) {
    if (context) {
        context->destroy();
    }
};


// Create a TensorRT execution context
InferModelBase::InferModelBase(const std::string& engine_path,
                                const std::map<std::string, std::string>& names,
                                const std::vector<int>& input_dims,
                                const std::vector<int>& output_dims):
    engine(nullptr, engineDeleter), context(nullptr, contextDeleter) {

    // Load the TensorRT engine
    loadEngine(engine_path, names, input_dims, output_dims);
    LOG_VERBOSE_TOPIC("InferModelBase", "InferModelBase", "TensorRT engine loaded successfully.");

    // Initialize tensorrt buffers
    allocateTensorRTBufs(names, input_dims, output_dims);
    LOG_VERBOSE_TOPIC("InferModelBase", "InferModelBase", "TensorRT buffers are ready.");

    // Allocate input and output buffers for CUDA
    allocateCudaBufs(input_dims, output_dims);
    LOG_VERBOSE_TOPIC("InferModelBase", "InferModelBase", "Temporary buffers for CUDA are ready.");

}


// Destroy the TensorRT engine and buffers
InferModelBase::~InferModelBase() {

    // Clear the buffers
    trt_buffers.clear();
    cuda_input_buffers.clear();
    cuda_output_buffers.clear();
    LOG_VERBOSE_TOPIC("InferModelBase", "~InferModelBase", "Buffers cleared successfully.");

    // Destroy the execution context and engine
    context.reset();
    engine.reset();
    LOG_VERBOSE_TOPIC("InferModelBase", "~InferModelBase", "Execution context and engine destroyed successfully.");

    // Log the destruction of the object
    LOG_VERBOSE_TOPIC("InferModelBase", "~InferModelBase", "InferModelBase object destroyed successfully.");
}


// Convert std::vector<int> to nvinfer1::Dims4
nvinfer1::Dims4 toDims4(const std::vector<int>& dims) {
    if (dims.size() != 4) {
        throw std::runtime_error("Invalid dimensions for Dims4. Expected size 4.");
    }
    return nvinfer1::Dims4{dims[0], dims[1], dims[2], dims[3]};
}


// Load the TensorRT engine and initialize buffers
void InferModelBase::loadEngine(
    const std::string& engine_path,
    const std::map<std::string, std::string>& names,
    const std::vector<int>& input_dims,
    const std::vector<int>& output_dims) {

    // Load the TensorRT engine from the serialized engine file
    engine = loadEngineFromFile(engine_path);
    if (!engine) {
        // throw std::runtime_error("Failed to load engine from file.");
        LOG_ERROR_TOPIC("InferModelBase", "loadEngine", "Failed to load engine from file.");
        exit(EXIT_FAILURE);
    }

    // Convert input_dims to nvinfer1::Dims4
    nvinfer1::Dims4 dims4_input = toDims4(input_dims);

    // Create a context for executing the engine
    context = createExecutionContext(engine, names.at("input"), dims4_input);
}


void InferModelBase::allocateTensorRTBufs(const std::map<std::string, std::string>& names,
                            const std::vector<int>& input_dims,
                            const std::vector<int>& output_dims) {

    std::map<std::string, std::vector<int>> trt_binding_dims;
    trt_binding_dims[names.at("input")] = input_dims;
    trt_binding_dims[names.at("output")] = output_dims;
    trt_buffers = allocateCudaTensors(trt_binding_dims);
}


void InferModelBase::allocateCudaBufs(const std::vector<int>& input_dims, const std::vector<int>& output_dims) {

    // Calculate the size of the input and output buffers
    size_t input_size = 1;
    for (int i = 1; i < input_dims.size(); i++) { // skip the batch dimension
        input_size *= input_dims[i];
    }

    size_t output_size = 1;
    for (int i = 1; i < output_dims.size(); i++) { // skip the batch dimension
        output_size *= output_dims[i];
    }

    // Allocate input and output buffers for CUDA
    cuda_input_buffers[0] = createZerosTensor<TensorType::FLOAT32>(input_size);
    cuda_input_buffers[1] = createZerosTensor<TensorType::FLOAT32>(input_size);
    cuda_output_buffers[0] = createZerosTensor<TensorType::FLOAT32>(output_size);
    cuda_output_buffers[1] = createZerosTensor<TensorType::FLOAT32>(output_size);
}


void InferModelBase::loadDataToEngine(const Tensor<float>& data, size_t size, int offset) {
    // Copy the data to the input buffer
    cudaMemcpy(trt_buffers[tensor_names.at("input")].ptr() + offset,
                data.ptr(),
                size,
                cudaMemcpyHostToDevice);
}


void InferModelBase::loadDataFromEngine(Tensor<float>& data, size_t size, int offset) {
    // Copy the data from the output buffer
    cudaMemcpy(data.ptr(),
                trt_buffers[tensor_names.at("output")].ptr() + offset,
                size,
                cudaMemcpyDeviceToHost);
}


void InferModelBase::fireEngine() {
    // Execute the TensorRT engine
    inference(context, trt_buffers[tensor_names.at("input")], trt_buffers[tensor_names.at("output")]);
}