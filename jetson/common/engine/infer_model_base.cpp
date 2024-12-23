//infer_model_base.cpp
#include "common/engine/infer_model_base.h"
#include "common/utils/logger.h"

#include "simple_cuda_toolkits/cores/cores.h"


InferModelBase::InferModelBase(
    const std::string& engine_path,
    const std::map<std::string, std::string>& names,
    const std::vector<int>& input_dims,
    const std::vector<int>& output_dims)
    : _tensor_names(names), _stream(nullptr) {

    // Create CUDA stream for asynchronous operations
    cudaError_t status = cudaStreamCreate(&_stream);
    if (status != cudaSuccess) {
        LOG_ERROR("InferModelBase", "Failed to create CUDA stream");
        exit(EXIT_FAILURE);
    }

    // Load the TensorRT engine
    if (!loadEngine(engine_path, names, input_dims)) {
        LOG_ERROR("InferModelBase", "Failed to load engine");
        exit(EXIT_FAILURE);
    }

    // Successful message
    LOG_VERBOSE_TOPIC("InferModelBase", "InferModelBase", "TensorRT engine loaded successfully.");

    // Initialize tensorrt buffers
    if (!allocateBufsForEngine(names, input_dims, output_dims)) {
        LOG_ERROR("InferModelBase", "Failed to allocate TensorRT buffers");
        exit(EXIT_FAILURE);
    }

    // Successful message
    LOG_VERBOSE_TOPIC("InferModelBase", "InferModelBase", "TensorRT buffers are ready.");

    // Allocate input and output buffers for CUDA
    if (!allocateBufsForCuda(input_dims, output_dims)) {
        LOG_ERROR("InferModelBase", "Failed to allocate CUDA buffers");
        exit(EXIT_FAILURE);
    }

    // Successful message
    LOG_VERBOSE_TOPIC("InferModelBase", "InferModelBase", "Temporary buffers for CUDA are ready.");
}

InferModelBase::~InferModelBase() {
    if (_stream) {
        cudaStreamDestroy(_stream);
        _stream = nullptr;
    }
}

bool InferModelBase::loadEngine(
    const std::string& engine_path,
    const std::map<std::string, std::string>& names,
    const std::vector<int>& input_dims) {

    // Initialize TrtEngineV8
    _engine = std::make_unique<TrtEngineV8>();

    // Load the Engine file
    if (!_engine->loadFromFile(engine_path)) {
        LOG_ERROR_TOPIC("InferModelBase", "loadEngine", "Failed to load engine from file.");
        return false;
    }

    if (input_dims.size() != 4) { // [batch, channel, width, height]
        LOG_ERROR_TOPIC("InferModelBase", "loadEngine",
            "Invalid input dimensions. Please specify the dimensions as [batch, channel, width, height]");
        return false;
    }

    nvinfer1::Dims4 dims4_input{input_dims[0], input_dims[1], input_dims[2], input_dims[3]};
    if (!_engine->createContext(names.at("input"), dims4_input)) {
        LOG_ERROR_TOPIC("InferModelBase", "loadEngine", "Failed to create execution context.");
        return false;
    }

    return true;
}

bool InferModelBase::allocateBufsForEngine(
    const std::map<std::string, std::string>& names,
    const std::vector<int>& input_dims,
    const std::vector<int>& output_dims) {
    
    std::map<std::string, std::vector<int>> tensor_info;
    tensor_info[names.at("input")] = input_dims;
    tensor_info[names.at("output")] = output_dims;
    
    try {
        _trt_buffers = _engine->allocateTensors(tensor_info);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR_TOPIC("InferModelBase", "allocateTensorRTBufs", e.what());
        return false;
    }
}

bool InferModelBase::allocateBufsForCuda(
    const std::vector<int>& input_dims,
    const std::vector<int>& output_dims) {
    
    try {
        // Calculate sizes skipping batch dimension
        size_t input_size = 1;
        for (size_t i = 1; i < input_dims.size(); i++) {
            input_size *= input_dims[i];
        }

        size_t output_size = 1;
        for (size_t i = 1; i < output_dims.size(); i++) {
            output_size *= output_dims[i];
        }

        // Allocate buffers
        _cuda_input_buffers[0] = createZerosTensor<TensorType::FLOAT32>(input_size);
        _cuda_input_buffers[1] = createZerosTensor<TensorType::FLOAT32>(input_size);
        _cuda_output_buffers[0] = createZerosTensor<TensorType::FLOAT32>(output_size);
        _cuda_output_buffers[1] = createZerosTensor<TensorType::FLOAT32>(output_size);
        
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR_TOPIC("InferModelBase", "allocateCudaBufs", e.what());
        return false;
    }
}

bool InferModelBase::fireEngine() {
    return _engine->infer(_trt_buffers[_tensor_names.at("input")],
                         _trt_buffers[_tensor_names.at("output")]);
}