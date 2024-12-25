#include "common/models/engine/trt_engine.h"
#include "common/models/infer_model_base.h"
#include "common/utils/logger.h"

#include "simple_cuda_toolkits/tensor.hpp"
#include "simple_cuda_toolkits/tensor_utils.hpp"

InferModelBase::InferModelBase(const std::string &engine_path,
    const std::string &input_name,
    const std::vector<int> &input_dims,
    const std::string &output_name,
    const std::vector<int> &output_dims) {

    // Keep track of input/output names and dimensions
    g_str_input = input_name;
    g_str_output = output_name;
    g_vec_inputDims = input_dims;
    g_vec_outputDims = output_dims;

    // Create TensorRT engine
    g_ptr_engine = new TrtEngine();

    // Load engine from file
    if (!loadEngine(engine_path, input_name, input_dims)) {
        LOG_ERROR("InferModelBase", "Failed to load engine from file: " + engine_path);
        exit(EXIT_FAILURE);
    }
    LOG_VERBOSE("InferModelBase", "Engine loaded from file: " + engine_path);

    // Allocate buffers for TensorRT engine
    if (!allocateBufsForTrtEngine(input_name, input_dims, output_name, output_dims)) {
        LOG_ERROR("InferModelBase", "Failed to allocate buffers for TensorRT engine");
        exit(EXIT_FAILURE);
    }
    LOG_VERBOSE("InferModelBase", "Buffers allocated for TensorRT engine");
}


InferModelBase::~InferModelBase() {
    if (g_ptr_engine) {
        delete g_ptr_engine;
        g_ptr_engine = nullptr;
    }

    if (!g_map_trtTensors.empty()) {
        g_map_trtTensors.clear();
    }
}


bool InferModelBase::loadEngine(const std::string& engine_path,
    const std::string& input_name,
    const std::vector<int>& input_dims) const {

    // Load engine from file
    if (!g_ptr_engine->loadFromFile(engine_path)) {
        LOG_ERROR("InferModelBase::loadEngine", "Failed to load engine from file: " + engine_path);
        return false;
    }

    // Validate input dimensions
    if (input_dims.size() != 4) {
        LOG_ERROR("InferModelBase::loadEngine", "Invalid input dimensions, (batch, channel, height, weight).");
        return false;
    }

    // Create engine context
    if (const nvinfer1::Dims4 cls_dims4{input_dims[0], input_dims[1], input_dims[2], input_dims[3]};
        !g_ptr_engine->createContext(input_name, cls_dims4)) {
        LOG_ERROR("InferModelBase::loadEngine", "Failed to create engine context");
        return false;
    }

    return true;
}


bool InferModelBase::allocateBufsForTrtEngine(const std::string &input_name,
    const std::vector<int> &input_dims,
    const std::string &output_name,
    const std::vector<int> &output_dims) {

    try {
        g_map_trtTensors[input_name] = createZerosTensor<TensorType::FLOAT32>(input_dims);
        g_map_trtTensors[output_name] = createZerosTensor<TensorType::FLOAT32>(output_dims);
        return true;
    } catch (const std::exception &e) {
        LOG_ERROR("InferModelBase::allocateBufsForTrtEngine", e.what());
        g_map_trtTensors.clear();
        return false;
    }
}


bool InferModelBase::inference() {
    return g_ptr_engine->infer(g_map_trtTensors[g_str_input], g_map_trtTensors[g_str_output]);
}


void InferModelBase::copyCudaDataToInputBuffer(const Tensor<float>& input_data, int const batch_idx) {
    // 计算单个样本的大小（不包括批次维度）
    const auto single_sample_size = std::accumulate(g_vec_inputDims.begin() + 1, g_vec_inputDims.end(), 1, std::multiplies());

    // 校验输入数据的大小是否匹配单个样本大小
    if (input_data.elements() != static_cast<size_t>(single_sample_size)) {
        throw std::runtime_error("Input data size does not match the size of a single batch sample.");
    }

    // 计算批次偏移量
    const auto offset = batch_idx * single_sample_size;

    // 拷贝数据到对应的偏移位置
    cudaMemcpy(g_map_trtTensors[g_str_input].ptr() + offset, input_data.ptr(),
               sizeof(float) * input_data.elements(), cudaMemcpyDeviceToDevice);
}


void InferModelBase::copyCudaDataFromOutputBuffer(Tensor<float>& output_data, int const batch_idx) {
    // 计算单个样本的大小（不包括批次维度）
    const auto single_sample_size = std::accumulate(g_vec_outputDims.begin() + 1, g_vec_outputDims.end(), 1, std::multiplies());

    // 校验输出数据的大小是否匹配单个样本大小
    if (output_data.elements() != static_cast<size_t>(single_sample_size)) {
        throw std::runtime_error("Output data size does not match the size of a single batch sample.");
    }

    // 计算批次偏移量
    const auto offset = batch_idx * single_sample_size;

    // 从对应的偏移位置拷贝数据到输出张量
    cudaMemcpy(output_data.ptr(), g_map_trtTensors[g_str_output].ptr() + offset,
               sizeof(float) * output_data.elements(), cudaMemcpyDeviceToDevice);
}


void InferModelBase::copyCpuDataToInputBuffer(const std::vector<float>& input_data, int const batch_idx) {
    // 计算单个样本的大小（不包括批次维度）
    const auto single_sample_size = std::accumulate(g_vec_inputDims.begin() + 1, g_vec_inputDims.end(), 1, std::multiplies());

    // 校验输入数据的大小是否匹配单个样本大小
    if (input_data.size() != static_cast<size_t>(single_sample_size)) {
        throw std::runtime_error("Input data size does not match the size of a single batch sample.");
    }

    // 计算批次偏移量
    const auto offset = batch_idx * single_sample_size;

    // 拷贝数据到对应的偏移位置
    cudaMemcpy(g_map_trtTensors[g_str_input].ptr() + offset, input_data.data(),
               sizeof(float) * input_data.size(), cudaMemcpyHostToDevice);
}


void InferModelBase::copyCpuDataFromOutputBuffer(std::vector<float>& output_data, int const batch_idx) {
    // 计算单个样本的大小（不包括批次维度）
    const auto single_sample_size = std::accumulate(g_vec_outputDims.begin() + 1, g_vec_outputDims.end(), 1, std::multiplies());

    // 校验输出数据的大小是否匹配单个样本大小
    if (output_data.size() != static_cast<size_t>(single_sample_size)) {
        throw std::runtime_error("Output data size does not match the size of a single batch sample.");
    }

    // 计算批次偏移量
    const auto offset = batch_idx * single_sample_size;

    // 从对应的偏移位置拷贝数据到输出向量
    cudaMemcpy(output_data.data(), g_map_trtTensors[g_str_output].ptr() + offset,
               sizeof(float) * output_data.size(), cudaMemcpyDeviceToHost);
}
