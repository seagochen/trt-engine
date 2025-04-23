//
// Created by user on 4/22/25.
//

#include "simple_cuda_toolkits/tensor_utils.hpp"
#include "serverlet/utils/logger.h"
#include "serverlet/models/infer_model_multi.h"
#include <cuda_runtime.h>

InferModelBaseMulti::InferModelBaseMulti(
        const std::string& engine_path,
        const std::vector<TensorDefinition>& input_defs,
        const std::vector<TensorDefinition>& output_defs)
        : g_input_defs(input_defs), g_output_defs(output_defs)
{
    // 初始化 TRT 引擎
    g_ptr_engine = new TrtEngineMultiTs();

    // 创建 CUDA 流
    if (cudaStreamCreate(&g_stream) != cudaSuccess) {
        LOG_ERROR("InferModelBase", "Failed to create CUDA stream");
    }

    // 加载模型并创建 Context
    if (!loadEngine(engine_path, g_input_defs, g_output_defs)) {
        LOG_ERROR("InferModelBase", "Failed to load engine: " + engine_path);
        exit(EXIT_FAILURE);
    }

    // 分配输入/输出 Buffer
    if (!allocateBufsForTrtEngine(g_input_defs, g_output_defs)) {
        LOG_ERROR("InferModelBase", "Failed to allocate buffers");
        exit(EXIT_FAILURE);
    }
}

InferModelBaseMulti::~InferModelBaseMulti() {
    delete g_ptr_engine;
    g_map_trtTensors.clear();
    cudaStreamDestroy(g_stream);
}

bool InferModelBaseMulti::inference() {
    // 构造输入列表
    std::vector<Tensor<float>> inputs;
    inputs.reserve(g_input_defs.size());
    for (const auto& def : g_input_defs) {
        inputs.emplace_back(g_map_trtTensors[def.name]);
    }

    // 构造输出列表
    std::vector<Tensor<float>> outputs;
    outputs.reserve(g_output_defs.size());
    for (const auto& def : g_output_defs) {
        outputs.emplace_back(g_map_trtTensors[def.name]);
    }

    // 执行推理
    if (!g_ptr_engine->infer(inputs, outputs, g_stream)) {
        LOG_ERROR("InferModelBase", "Inference failed");
        return false;
    }
    cudaStreamSynchronize(g_stream);
    return true;
}

bool InferModelBaseMulti::loadEngine(
        const std::string& engine_path,
        const std::vector<TensorDefinition>& input_defs,
        const std::vector<TensorDefinition>& output_defs)
{
    if (!g_ptr_engine->loadFromFile(engine_path)) {
        LOG_ERROR("loadEngine", "Failed to load engine: " + engine_path);
        return false;
    }

    // 准备名称和维度
    std::vector<std::string> input_names;
    std::vector<nvinfer1::Dims4> input_dims;
    for (const auto& def : input_defs) {
        if (def.dims.size() != 4) {
            LOG_ERROR("loadEngine", "Each input must be 4D (batch,C,H,W)");
            return false;
        }
        input_names.push_back(def.name);
        input_dims.emplace_back(def.dims[0], def.dims[1], def.dims[2], def.dims[3]);
    }
    std::vector<std::string> output_names;
    for (const auto& def : output_defs) {
        output_names.push_back(def.name);
    }

    if (!g_ptr_engine->createContext(input_names, input_dims, output_names)) {
        LOG_ERROR("loadEngine", "createContext failed");
        return false;
    }
    return true;
}

bool InferModelBaseMulti::allocateBufsForTrtEngine(
        const std::vector<TensorDefinition>& input_defs,
        const std::vector<TensorDefinition>& output_defs)
{
    try {
        // 输入 Buffers
        for (const auto& def : input_defs) {
            g_map_trtTensors[def.name] = createZerosTensor<TensorType::FLOAT32>(def.dims);
        }
        // 输出 Buffers
        for (const auto& def : output_defs) {
            g_map_trtTensors[def.name] = createZerosTensor<TensorType::FLOAT32>(def.dims);
        }
        return !g_map_trtTensors.empty();
    } catch (const std::exception& e) {
        LOG_ERROR("allocateBufsForTrtEngine", e.what());
        return false;
    }
}

void InferModelBaseMulti::copyCpuDataToInputBuffer(
        const std::string& tensor_name,
        const std::vector<float>& input_data,
        int batch_idx)
{
    auto& tensor = g_map_trtTensors.at(tensor_name);
    auto dims = tensor.getDims();
    size_t total = tensor.elements();
    size_t batch_count = static_cast<size_t>(dims[0]);
    size_t single = total / batch_count;

    if (input_data.size() != single) {
        throw std::runtime_error("Input data size mismatch for " + tensor_name);
    }
    size_t offset = static_cast<size_t>(batch_idx) * single;
    cudaMemcpy(
            tensor.ptr() + offset,
            input_data.data(),
            sizeof(float) * single,
            cudaMemcpyHostToDevice
    );
}

void InferModelBaseMulti::copyCpuDataFromOutputBuffer(
        const std::string& tensor_name,
        std::vector<float>& output_data,
        int batch_idx)
{
    auto& tensor = g_map_trtTensors.at(tensor_name);
    auto dims = tensor.getDims();
    size_t total = tensor.elements();
    size_t batch_count = static_cast<size_t>(dims[0]);
    size_t single = total / batch_count;

    if (output_data.size() != single) {
        throw std::runtime_error("Output data size mismatch for " + tensor_name);
    }
    size_t offset = static_cast<size_t>(batch_idx) * single;
    cudaMemcpy(
            output_data.data(),
            tensor.ptr() + offset,
            sizeof(float) * single,
            cudaMemcpyDeviceToHost
    );
}
