//
// Created by user on 4/22/25.
//

#include <simple_cuda_toolkits/tensor_utils.hpp>
#include <cuda_runtime.h>
#include <fstream>

#include "trtengine/utils/logger.h"
#include "trtengine/serverlet/models/infer_model_multi.h"

#define DEBUG 0

InferModelBaseMulti::InferModelBaseMulti(
        const std::string& engine_path,
        const std::vector<TensorDefinition>& input_defs,
        const std::vector<TensorDefinition>& output_defs)
        : g_input_defs(input_defs), g_output_defs(output_defs)
{
    // 初始化 TRT 引擎
    g_ptr_engine = new TrtEngineMultiTs();

    // 加载模型并创建 Context
    if (!loadEngine(engine_path, g_input_defs, g_output_defs)) {
        LOG_ERROR("InferModelBaseMulti", "Failed to load engine: " + engine_path);
        exit(EXIT_FAILURE);
    }

    // 分配输入/输出 Buffer
    if (!allocateBufForTrtEngine(g_input_defs, g_output_defs)) {
        LOG_ERROR("InferModelBaseMulti", "Failed to allocate buffers.");
        exit(EXIT_FAILURE);
    }
}

InferModelBaseMulti::~InferModelBaseMulti() {
    // 修复：调整资源释放顺序，先释放 Tensor 内部的 CUDA 内存，再销毁 CUDA 流，最后删除引擎对象。
    // 1. 清理 map 中所有的 Tensor 对象，确保其内部的 CUDA 内存被释放。
    // 这需要在 TrtEngineMultiTs 对象（及其 TensorRT 上下文）存在时进行，以保证 CUDA 环境有效。
    g_map_trtTensors.clear();

    // 2. 删除 TrtEngineMultiTs 对象。
    // 它负责清理 TensorRT 的 IExecutionContext, ICudaEngine, IRuntime。
    if (g_ptr_engine) { // 检查是否为 nullptr，避免对已删除指针的操作
        delete g_ptr_engine;
        g_ptr_engine = nullptr; // 设为 nullptr 防止二次删除
    }

    LOG_VERBOSE_TOPIC("InferModelBaseMulti", "deconstructor", "All resources cleaned up in InferModelBaseMulti.");
}

bool InferModelBaseMulti::inference() {
    // 构造“指针列表”而非拷贝
    std::vector<Tensor<float>*> inputs, outputs;
    inputs.reserve(g_input_defs.size());
    outputs.reserve(g_output_defs.size());

    // 构造输入，输出列表
    for (auto& def : g_input_defs)
        inputs.push_back(&g_map_trtTensors[def.name]);
    for (auto& def : g_output_defs)
        outputs.push_back(&g_map_trtTensors[def.name]);

    // 这次 infer 绑定的就是 map 里原始的 Tensor，结果自然写回到它们
    if (!g_ptr_engine->infer(inputs, outputs)) {
        LOG_ERROR("InferModelBaseMulti", "Inference failed");
        return false;
    }

    return true;
}

bool InferModelBaseMulti::loadEngine(
        const std::string& engine_path,
        const std::vector<TensorDefinition>& input_defs,
        const std::vector<TensorDefinition>& output_defs) const
{
    if (!g_ptr_engine->loadFromFile(engine_path)) {
        LOG_ERROR_TOPIC("InferModelBaseMulti", "loadEngine", "Failed to load engine: " + engine_path);
        return false;
    }

    // 准备名称和维度
    std::vector<std::string> input_names;
    std::vector<nvinfer1::Dims4> input_dims;
    for (const auto& [name, dims] : input_defs) {
        if (dims.size() != 4) {
            LOG_ERROR_TOPIC("InferModelBaseMulti", "loadEngine", "Each input must be 4D (batch,C,H,W).");
            return false;
        }
        input_names.push_back(name);
        // 注意：Dims4 的构造函数是 Dims4(N, C, H, W)，对应 def.dims[0] 到 def.dims[3]
        input_dims.emplace_back(dims[0], dims[1], dims[2], dims[3]);
    }
    std::vector<std::string> output_names;
    for (const auto& [name, dims] : output_defs) {
        output_names.push_back(name);
    }

    if (!g_ptr_engine->createContext(input_names, input_dims, output_names)) {
        LOG_ERROR_TOPIC("InferModelBaseMulti", "loadEngine",
            "Failed to create context for engine: " + engine_path);
        return false;
    }
    return true;
}

bool InferModelBaseMulti::allocateBufForTrtEngine(
        const std::vector<TensorDefinition>& input_defs,
        const std::vector<TensorDefinition>& output_defs)
{
    try {
        // 输入 Buffers
        for (const auto& def : input_defs) {
            // 确保创建的 Tensor 具有正确的所有权语义，如果内部是原始指针，请确保没有浅拷贝问题
            g_map_trtTensors[def.name] = createZerosTensor<TensorType::FLOAT32>(def.dims);
        }
        // 输出 Buffers
        for (const auto& def : output_defs) {
            g_map_trtTensors[def.name] = createZerosTensor<TensorType::FLOAT32>(def.dims);
        }
        return !g_map_trtTensors.empty();
    } catch (const std::exception& e) {
        LOG_DEBUG_TOPIC("InferModelBaseMulti", "allocateBufsForTrtEngine",
            "Failed to allocate buffers for TRT engine: " + std::string(e.what()));
        return false;
    }
}

void InferModelBaseMulti::copyCpuDataToInputBuffer(
        const std::string& tensor_name,
        const std::vector<float>& input_data,
        int batch_idx)
{
    auto& tensor = g_map_trtTensors.at(tensor_name);
    const auto& dims = tensor.getDims();
    if (dims.empty() || dims[0] <= 0) {
        throw std::runtime_error("Invalid tensor dimensions for " + tensor_name);
    }
    auto total = tensor.elements();
    auto batch_count = static_cast<size_t>(dims[0]);
    auto single = total / batch_count;

    if (input_data.size() != single) {
        throw std::runtime_error("Input data size mismatch for " + tensor_name + ". Expected: " + std::to_string(single) + ", Got: " + std::to_string(input_data.size()));
    }
    auto offset = static_cast<size_t>(batch_idx) * single;

#if DEBUG
    {
        std::ofstream ofs(tensor_name + "_input_batch" + std::to_string(batch_idx) + ".csv");
        for (size_t i = 0; i < single; ++i) {
            ofs << input_data[i];
            if (i + 1 < single) ofs << ',';
        }
        ofs << '\n';
    }
#endif

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
    const auto& dims = tensor.getDims();
    if (dims.empty() || dims[0] <= 0) {
        throw std::runtime_error("Invalid tensor dimensions for " + tensor_name);
    }
    auto total = tensor.elements();
    auto batch_count = static_cast<size_t>(dims[0]);
    auto single = total / batch_count;

    // 如果 output_data 大小不匹配，需要重新分配
    if (output_data.size() != single) {
        throw std::runtime_error("Output data size mismatch for " + tensor_name);
    }
    auto offset = static_cast<size_t>(batch_idx) * single;

    cudaMemcpy(
            output_data.data(),
            tensor.ptr() + offset,
            sizeof(float) * single,
            cudaMemcpyDeviceToHost
    );

#if DEBUG
    {
        std::ofstream ofs(tensor_name + "_output_batch" + std::to_string(batch_idx) + ".csv");
        for (size_t i = 0; i < single; ++i) {
            ofs << output_data[i];
            if (i + 1 < single) ofs << ',';
        }
        ofs << '\n';
    }
#endif
}

// 获取指定张量在 CUDA 上指定 batch 索引的指针（const 版本）
const float* InferModelBaseMulti::accessCudaBufByBatchIdx(const std::string& tensor_name, int batch_idx) const
{
    // 检查张量映射表是否为空
    if (g_map_trtTensors.empty()) {
        throw std::runtime_error("No tensors available in g_map_trtTensors");
    }

    // 查找指定名称的张量
    auto it = g_map_trtTensors.find(tensor_name);
    if (it == g_map_trtTensors.end()) {
        throw std::runtime_error("Tensor " + tensor_name + " not found");
    }
    const auto& tensor = it->second;
    const auto& dims = tensor.getDims();

    // 检查张量维度是否合法
    if (dims.empty() || dims[0] <= 0) {
        throw std::runtime_error("Invalid tensor dimensions for " + tensor_name);
    }

    // 计算每个 batch 的元素数量
    auto total = tensor.elements();
    auto batch_count = static_cast<size_t>(dims[0]);
    auto single = total / batch_count;

    // 检查 batch 索引是否越界
    if (batch_idx < 0 || static_cast<size_t>(batch_idx) >= batch_count) {
        throw std::out_of_range("Batch index out of range for " + tensor_name + ". Requested: " + std::to_string(batch_idx) + ", Max: " + std::to_string(batch_count - 1));
    }

    // 返回指定 batch 的 CUDA 指针
    return tensor.ptr() + static_cast<size_t>(batch_idx) * single;
}
