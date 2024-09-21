//
// Created by vipuser on 8/23/24.
//

#ifndef INFER_ENGINE_LOADER_H
#define INFER_ENGINE_LOADER_H

#include <NvInferRuntime.h>

#include <vector>
#include <memory>
#include <map>

#include <simple_cuda_toolkits/tensor.hpp>

// Define unique pointers for ICudaEngine and IExecutionContext objects.
using ICudaEngineUniquePtr = std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)>;
using IExecutionContextUniquePtr = std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)>;

/**
 * @brief Loads a serialized TensorRT engine from a file.
 *
 * This function reads a serialized engine from the specified file and
 * deserializes it to create an ICudaEngine object, which is used to execute
 * inference on the GPU.
 *
 * @param engineFile The path to the engine file.
 * @return A unique pointer to the deserialized ICudaEngine, with a custom deleter
 *         to ensure proper cleanup.
 */
ICudaEngineUniquePtr loadEngineFromFile(const std::string& engineFile);

/**
 * @brief Loads a TensorRT engine from an ONNX file.
 *
 * @param onnxFile The path to the ONNX file.
 * @return A unique pointer to the ICudaEngine object.
 */
ICudaEngineUniquePtr loadEngineFromONNX(const std::string& onnxFile);

/**
 * @brief Creates an execution context for the given engine.
 *
 * This function creates an IExecutionContext object for the specified engine.
 * The execution context is used to execute inference on the GPU.
 *
 * @param engine A pointer to the ICudaEngine object.
 * @return A unique pointer to the IExecutionContext object, with a custom deleter
 *         to ensure proper cleanup.
 */
IExecutionContextUniquePtr
createExecutionContext(ICudaEngineUniquePtr& engine,
                       const std::string& input_name,
                       const nvinfer1::Dims4& input_shape);

/**
 * @brief Retrieves the names of all tensors in the model.
 *
 * This function queries the engine to retrieve the names of all tensors in the model.
 * The function returns a vector of strings containing the tensor names.
 *
 * @param engine A pointer to the ICudaEngine object.
 * @return A vector of strings containing the names of all tensors in the model.
 */
std::vector<std::string>
getTensorNamesFromModel(ICudaEngineUniquePtr& engine);

/**
 * Infer the model with the given input tensor and output tensor.
 * @param context
 * @param input
 * @param output
 */
void inference(IExecutionContextUniquePtr& context, Tensor<float>& input, Tensor<float>& output);


// Function to allocate tensors for CUDA computation based on key-value pairs.
// key: string (tensor name)
// value: std::vector<int> (tensor dimensions)
std::map<std::string, Tensor<float>> allocateCudaTensors(const std::map<std::string, std::vector<int>>& tensor_info);

#endif //INFER_ENGINE_LOADER_H
