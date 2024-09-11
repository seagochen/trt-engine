//
// Created by vipuser on 8/23/24.
//

#ifndef JETSON_INFER_ENGINE_LOADER_H
#define JETSON_INFER_ENGINE_LOADER_H

#include <NvInferRuntime.h>

#include <vector>
#include <memory>
#include <map>

#include "tensor_base.hpp"
#include "tensor_cuda.hpp"

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
std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)>
        loadEngineFromFile(const std::string& engineFile);

/**
 * @brief Loads a TensorRT engine from an ONNX file.
 *
 * @param onnxFile The path to the ONNX file.
 * @return A unique pointer to the ICudaEngine object.
 */
std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)>
        loadEngineFromONNX(const std::string& onnxFile);

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
std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)>
        createExecutionContext(std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)>& engine);

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
        getTensorNamesFromModel(std::unique_ptr<nvinfer1::ICudaEngine,
                                void(*)(nvinfer1::ICudaEngine*)>& engine);

/**
 * @brief Retrieves the dimensions of a tensor by name.
 *
 * This function queries the engine to retrieve the dimensions of a tensor by name.
 * The function returns a TensorDimensions object containing the dimensions of the tensor.
 *
 * @param engine A pointer to the ICudaEngine object.
 * @param tensor_name The name of the tensor.
 * @return A TensorDimensions object containing the dimensions of the tensor.
 */
TensorDimensions
getTensorDimsByName(std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)>& engine,
                    const std::string& tensor_name, tensor_type type);

/**
 * Infer the model with the given input tensor and output tensor.
 * @param context
 * @param input
 * @param output
 */
void inference(std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)>& context,
               CudaTensor<float>& input, CudaTensor<float>& output);

/**
 * Load all tensors from the model.
 * @param engine
 * @return
 */
std::map<std::string, CudaTensor<float>>
loadTensorsFromModel(std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)>& engine);

#endif //JETSON_INFER_ENGINE_LOADER_H
