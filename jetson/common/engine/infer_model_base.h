//
// Created by orlando on 12/19/24.
//

#ifndef INFER_MODEL_BASE_H
#define INFER_MODEL_BASE_H

#include <string>
#include <map>
#include <vector>
#include <any>
#include <memory>
#include <opencv2/opencv.hpp>
#include "common/engine/engine_loader.h"


class InferModelBase {

protected:

    // TensorRT engine and execution context
    ICudaEngineUniquePtr engine;
    IExecutionContextUniquePtr context;

    // Tensor names for input and output
    std::map<std::string, std::string> tensor_names;

    // Input and output buffers for TensorRT
    std::map<std::string, Tensor<float>> trt_buffers;

    // Input and output buffers for temporary storage
    std::map<int, Tensor<float>> cuda_input_buffers;
    std::map<int, Tensor<float>> cuda_output_buffers;

    // Input and output dimensions
    std::vector<int> input_dims;
    std::vector<int> output_dims;


public:

    // Constructor and destructor
    explicit InferModelBase(const std::string& engine_path,
                            const std::map<std::string, std::string>& names,
                            const std::vector<int>& input_dims,
                            const std::vector<int>& output_dims);

    // Destructor
    virtual ~InferModelBase();

    // Run inference (to be implemented by subclasses)
    virtual std::any infer(const cv::Mat& image) = 0;

protected:

    // Load the TensorRT engine and initialize buffers
    void loadEngine(const std::string& engine_path,
                    const std::map<std::string, std::string>& names,
                    const std::vector<int>& input_dims,
                    const std::vector<int>& output_dims);

    // Allocate buffers for TensorRT
    void allocateTensorRTBufs(const std::map<std::string, std::string>& names,
                                const std::vector<int>& input_dims,
                                const std::vector<int>& output_dims);

    // Allocate buffers for CUDA
    void allocateCudaBufs(const std::vector<int>& input_dims, 
                            const std::vector<int>& output_dims);

    // Load the data to the tensorrt engine
    void loadDataToEngine(const Tensor<float>& data, size_t size, int offset=0);

    // Load the data from the tensorrt engine
    void loadDataFromEngine(Tensor<float>& data, size_t size, int offset=0);

    // Launch the tensorrt engine
    void fireEngine();
};

#endif // INFER_MODEL_BASE_H
