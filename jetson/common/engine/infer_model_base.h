//
// Created by orlando on 12/19/24.
//

#ifndef INFER_MODEL_BASE_H
#define INFER_MODEL_BASE_H

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "engine_loader.h"

#define MAX_BATCH_SIZE 4


class InferModelBase {

protected:
    // TensorRT engine and execution context
    ICudaEngineUniquePtr engine;
    IExecutionContextUniquePtr context;

    // Tensor names for input and output
    std::map<std::string, std::string> tensor_names;

    // Input and output buffers for TensorRT
    std::map<std::string, Tensor<float>> trt_buffers;

    // Input and output dimensions
    std::vector<int> input_dims;
    std::vector<int> output_dims;

    // Maximum batch size
    int max_batch_size;

public:

    explicit InferModelBase(int max_batch_size = MAX_BATCH_SIZE);

    virtual ~InferModelBase();

    // Load the TensorRT engine and initialize buffers
    void loadEngine(const std::string& engine_path,
                    const std::map<std::string, std::string>& names,
                    const std::vector<int>& input_dims,
                    const std::vector<int>& output_dims);

    // Preprocess the input image (to be implemented by subclasses)
    virtual void preprocess(const cv::Mat& image) = 0;

    // Run inference (to be implemented by subclasses)
    virtual void infer(const std::map<std::string, float>& params) = 0;

    // Postprocess the output (to be implemented by subclasses)
    virtual std::vector<float> postprocess(int idx) = 0;

protected:
    void allocateBuffers();

    void copyToDevice(const cv::Mat& image, const std::string& input_tensor_name);

};

#endif // INFER_MODEL_BASE_H
