//infer_model_base.h
#ifndef INFER_MODEL_BASE_H
#define INFER_MODEL_BASE_H

#include <string>
#include <map>
#include <vector>
#include <any>
#include <memory>
#include <opencv2/opencv.hpp>
#include "common/engine/trt_engine_v8.h"

class InferModelBase {
protected:
    // TensorRT engine wrapper
    std::unique_ptr<TrtEngineV8> _engine;

    // Tensor names for input and output
    std::map<std::string, std::string> _tensor_names;

    // Input and output buffers for TensorRT
    std::map<std::string, Tensor<float>> _trt_buffers;

    // Input and output buffers for temporary storage
    std::vector<Tensor<float>> _cuda_input_buffers;
    std::vector<Tensor<float>> _cuda_output_buffers;

    // CUDA stream for asynchronous operations
    cudaStream_t _stream;

public:
    // Constructor and destructor
    explicit InferModelBase(const std::string& engine_path,
                          const std::map<std::string, std::string>& names,
                          const std::vector<int>& input_dims,
                          const std::vector<int>& output_dims);
    virtual ~InferModelBase();

    // Run inference (to be implemented by subclasses)
    virtual std::any infer(const cv::Mat& image) = 0;

protected:
    // Load the TensorRT engine and initialize buffers
    bool loadEngine(const std::string& engine_path,
                   const std::map<std::string, std::string>& names,
                   const std::vector<int>& input_dims);

    // Allocate buffers for TensorRT
    bool allocateBufsForEngine(const std::map<std::string, std::string>& names,
                             const std::vector<int>& input_dims,
                             const std::vector<int>& output_dims);

    // Allocate buffers for CUDA
    bool allocateBufsForCuda(const std::vector<int>& input_dims,
                         const std::vector<int>& output_dims);

    // Launch inference with error checking
    bool fireEngine();
};

#endif // INFER_MODEL_BASE_H