//infer_model_base.h
#ifndef INFER_MODEL_BASE_H
#define INFER_MODEL_BASE_H

#include <string>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>

#include "serverlet/trt_engine/trt_engine.h"



class InferModelBase {
protected:
    // TensorRT engine
    TrtEngine* g_ptr_engine;

    // Input and output names for TensorRT
    std::string g_str_input, g_str_output;

    // Input and output dimensions for TensorRT
    std::vector<int> g_vec_inputDims, g_vec_outputDims;

    // Input and output buffers for TensorRT
    std::map<std::string, Tensor<float>> g_map_trtTensors;

public:
    // Constructor and destructor
    explicit InferModelBase(const std::string& engine_path,
                          const std::string& input_name,
                          const std::vector<int>& input_dims,
                          const std::string& output_name,
                          const std::vector<int>& output_dims);
    virtual ~InferModelBase();

    // Preprocess the image
    virtual void preprocess(const cv::Mat& image, int batchIdx) = 0;

    // Copy data to input buffer
    void copyCpuDataToInputBuffer(const std::vector<float>& input_data, int batch_idx=0);

    // Copy data to input buffer
    void copyCudaDataToInputBuffer(const Tensor<float>& input_data, int batch_idx=0);

    // Copy data from output buffer
    void copyCpuDataFromOutputBuffer(std::vector<float>& output_data, int batch_idx=0);

    // Copy data from output buffer
    void copyCudaDataFromOutputBuffer(Tensor<float>& output_data, int batch_idx=0);

    // Run inference (to be implemented by subclasses)
    bool inference();

    // Get input buffer
    const float* getInputBuffer()  { return g_map_trtTensors[g_str_input].ptr(); }

    // Get output buffer
    const float* getOutputBuffer() { return g_map_trtTensors[g_str_output].ptr(); }

protected:
    // Load the TensorRT engine and initialize buffers
    [[nodiscard]] bool loadEngine(const std::string& engine_path,
                  const std::string& input_name,
                  const std::vector<int>& input_dims) const;

    // Allocate buffers for TensorRT
    bool allocateBufsForTrtEngine(
        const std::string& input_name,
        const std::vector<int>& input_dims,
        const std::string& output_name,
        const std::vector<int>& output_dims);

    // // Allocate buffers for CUDA
    // bool allocateBufsForCuda(const std::vector<int>& input_dims,
    //                      const std::vector<int>& output_dims);
};

#endif // INFER_MODEL_BASE_H