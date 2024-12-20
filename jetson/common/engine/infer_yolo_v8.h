//
// Created by orlando on 9/20/24.
//

#ifndef INFER_YOLO_V8_H
#define INFER_YOLO_V8_H

#include <string>
#include <memory>
#include <map>
#include <vector>

#include <opencv2/opencv.hpp>
#include <simple_cuda_toolkits/tensor.hpp>

#include "engine_loader.h"

#define MAX_BATCH_SIZE 4

class InferYoloV8 {

    // Engines and contexts for TensorRT
    ICudaEngineUniquePtr engine;
    IExecutionContextUniquePtr context;

    // Tensor names for input and output
    std::map<std::string, std::string> tensor_names;

    // Input and output buffers for CUDA and TensorRT
    std::map<std::string, Tensor<float>> trt_buffers;
    std::map<int, Tensor<float>> cuda_input_buffers;
    std::map<int, Tensor<float>> cuda_output_buffers;

    // The storing buffer for processed results, which are ready to be copied to the CPU
    std::vector<Tensor<float>> results;

    // Input and output temporary images for OpenCV
    std::map<std::string, cv::Mat> temp_images;

    // Dimensions of input and output tensors
    std::vector<int> input_dims;
    std::vector<int> output_dims;

    // Index for counting the preprocessed images and available output results
    int image_idx = 0;
    int boxes = 0;

    // Ptr for CPU copying
    std::vector<float> raw_output;

public:
    InferYoloV8();

    InferYoloV8(const std::string &engine_path,            // File path for loading the engine file
        const std::map<std::string, std::string> &names,    // Names of input and output tensors
        const nvinfer1::Dims4 &input_dims,                  // Dimensions of input tensor
        const nvinfer1::Dims3 &output_dims,                 // Dimensions of output tensor
        int boxes=1024);                                    // Number of boxes for detection

    ~InferYoloV8();

    void update(const std::string &engine_path,
        const std::map<std::string, std::string> &names,
        const nvinfer1::Dims4 &input_dims,
        const nvinfer1::Dims3 &output_dims,
        int boxes);

    /**
    * @brief Preprocess input image for inference
    * @param image Input image for inference
    */
    void addImage(const cv::Mat &image, bool isRGB=false);

    /**
     * @brief Perform inference on the input images
     */
    void inferObjectDetection(float cls_threshold, float nms_threshold, float alpha=0.f, float beta=640.f);

    /**
     * @brief Perform inference on the input images
     */
    void inferPoseEstimation(float cls_threshold, float nms_threshold, float alpha=0.f, float beta=640.f);

    /**
     * @brief Get the available slots count for storing preprocessed images
     * @return Number of available slots
     */
    [[nodiscard]] int getAvailableSlot() const;

    /**
     * @brief When the inference is done, get the results.
     * @tparam T The type of result elements.
     * @param idx Index of the result to retrieve.
     * @param callback A callback function to process the raw output.
     * @return A vector of results of type T.
     * @throws std::runtime_error if the index is invalid.
     */
    template<typename T>
    std::vector<T> getResults(int idx, std::vector<T>(*callback)(float*, int, int)) {
        // Ensure the index is within valid bounds.
        if (idx < 0 || idx >= static_cast<int>(results.size())) {
            throw std::runtime_error("Invalid index for getting results.");
        }

        // Copy the results to CPU.
        results[idx].copyTo(raw_output);

        // Invoke the callback function to process the results.
        return callback(raw_output.data(), boxes, output_dims[1]); // data, boxes, features
    }
};

#endif //INFER_YOLO_V8_H