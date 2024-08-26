//
// Created by ubuntu on 9/10/24.
//

#ifndef JETSON_INFER_JETSONINFERENCE_H
#define JETSON_INFER_JETSONINFERENCE_H

#include <opencv2/opencv.hpp>
#include <map>
#include <chrono>
#include <csignal>
#include <atomic>
#include <stdexcept>
#include <memory>

#include "common/engine_loader.h"
#include "common/framework.h"
#include "common/MQTTClient.h"
#include "common/yaml_config.h"
#include "yolo/yolov8_utils.h"
#include "yolo/yolov8_str.h"
#include "protobufs/video_frame.pb.h"
#include "protobufs/inference_result.pb.h"

class JetsonInference {
public:
    explicit JetsonInference(const std::string& config_path);
    void run();
    static void signalHandler(int signum);

private:
    void receiveDataFrame(const VideoFrame& frame, cv::Mat &resized);
    void processMessage(const void* payload, size_t len);
    void postprocess(std::vector<YoloResult>& _results);
    void sendResults(const std::vector<YoloResult>& _results);

    YamlConfig config;
    std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> engine;
    std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)> context;
    std::map<std::string, CudaTensor<float>> gpu_tensors;
    MQTTClient mqtt;
    std::vector<YoloResult> results;

    static VideoFrame proto_frame;
    static InferenceResult proto_result;
    static cv::Mat resized_frame;
    static inline std::atomic<bool> is_running{true};

    void displayResults(cv::Mat &mat, std::vector<YoloResult> _results);
};

#endif //JETSON_INFER_JETSONINFERENCE_H
