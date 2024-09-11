#include <opencv2/opencv.hpp>
#include <csignal>
#include <stdexcept>
#include <iostream>
#include <filesystem>

#include "common/engine/engine_loader.h"
#include "common/infer/yolov8.h"

#define MODEL_PATH "/home/ubuntu/Documents/models/yolov8n.engine"
#define VIDEO_PATH "/home/ubuntu/Videos/highway_birdeye_01.mp4"
#define IMAGE_PATH "/home/ubuntu/Pictures/human_and_pets.png"

int main() {
    // Load the TensorRT engine from the file
    auto engine = loadEngineFromFile(MODEL_PATH);
    if (engine == nullptr) {
        std::cerr << "Failed to load engine from file" << std::endl;
        return 1;
    }

    // Create context from the engine
    auto context = createExecutionContext(engine);

    // Create CUDA buffers for input and output
    auto buffers = loadTensorsFromModel(engine);

    // Initialize the buffers for TensorRT model
    initCudaTemporaryBuffer(640, 640);

    // Load the image for inference
    cv::Mat image = cv::imread(IMAGE_PATH);
    if (image.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return 1;
    }

    // Preprocess the image
    preprocess(image, buffers["images"]);

    // Run inference
    inference(context, buffers["images"], buffers["output0"]);

    // Postprocess the output
    std::vector<YoloResult> results;
    postprocess(buffers["output0"], results, 0.1);

    // Draw the bounding boxes
    for (const auto& result : results) {
        auto lx = result.lx;
        auto ly = result.ly;
        auto rx = result.rx;
        auto ry = result.ry;
        auto cls = result.cls;
        auto conf = result.conf;

        cv::rectangle(image, cv::Point(lx, ly), cv::Point(rx, ry), cv::Scalar(0, 255, 0), 2);
        cv::putText(image, std::to_string(cls) + " " + std::to_string(conf), cv::Point(lx, ly), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }

    // Display the image
    cv::imshow("Image", image);
    cv::waitKey(0);

    return 0;
}