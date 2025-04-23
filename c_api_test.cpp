//
// Created by user on 4/8/25.
//

#include "serverlet/c_yolo_v8_apis.h"
#include "serverlet/models/efficient_net/efficient_net.h"

#include <opencv2/opencv.hpp>

int c_yolo_api_test() {

    // Initialize the model
    c_yolo_init("/opt/models/yolov8n.engine");

    // Load the image, and check if it was loaded successfully
    cv::Mat image = cv::imread("./test.png", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        c_yolo_release();
        return -1;
    }

    // Load the image to the model
    c_yolo_add_image(0, image.data, 3, image.cols, image.rows);

    // Run inference
    if (!c_yolo_inference()) {
        std::cerr << "Error: Inference failed." << std::endl;
        c_yolo_release();
    }

    // Print out the counts available
    int count_results = c_yolo_available_results(0, 0.4, 0.4);
    std::cout << "Count of results: " << count_results << std::endl;

    // Get the first detected object from the model
    auto fptr_results = c_yolo_get_result(0);
    for (int i = 0; i < LEN_YOLO_ENTITY; i++) {
        std::cout << fptr_results[i] << "\t";
    }
    std::cout << std::endl;

    // Release the model
    c_yolo_release();
    return 0;
}


int infer_model_multi_test() {

    // 使用C++11的特性，直接用 aggregate-init + initializer_list 创建 engine 所需要的参数
    // std::vector<TensorDefinition> input_ts = {
    //         { "input", {1, 3, 224, 224}}
    // };
    // std::vector<TensorDefinition> output_ts = {
    //         {"logits", {1, 2}},
    //         {"feat", {1, 256}}
    // };

    // Load the engine from file
    EfficientNetForFeatAndClassification efficient_net("/opt/models/efficientnet_b0_feat_logits.engine", 2);

    // Load the image, and check if it was loaded successfully
    cv::Mat image1 = cv::imread("/opt/images/0_1.bmp", cv::IMREAD_COLOR);
    cv::Mat image2 = cv::imread("/opt/images/1_4.bmp", cv::IMREAD_COLOR);

    // Check if the images were loaded successfully
    if (image1.empty() || image2.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    // Process the image1
    efficient_net.preprocess(image1, 0);
    efficient_net.preprocess(image2, 1);

    // Inference
    if (!efficient_net.inference()) {
        std::cerr << "Error: Inference failed." << std::endl;
        return -1;
    }

    // Process the image2
    auto vec1 = efficient_net.postprocess(0);
    std::cout << "Classification result for image1: " << vec1[0] << std::endl;
    auto vec2 = efficient_net.postprocess(1);
    std::cout << "Classification result for image2: " << vec2[0] << std::endl;

    return 0;
}


int main() {

    // Test the C YOLO API
    c_yolo_api_test();

    // Test the EfficientNet model
    infer_model_multi_test();
}