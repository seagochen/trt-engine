#include <opencv2/opencv.hpp>
#include <iostream>

#include "include/c_inference.h"
#include "include/c_yolo_infer.h"
#include "include/c_vit_infer.h"


int yolov8_test(const std::string& model_path) {

    // initialize the model
    // if (c_model_init(model_path.c_str(), "yolov8n") == 0) {
    //     std::cout << "Failed to initialize the model" << std::endl;
    //     return -1;
    // }
    if (c_yolo_init(model_path.c_str()) == 0) {
        std::cout << "Failed to initialize the model" << std::endl;
        return -1;
    }

    // 读取图片
    cv::Mat img1 = cv::imread("/opt/images/boxing.png", cv::IMREAD_COLOR);
    if (!img1.data) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    cv::Mat img2 = cv::imread("/opt/images/using_laptop.png", cv::IMREAD_COLOR);
    if (!img2.data) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // Add the images to the input buffer
    if (c_add_image(0, img1.data, 3, img1.cols, img1.rows) == 0) {
        std::cout << "Failed to add the image to the input buffer" << std::endl;
        return -1;
    }
    if (c_add_image(1, img2.data, 3, img2.cols, img2.rows) == 0) {
        std::cout << "Failed to add the image to the input buffer" << std::endl;
        return -1;
    }

    // Run the inference
    if (c_do_inference() == 0) {
        std::cout << "Failed to run the inference" << std::endl;
        return -1;
    }

    // Get the output of the input #0
    std::cout << "output size: " << c_results_of_yolov8_obj(0, 0.4, 0.4) << std::endl;
    std::cout << "output size: " << c_results_of_yolov8_obj(1, 0.4, 0.4) << std::endl;

    // Release the model
    c_release_model();

    return 0;
}


int yolov8_pose_test(const std::string& model_path) {

    // initialize the model
    // if (c_model_init(model_path.c_str(), "yolov8n-pose") == 0) {
    //     std::cout << "Failed to initialize the model" << std::endl;
    //     return -1;
    // }
    if (c_pose_init(model_path.c_str()) == 0) {
        std::cout << "Failed to initialize the model" << std::endl;
        return -1;
    }

    // 读取图片
    cv::Mat img1 = cv::imread("/opt/images/boxing.png", cv::IMREAD_COLOR);
    if (!img1.data) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    cv::Mat img2 = cv::imread("/opt/images/using_laptop.png", cv::IMREAD_COLOR);
    if (!img2.data) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // Add the images to the input buffer
    if (c_add_image(0, img1.data, 3, img1.cols, img1.rows) == 0) {
        std::cout << "Failed to add the image to the input buffer" << std::endl;
        return -1;
    }
    if (c_add_image(1, img2.data, 3, img2.cols, img2.rows) == 0) {
        std::cout << "Failed to add the image to the input buffer" << std::endl;
        return -1;
    }

    // Run the inference
    if (c_do_inference() == 0) {
        std::cout << "Failed to run the inference" << std::endl;
        return -1;
    }

    // Get the output of the input #0
    std::cout << "output size: " << c_results_of_yolov8_pose(0, 0.4, 0.4) << std::endl;
    std::cout << "output size: " << c_results_of_yolov8_pose(1, 0.4, 0.4) << std::endl;

    // Release the model
    c_release_model();

    return 0;
}


#include "common/models/infer_yolov8_obj.h"

int load_model_test(const std::string& model_path) {

    // 创建 YOLO 模型实例
    InferYoloV8Obj model_yolo(
        model_path,
        "images", {2, 3, 640, 640},
        "output0", {2, 56, 8400}
    );

    // 读取图片
    cv::Mat img1 = cv::imread("/opt/images/boxing.png", cv::IMREAD_COLOR);
    if (!img1.data) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    cv::Mat img2 = cv::imread("/opt/images/using_laptop.png", cv::IMREAD_COLOR);
    if (!img2.data) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // 图片预处理
    model_yolo.preprocess(img1, 0);
    model_yolo.preprocess(img2, 1);

    // 模型推理
    model_yolo.inference();

    // NMS定义
    extern std::vector<Yolo> NMS(const std::vector<Yolo>& boxes, float iouThreshold);

    // 获取输出
    std::cout << "output size: " <<  NMS(model_yolo.postprocess(0), 0.4).size() << std::endl;
    std::cout << "output size: " << NMS(model_yolo.postprocess(1), 0.4).size() << std::endl;

    return 0;
}


int main() {
    load_model_test("/opt/models/yolov8n.engine");
    yolov8_test("/opt/models/yolov8n.engine");
    yolov8_pose_test("/opt/models/yolov8n-pose.engine");
    return 0;
}
