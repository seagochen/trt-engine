//
// Created by user on 6/13/25.
//

#include <opencv2/opencv.hpp>
#include "serverlet/models/common/yolo_drawer.h"
#include "serverlet/models/inference/model_init_helper.hpp"

// 示例用法
#include <iostream>



void test_yolo_pose()
{
    // 定义模型参数
    std::map<std::string, std::any> params;
    int maximum_batch = 4; // 明确批处理大小
    params["maximum_batch"] = maximum_batch;
    params["maximum_items"] = 100;
    params["infer_features"] = 56;
    params["infer_samples"] = 8400;

    // 用于postprocess的参数
    params["cls"] = 0.4f;
    params["iou"] = 0.5f;

    // 创建 YOLOv8 姿态估计模型
    std::unique_ptr<InferModelBaseMulti> pose_model = ModelFactory::createModel(
        "YoloV8_Pose", "/opt/models/yolov8s-pose.engine", params
    );

    // 检查模型是否成功创建
    if (pose_model) {
        std::cout << "YOLOv8 Pose Estimation Model created successfully." << std::endl;
    } else {
        std::cerr << "Failed to create YOLOv8 Pose Estimation Model." << std::endl;
        return;
    }

    // --- 加载不同的图片 ---
    std::vector<std::string> image_paths = {
        "/opt/images/human_and_pets.png",
        "/opt/images/apples.png",
        "/opt/images/cartoon.png",
        "/opt/images/pedestrian.png"
    };

    std::vector<cv::Mat> original_images(maximum_batch);
    for (int i = 0; i < maximum_batch; ++i) {
        if (i < image_paths.size()) {
            original_images[i] = cv::imread(image_paths[i]);
            if (original_images[i].empty()) {
                std::cerr << "Failed to load image: " << image_paths[i] << std::endl;
                // 如果图片加载失败，可以考虑跳过或用一个占位符图片
                // 这里我们直接返回，实际应用中可能需要更健壮的错误处理
                return;
            }
            std::cout << "Loaded image: " << image_paths[i] << std::endl;
        } else {
            // 如果图片数量少于 maximum_batch，可以重复使用最后一张图片或跳过
            std::cerr << "Warning: Not enough unique images for batch size " << maximum_batch << ". Reusing images." << std::endl;
            original_images[i] = original_images.back(); // 重复最后一张
        }
    }

    // --- 预处理所有图片，并将其拷贝到对应的批次索引 ---
    for (int i = 0; i < maximum_batch; ++i) {
        pose_model->preprocess(original_images[i], i);
    }

    // --- 执行推理 ---
    pose_model->inference();

    // --- 后处理和显示每张图片的结果 ---
    YoloDrawer drawer; // 只需要一个绘图器实例

    for (int i = 0; i < maximum_batch; ++i) {
        std::any raw_results;
        // 对每个批次索引调用 postprocess
        pose_model->postprocess(i, params, raw_results);

        try {
            std::vector<YoloPose> current_batch_results = std::any_cast<std::vector<YoloPose>>(raw_results);

            // 获取当前批次的原始图片进行绘制
            cv::Mat display_image = original_images[i].clone(); // 克隆一份，避免修改原始图片
            
            // 对图片resize到模型期望的输入大小
            cv::resize(display_image, display_image, cv::Size(640, 640));

            // 绘制当前批次的姿态估计结果
            drawer.drawPoses(display_image, current_batch_results);

            // 显示结果
            std::string window_name = "Pose Detection Result - Image " + std::to_string(i + 1);
            cv::imshow(window_name, display_image);

        } catch (const std::bad_any_cast& e) {
            std::cerr << "Error: Failed to cast postprocess result for batch " << i << ": " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "An unexpected error occurred during YOLOv8 postprocessing for batch " << i << ": " << e.what() << std::endl;
        }
    }

    cv::waitKey(0); // 等待按键事件，直到所有窗口被关闭
    cv::destroyAllWindows();
    std::cout << "Pose detection completed and results displayed for all images." << std::endl;
}


void test_yolo_detection() 
{
    // 定义模型参数
    std::map<std::string, std::any> params;
    params["maximum_batch"] = 1;
    params["maximum_items"] = 100;
    params["infer_features"] = 84;
    params["infer_samples"] = 8400;

    // 用于postprocess的参数
    params["cls"] = 0.4f;
    params["iou"] = 0.5f;

    // 创建 YOLOv8 姿态估计模型
    std::unique_ptr<InferModelBaseMulti> detection_model = ModelFactory::createModel(
        "YoloV8_Detection", "/opt/models/yolov8s.engine", params
    );

    // 检查模型是否成功创建
    if (detection_model) {
        std::cout << "YOLOv8 Detection Model created successfully." << std::endl;
    } else {
        std::cerr << "Failed to create YOLOv8 Detection Model." << std::endl;
        return;
    }

    // --- 加载图片 ---
    std::string image_path = "/opt/images/india_road.png";
    cv::Mat original_image = cv::imread(image_path);
    if (original_image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return; // 如果图片加载失败，可以考虑跳过或用一个占位符图片
    }

    // --- 预处理图片，并将其拷贝到对应的批次索引 ---
    detection_model->preprocess(original_image, 0);
    std::cout << "Image preprocessed and copied to batch index 0." << std::endl;

    // --- 执行推理 ---
    if (detection_model->inference()) {
        std::cout << "Inference executed successfully." << std::endl;
    } else {
        std::cerr << "Inference failed." << std::endl;
        return;
    }

    // --- 后处理和显示结果 ---
    YoloDrawer drawer; // 只需要一个绘图器实例
    std::any raw_results;
    detection_model->postprocess(0, params, raw_results);

    try {
        std::vector<Yolo> detection_results = std::any_cast<std::vector<Yolo>>(raw_results);

        // 在这里处理检测结果，例如绘制边界框
        cv::Mat display_image = original_image.clone(); // 克隆一份，避免修改原始图片
        cv::resize(display_image, display_image, cv::Size(640, 640)); // 调整大小到模型期望的输入大小

        // 绘制检测结果
        drawer.drawBoundingBoxes(display_image, detection_results);

        // 显示结果
        cv::imshow("YOLOv8 Detection Result", display_image);
        cv::waitKey(0);

    } catch (const std::bad_any_cast& e) {
        std::cerr << "Error: Failed to cast postprocess result: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "An unexpected error occurred during YOLOv8 postprocessing: " << e.what() << std::endl;
    }

    cv::destroyAllWindows();
    std::cout << "Detection completed and results displayed." << std::endl;

    // 执行超 maximum_batch 测试
    detection_model->preprocess(original_image, 1); // 尝试使用批次索引 1
    detection_model->preprocess(original_image, 2); // 尝试使用批次索引 2
    detection_model->preprocess(original_image, 3); // 尝试使用批次索引 3
    detection_model->preprocess(original_image, 4); // 尝试使用批次索引 4
}


void test_yolo_pose_efficient()
{
    // // 定义模型参数
    // std::map<std::string, std::any> params;
    // int maximum_batch = 4; // 明确批处理大小
    // params["maximum_batch"] = 1;
    // params["maximum_items"] = 100;
    // params["infer_features"] = 56;
    // params["infer_samples"] = 8400;

    // // 用于postprocess的参数
    // params["cls"] = 0.4f;
    // params["iou"] = 0.5f;

    // // 创建 YOLOv8 姿态估计模型
    // std::unique_ptr<InferModelBaseMulti> pose_model = ModelFactory::createModel(
    //     "YoloV8_Pose", "/opt/models/yolov8s-pose.engine", params
    // );
}

void unknown_model_test() {
    // 尝试创建不存在的模型
    std::unique_ptr<InferModelBaseMulti> unknown_model = ModelFactory::createModel(
        "UnknownModel", "path/to/unknown_engine.trt", {}
    );

    if (!unknown_model) {
        std::cout << "Correctly failed to create UnknownModel." << std::endl;
    } else {
        std::cout << "Unexpectedly created UnknownModel." << std::endl;
    }
}

int main() {

    // 注册所有模型
    registerModels();

    // 黄色输出前缀和后缀
    const std::string YELLOW = "\033[33m";
    const std::string RESET = "\033[0m";
    const std::string YELLOW_LINE = YELLOW + "-------------------" + RESET;

    // 测试 YOLOv8 姿态估计模型
    std::cout << YELLOW << "Detecting poses using YOLOv8..." << RESET << std::endl;
    test_yolo_pose();
    std::cout << YELLOW_LINE << std::endl;

    // 测试 YOLOv8 检测模型
    std::cout << YELLOW << "Detecting objects using YOLOv8..." << RESET << std::endl;
    test_yolo_detection();
    std::cout << YELLOW_LINE << std::endl;

    // 测试 YOLOv8 + EfficientNet 模型
    std::cout << YELLOW << "Testing YOLOv8 + EfficientNet..." << RESET << std::endl;
    test_yolo_pose_efficient();
    std::cout << YELLOW_LINE << std::endl;

    // 尝试创建不存在的模型
    std::cout << YELLOW << "Testing unknown model creation..." << RESET << std::endl;
    unknown_model_test();
    std::cout << YELLOW_LINE << std::endl;

    return 0;
}