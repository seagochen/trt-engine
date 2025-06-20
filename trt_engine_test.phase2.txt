#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // For std::min, std::max
#include <any>       // For std::any
#include <map>       // For std::map
#include <chrono>    // For std::chrono
#include <numeric>   // For std::accumulate (if calculating average throughput)

#include <opencv2/opencv.hpp>

#include "trtengine/c_apis/c_pose_detection.h"
#include "trtengine/utils/logger.h"

int main()
{
    std::string yolo_engine_path = "/opt/models/yolov8s-pose.engine";
    std::string efficient_engine_path = "/opt/models/efficientnet_b0_feat_logits.engine";

    // 初始化pose检测模型
    if (!init_pose_detection_pipeline(
        yolo_engine_path.c_str(),
        efficient_engine_path.c_str(),
        100, 0.4f, 0.3f))
    {
        deinit_pose_detection_pipeline();
        LOG_ERROR("TrtEngineDemo", "Initialization failed for pose detection pipeline.");
        return -1;
    }
    LOG_INFO("TrtEngineDemo", "Pose detection pipeline initialized successfully.");

    // 加载测试图片
    std::vector<std::string> batch_images_paths = {
        "/opt/images/supermarket/customer1.png",
        "/opt/images/supermarket/customer2.png",
        "/opt/images/supermarket/customer3.png",
        "/opt/images/supermarket/customer4.png",
        "/opt/images/supermarket/customer5.png",
        "/opt/images/supermarket/customer6.png",
        "/opt/images/supermarket/customer7.png",
        "/opt/images/supermarket/customer8.png",

        "/opt/images/supermarket/staff1.png",
        "/opt/images/supermarket/staff2.png",
        "/opt/images/supermarket/staff3.png",
        "/opt/images/supermarket/staff4.png",
        "/opt/images/supermarket/staff5.png",
        "/opt/images/supermarket/staff6.png",
        "/opt/images/supermarket/staff7.png",
        "/opt/images/supermarket/staff8.png",
    };

    // 读取图片
    std::vector<cv::Mat> images_blobs;
    for (const auto& image_path : batch_images_paths)
    {
        cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
        if (img.empty())
        {
            LOG_ERROR("TrtEngineDemo", "Failed to read image: " + image_path);
            return -1;
        }
        images_blobs.push_back(img);
    }
    LOG_DEBUG_V2("TrtEngineDemo", "Loaded " + std::to_string(images_blobs.size()) + " images for pose detection.");

    // 将图片添加到pose检测管道中
    for (int i = 0; i < images_blobs.size(); ++i)
    {
        add_image_to_pose_detection_pipeline(images_blobs[i].data, images_blobs[i].cols, images_blobs[i].rows);
        LOG_DEBUG_V1("TrtEngineDemo",
            "Added image " + batch_images_paths[i] + " to pose detection pipeline, size: " +
            std::to_string(images_blobs[i].cols) + "x" + std::to_string(images_blobs[i].rows));
    }

    // 启动pose检测
    LOG_INFO("TrtEngineDemo", "Starting pose detection on the batch of images.");
    void* out_results = nullptr;
    int out_num_results = 0;
    if (!run_pose_detection_pipeline(&out_results, &out_num_results))
    {
        LOG_ERROR("TrtEngineDemo", "Pose detection pipeline failed to run.");
        deinit_pose_detection_pipeline();
        return -1;
    } 

    // 释放推理结果
    LOG_INFO("TrtEngineDemo", "Releasing inference results.");
    release_inference_result(out_results);
    LOG_INFO("TrtEngineDemo", "Pose detection completed successfully.");

    // 删除模型
    LOG_INFO("TrtEngineDemo", "Starting to deinitialize pose detection pipeline.");
    deinit_pose_detection_pipeline();
    return 0;
}