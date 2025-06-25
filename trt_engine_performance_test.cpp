#include <opencv2/opencv.hpp>
#include "trtengine/serverlet/models/inference/model_init_helper.hpp"
#include "trtengine/utils/system.h" // 假设 getCurrentRSS 在此头文件中

// 示例用法
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // For std::min, std::max
#include <any>       // For std::any
#include <map>       // For std::map
#include <chrono>    // For std::chrono
#include <numeric>   // For std::accumulate (if calculating average throughput)

// 包含 OpenMP 头文件
#include <omp.h>

// 绘制姿态检测结果的辅助函数
void draw_pose_results(cv::Mat& image, const std::vector<YoloPose>& pose_detections) {
    for (const auto& pose : pose_detections) {
        // 根据 pose.cls 选择颜色
        cv::Scalar box_color;
        // 假设 cls 是 0 到 1 之间的浮点数
        // 我们可以根据 cls 值分段来选择颜色
        if (pose.cls < 0.2f) {
            box_color = cv::Scalar(0, 0, 255); // 红色 for low confidence
        } else if (pose.cls < 0.5f) {
            box_color = cv::Scalar(0, 165, 255); // 橙色 for medium-low confidence
        } else if (pose.cls < 0.8f) {
            box_color = cv::Scalar(0, 255, 255); // 黄色 for medium-high confidence
        } else {
            box_color = cv::Scalar(0, 255, 0); // 绿色 for high confidence
        }

        // 绘制边界框
        cv::rectangle(image, cv::Rect(pose.lx, pose.ly, pose.rx - pose.lx, pose.ry - pose.ly), box_color, 2);

        // 绘制关节点 (通常关节点颜色可以固定，或者也可以根据 cls 改变)
        // 这里我们保持关节点为红色，以便区分
        for (const auto& pt : pose.pts) {
            if (pt.x >= 0 && pt.y >= 0) { // 确保关节点有效
                cv::circle(image, cv::Point(pt.x, pt.y), 3, cv::Scalar(0, 0, 255), -1);
            }
        }
        // 绘制类别分数 (可选)
        std::string label = "Cls: " + std::to_string(pose.cls);
        cv::putText(image, label, cv::Point(pose.lx, pose.ly - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1); // 文本颜色与框同色
    }
}

// 执行推理1000次并统计平均耗时
void benchmark_yolo_pose_efficient(int num_iterations = 1000,
    bool calculate_every_step = false,
    bool display_results = true) {

    std::string image_path = "/opt/images/supermarket/customer8.png";
    cv::Mat original_image = cv::imread(image_path);
    if (original_image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return;
    }

    std::map<std::string, std::any> params1{
        {"maximum_batch", 1},
        {"maximum_items", 100},
        {"infer_features", 56},
        {"infer_samples", 8400},
        {"cls", 0.4f},
        {"iou", 0.5f}
    };

    std::unique_ptr<InferModelBaseMulti> pose_model = ModelFactory::createModel("YoloV8_Pose",
        "/opt/models/yolov8n-pose.engine", params1);
    if (!pose_model) {
        std::cerr << "Failed to create pose model in benchmark_yolo_pose_efficient" << std::endl;
        return;
    }

    std::map<std::string, std::any> params2{
        {"maximum_batch", 32}
    };

    std::unique_ptr<InferModelBaseMulti> efficient_model = ModelFactory::createModel("EfficientNet",
        "/opt/models/efficientnet_b0_feat_logits.engine", params2);
    if (!efficient_model) {
        std::cerr << "Failed to create efficient model in benchmark_yolo_pose_efficient" << std::endl;
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    cv::Mat resized_image;
    cv::resize(original_image, resized_image, cv::Size(640, 640));

    std::vector<YoloPose> last_pose_detections;

    for (int iter = 0; iter < num_iterations; ++iter) {
        auto step_time = std::chrono::high_resolution_clock::now();

        pose_model->preprocess(resized_image, 0);
        if (calculate_every_step) {
            auto step_duration = std::chrono::high_resolution_clock::now() - step_time;
            std::cout << "YoloPose model preprocess time for iteration " << iter << ": "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(step_duration).count()
                      << " ms" << std::endl;
        }

        step_time = std::chrono::high_resolution_clock::now();
        pose_model->inference();
        if (calculate_every_step) {
            auto step_duration = std::chrono::high_resolution_clock::now() - step_time;
            std::cout << "YoloPose model inference time for iteration " << iter << ": "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(step_duration).count()
                      << " ms" << std::endl;
        }

        step_time = std::chrono::high_resolution_clock::now();
        std::any pose_results;
        pose_model->postprocess(0, params1, pose_results);
        if (calculate_every_step) {
            auto step_duration = std::chrono::high_resolution_clock::now() - step_time;
            std::cout << "YoloPose model postprocess time for iteration " << iter << ": "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(step_duration).count()
                      << " ms" << std::endl;
        }

        std::vector<YoloPose> current_pose_detections;
        try {
            current_pose_detections = std::any_cast<std::vector<YoloPose>>(pose_results);
        } catch (...) {
            continue;
        }

        step_time = std::chrono::high_resolution_clock::now();
        const float scale_factor = 1.2f;
        std::vector<cv::Mat> cropped_images;
        size_t max_efficient_batch = 4;
        for (size_t i = 0; i < current_pose_detections.size() && cropped_images.size() < max_efficient_batch; ++i) {
            const auto& pose = current_pose_detections[i];
            if (pose.pts.empty()) continue;

            int min_x = std::min(pose.lx, pose.rx);
            int min_y = std::min(pose.ly, pose.ry);
            int max_x = std::max(pose.lx, pose.rx);
            int max_y = std::max(pose.ly, pose.ry);
            int width = max_x - min_x;
            int height = max_y - min_y;

            int crop_x = std::max(0, static_cast<int>(min_x - width * (scale_factor - 1) / 2));
            int crop_y = std::max(0, static_cast<int>(min_y - height * (scale_factor - 1) / 2));
            int crop_width = std::min(resized_image.cols - crop_x, static_cast<int>(width * scale_factor));
            int crop_height = std::min(resized_image.rows - crop_y, static_cast<int>(height * scale_factor));

            if (crop_width > 0 && crop_height > 0)
                cropped_images.emplace_back(resized_image(cv::Rect(crop_x, crop_y, crop_width, crop_height)));
        }
        if (calculate_every_step) {
            auto step_duration = std::chrono::high_resolution_clock::now() - step_time;
            std::cout << "Crop time for iteration " << iter << ": "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(step_duration).count()
                      << " ms" << std::endl;
        }

        step_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < cropped_images.size(); ++i) {
            efficient_model->preprocess(cropped_images[i], i);
        }
        if (calculate_every_step) {
            auto step_duration = std::chrono::high_resolution_clock::now() - step_time;
            std::cout << "Efficient model preprocess time for iteration " << iter << ": "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(step_duration).count()
                      << " ms" << std::endl;
        }

        step_time = std::chrono::high_resolution_clock::now();
        efficient_model->inference();
        if (calculate_every_step) {
            auto step_duration = std::chrono::high_resolution_clock::now() - step_time;
            std::cout << "Efficient model inference time for iteration " << iter << ": "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(step_duration).count()
                      << " ms" << std::endl;
        }

        step_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < cropped_images.size(); ++i) {
            std::any results;
            efficient_model->postprocess(i, params2, results);
            try {
                auto cls_result = std::any_cast<std::vector<float>>(results);
                if (!cls_result.empty() && i < current_pose_detections.size())
                    current_pose_detections[i].cls = static_cast<float>(cls_result[0]);
            } catch (...) {
                continue;
            }
        }
        if (calculate_every_step) {
            auto step_duration = std::chrono::high_resolution_clock::now() - step_time;
            std::cout << "Efficient model postprocess time for iteration " << iter << ": "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(step_duration).count()
                      << " ms" << std::endl;
        }

        if (iter == num_iterations - 1) {
            last_pose_detections = current_pose_detections;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    double avg_ms = (duration.count() * 1000) / num_iterations;

    std::cout << "\n--- Efficient YOLO Pose Benchmark Results ---" << std::endl;
    std::cout << "  Total iterations: " << num_iterations << std::endl;
    std::cout << "  Total time: " << duration.count() << " seconds" << std::endl;
    std::cout << "  Average time per iteration: " << avg_ms << " ms" << std::endl;

    if (display_results)
    {
        if (!last_pose_detections.empty()) {
            cv::Mat display_image = resized_image.clone();
            draw_pose_results(display_image, last_pose_detections);
            cv::imshow("YOLO Pose Detection Results", display_image);
            cv::waitKey(0);
        } else {
            std::cout << "No pose detections found in the last iteration to display." << std::endl;
        }
    }
}

int main() {
    registerModels();
    benchmark_yolo_pose_efficient(1000, false, true);
    return 0;
}