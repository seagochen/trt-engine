// functional_test_yolopose.cpp
#include <opencv2/opencv.hpp>
#include "trtengine/servlet/models/inference/model_init_helper.hpp" // 取决于你的模型工厂注册实现
#include "trtengine/c_apis/c_yolopose_detection.h"                  // 新的 YOLOPose C API
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <chrono>
#include <numeric>

// —— 画框与关键点（沿用你原来的绘制函数）——
void draw_pose_results(cv::Mat& image, const std::vector<YoloPose>& pose_detections) {
    for (const auto& pose : pose_detections) {
        cv::Scalar box_color;
        if (pose.cls == 0)      box_color = cv::Scalar(255, 0, 0);
        else if (pose.cls == 1) box_color = cv::Scalar(0, 255, 0);
        else                    box_color = cv::Scalar(0, 0, 255);

        cv::rectangle(image, cv::Rect(pose.lx, pose.ly, pose.rx - pose.lx, pose.ry - pose.ly), box_color, 2);

        for (const auto& pt : pose.pts) {
            if (pt.x >= 0 && pt.y >= 0) {
                cv::circle(image, cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)), 3, cv::Scalar(0, 0, 255), -1);
            }
        }
        std::string label = "Cls: " + std::to_string(pose.cls) + " Score: " + std::to_string(pose.conf);
        cv::putText(image, label, cv::Point(pose.lx, std::max(0, pose.ly - 10)), cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1);
    }
}

// 记录每次 batch 调用耗时
std::vector<long long> total_process_times;

void benchmark_yolopose_api(
    const std::string& yolov8_pose_engine_path,
    int num_iterations = 10,
    bool display_results = true)
{
    // 1) 载入若干测试图（原文件里用的同一批路径）
    std::vector<std::string> image_paths = {
        "/opt/images/supermarket/customer1.png",
        "/opt/images/supermarket/customer2.png",
        "/opt/images/supermarket/customer3.png",
        "/opt/images/supermarket/customer4.png",
        // "/opt/images/supermarket/customer5.png",
        // "/opt/images/supermarket/customer6.png",
        // "/opt/images/supermarket/customer7.png",
        // "/opt/images/supermarket/customer8.png"
    };

    std::vector<cv::Mat> loaded_original_images;
    std::vector<unsigned char*> input_images_data_raw;
    std::vector<int> widths, heights, channels;

    for (const auto& path : image_paths) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << path << std::endl;
            return;
        }
        // 注意：不强制 resize 到 640x640；你的 YOLO 预处理内部会处理到 640x640
        loaded_original_images.push_back(img.clone());
        input_images_data_raw.push_back(loaded_original_images.back().data);
        widths.push_back(loaded_original_images.back().cols);
        heights.push_back(loaded_original_images.back().rows);
        channels.push_back(loaded_original_images.back().channels());
    }
    if (loaded_original_images.empty()) {
        std::cerr << "No images loaded for benchmarking." << std::endl;
        return;
    }

    // 2) 注册并创建 YOLOPose 上下文
    //    （旧代码是 c_register_models/c_create_pose_pipeline；现在只注册 YOLOPose 并创建其 context）
    c_register_yolopose_model();

    int yolo_max_batch    = static_cast<int>(loaded_original_images.size());
    float yolo_cls_thresh = 0.4f;
    float yolo_iou_thresh = 0.5f;

    void* yolo_ctx = c_create_yolopose_context(
        yolov8_pose_engine_path.c_str(),
        yolo_max_batch,
        yolo_cls_thresh,
        yolo_iou_thresh
    );
    if (!yolo_ctx) {
        std::cerr << "Failed to create YoloPose context. Exit." << std::endl;
        return;
    }

    // 用于最后一轮显示
    std::map<int, std::vector<YoloPose>> final_display_detections_map;

    auto start_total_time = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < num_iterations; ++iter) {
        auto start_iter = std::chrono::high_resolution_clock::now();

        // 3) 直接调用 YOLOPose 的批量处理接口
        C_BatchedPoseResults c_results = c_process_batched_images_with_yolopose(
            yolo_ctx,
            input_images_data_raw.data(),
            widths.data(),
            heights.data(),
            channels.data(),
            static_cast<int>(loaded_original_images.size())
        );

        auto end_iter = std::chrono::high_resolution_clock::now();
        total_process_times.push_back(
            std::chrono::duration_cast<std::chrono::milliseconds>(end_iter - start_iter).count()
        );

        // 4) 最后一轮转为 C++ 结构体用于显示
        if (iter == num_iterations - 1) {
            final_display_detections_map.clear();
            for (int i = 0; i < c_results.num_images; ++i) {
                if (c_results.results == nullptr) break;
                std::vector<YoloPose> cpp_poses_for_display;
                for (int j = 0; j < c_results.results[i].num_detections; ++j) {
                    YoloPose p;
                    p.lx = c_results.results[i].detections[j].lx;
                    p.ly = c_results.results[i].detections[j].ly;
                    p.rx = c_results.results[i].detections[j].rx;
                    p.ry = c_results.results[i].detections[j].ry;
                    p.cls = static_cast<int>(c_results.results[i].detections[j].cls);
                    p.conf = c_results.results[i].detections[j].conf;
                    p.pts.clear();
                    for (int k = 0; k < c_results.results[i].detections[j].num_pts; ++k) {
                        YoloPoint kp{};
                        kp.x = static_cast<int>(c_results.results[i].detections[j].pts[k].x);
                        kp.y = static_cast<int>(c_results.results[i].detections[j].pts[k].y);
                        kp.conf = c_results.results[i].detections[j].pts[k].conf;
                        p.pts.push_back(kp);
                    }
                    cpp_poses_for_display.push_back(p);
                }
                final_display_detections_map[c_results.results[i].image_idx] = std::move(cpp_poses_for_display);
            }
        }

        // 5) 释放本轮 C 结构结果
        c_free_batched_pose_results(&c_results);
    }

    auto end_total_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration_s = end_total_time - start_total_time;

    // 6) 销毁 YOLOPose 上下文
    c_destroy_yolopose_context(yolo_ctx);

    // 7) 打印统计
    double avg_total_ms = (total_duration_s.count() * 1000.0) / std::max(1, num_iterations);
    auto calc_avg = [](const std::vector<long long>& v) {
        if (v.empty()) return 0.0;
        long long sum = std::accumulate(v.begin(), v.end(), 0LL);
        return static_cast<double>(sum) / static_cast<double>(v.size());
    };

    std::cout << "\n--- YOLO Pose Benchmark (C API) ---\n";
    std::cout << "  Iterations: " << num_iterations << "\n";
    std::cout << "  Batch size: " << loaded_original_images.size() << " images\n";
    std::cout << "  Total time: " << total_duration_s.count() << " s\n";
    std::cout << "  Avg per batch (wall): " << avg_total_ms << " ms\n";
    std::cout << "  Avg c_process_batched_images_with_yolopose(): " << calc_avg(total_process_times) << " ms\n";

    // 8) 显示最后一轮可视化
    if (display_results && !final_display_detections_map.empty()) {
        for (auto const& [image_idx, detections] : final_display_detections_map) {
            if (image_idx < static_cast<int>(loaded_original_images.size())) {
                cv::Mat display_image = loaded_original_images[image_idx].clone();
                // 🔧 关键：显示前统一缩放到 640×640
                cv::resize(display_image, display_image, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);

                draw_pose_results(display_image, detections);
                std::string window_name = "YOLO Pose - Image " + std::to_string(image_idx);
                cv::imshow(window_name, display_image);
            }
        }
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

int main() {
    // 旧版测试中采用的整体注册与 pipeline 创建见原文件；现在只注册 YOLOPose。:contentReference[oaicite:2]{index=2}
    // 下面这个路径替换为你的 yolov8-pose TensorRT 引擎文件。
    const std::string yolov8_pose_engine = "/opt/models/yolov8n-pose.engine";

    benchmark_yolopose_api(yolov8_pose_engine, /*num_iterations=*/1, /*display_results=*/true);
    return 0;
}
