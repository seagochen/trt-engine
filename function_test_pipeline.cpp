//
// Created by user on 6/25/25.
//

#include <opencv2/opencv.hpp>
#include "trtengine/serverlet/models/inference/model_init_helper.hpp"
#include "trtengine/c_apis/c_pose_pipeline.h"

// Standard C++ headers (still needed for utilities, but threading primitives are now encapsulated)
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // For std::min, std::max
#include <map>       // For std::map
#include <chrono>    // For std::chrono
#include <numeric>   // For std::accumulate (if calculating average throughput)

// Helper function to draw pose detection results (accepts C++ YoloPose struct)
// This function is still useful for visualization within this benchmark client
void draw_pose_results(cv::Mat& image, const std::vector<YoloPose>& pose_detections) {
    for (const auto& pose : pose_detections) {
        // Choose color based on pose.cls
        cv::Scalar box_color;
        if (pose.cls == 0)
        {
            box_color = cv::Scalar(255, 0, 0); // Blue for class 0
        } else if (pose.cls == 1)
        {
            box_color = cv::Scalar(0, 255, 0); // Green for class 1
        } else
        {
            box_color = cv::Scalar(0, 0, 255); // Red for other classes
        }

        // Draw bounding box
        cv::rectangle(image, cv::Rect(pose.lx, pose.ly, pose.rx - pose.lx, pose.ry - pose.ly), box_color, 2);

        // Draw keypoints (keypoint color can usually be fixed, or also changed based on cls)
        // We'll keep keypoints red here for distinction
        for (const auto& pt : pose.pts) {
            if (pt.x >= 0 && pt.y >= 0) { // Ensure keypoint is valid
                cv::circle(image, cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)), 3, cv::Scalar(0, 0, 255), -1);
            }
        }
        // Draw class score (optional)
        std::string label = "Cls: " + std::to_string(pose.cls) + " Score: " + std::to_string(pose.conf);
        cv::putText(image, label, cv::Point(pose.lx, pose.ly - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1); // Text color matches box
    }
}

// Global vector to store durations for each C API call to process_image_batch
std::vector<long long> total_process_times;

// Function to benchmark YOLO Pose and EfficientNet inference using the new C API
void benchmark_yolo_pose_efficient_api(int num_iterations = 100, bool display_results = true) {
    // Load multiple images for batching
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

    std::vector<cv::Mat> loaded_original_images; // Keep original cv::Mat for data and cloning for display
    std::vector<unsigned char*> input_images_data_raw; // Pointers to raw image data
    std::vector<int> widths;
    std::vector<int> heights;
    std::vector<int> channels;

    for (const std::string& path : image_paths) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << path << std::endl;
            // If an image fails to load, the benchmark setup is incomplete.
            // Consider more robust error handling or skipping the image if acceptable.
            return;
        }
        // Resize the image to a fixed size for YOLO Pose input
        cv::resize(img, img, cv::Size(640, 640)); // Resize to 640x640 for YOLO Pose input

        // Store the original image for display purposes
        loaded_original_images.push_back(img.clone()); // Clone to keep the original image intact

        // Prepare raw image data and dimensions
        input_images_data_raw.push_back(loaded_original_images.back().data);
        widths.push_back(loaded_original_images.back().cols);
        heights.push_back(loaded_original_images.back().rows);
        channels.push_back(loaded_original_images.back().channels());
    }

    if (loaded_original_images.empty()) {
        std::cerr << "No images loaded for benchmarking." << std::endl;
        return;
    }

    int yolo_max_batch = static_cast<int>(loaded_original_images.size()); // YoloPose batch size based on loaded images
    int efficient_max_batch = 32; // EfficientNet can handle up to 32 crops in a batch
    float yolo_cls_thresh = 0.4f;
    float yolo_iou_thresh = 0.5f;

    // Create the context using the C API
    YoloEfficientContext* context = c_create_pose_pipeline(
        "/opt/models/yolov8n-pose.engine",
        "/opt/models/efficientnet_b0_feat_logits.engine",
        yolo_max_batch,
        efficient_max_batch,
        yolo_cls_thresh,
        yolo_iou_thresh
    );

    if (!context) {
        std::cerr << "Failed to create YoloEfficientContext. Exiting benchmark." << std::endl;
        return;
    }

    // To store the very last detection results for display (using C++ YoloPose struct for convenience)
    std::map<int, std::vector<YoloPose>> final_display_detections_map;

    auto start_total_pipeline_time = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < num_iterations; ++iter) {
        auto start_iteration_time = std::chrono::high_resolution_clock::now();

        // Call the C API function to process the image batch
        C_BatchedPoseResults c_results = c_process_batched_images(
            context, // Corrected: pass address of context pointer
            input_images_data_raw.data(),
            widths.data(),
            heights.data(),
            channels.data(),
            static_cast<int>(loaded_original_images.size()),
            1.2f // Crop scale factor for EfficientNet
        );

        auto end_iteration_time = std::chrono::high_resolution_clock::now();
        total_process_times.push_back(
            std::chrono::duration_cast<std::chrono::milliseconds>(end_iteration_time - start_iteration_time).count()
        );

        // Store the last results for display
        if (iter == num_iterations - 1) {
            final_display_detections_map.clear();
            for (int i = 0; i < c_results.num_images; ++i) {
                // Ensure image_idx is valid for loaded_original_images
                if (c_results.results[i].image_idx < loaded_original_images.size()) {
                    std::vector<YoloPose> cpp_poses_for_display;
                    for (int j = 0; j < c_results.results[i].num_detections; ++j) {

                        // 将 C_YoloPose 转换为 C++ YoloPose 结构体
                        YoloPose p;
                        p.lx = c_results.results[i].detections[j].lx;
                        p.ly = c_results.results[i].detections[j].ly;
                        p.rx = c_results.results[i].detections[j].rx;
                        p.ry = c_results.results[i].detections[j].ry;
                        p.cls = static_cast<int>(c_results.results[i].detections[j].cls);
                        p.conf = c_results.results[i].detections[j].conf;

                        // 解析关键点数据
                        p.pts.clear();
                        for(int k=0; k < c_results.results[i].detections[j].num_pts; ++k) {
                            YoloPoint kp{};
                            kp.x = static_cast<int>(c_results.results[i].detections[j].pts[k].x);
                            kp.y = static_cast<int>(c_results.results[i].detections[j].pts[k].y);
                            kp.conf = c_results.results[i].detections[j].pts[k].conf;
                            p.pts.push_back(kp);
                        }
                        cpp_poses_for_display.push_back(p);


                        // ———————— 这里是新打印逻辑 ————————
                        // 只打印第 0 个检测，并且 feats 指针非空
                        // auto* det = &c_results.results[i].detections[j];
                        //
                        // if (j == 0 && det->feats) {
                        //     std::cout << std::fixed << std::setprecision(2);
                        //     std::cout << "Image " << c_results.results[i].image_idx
                        //               << ", Detection " << j
                        //               << ", Features: ";
                        //     for (int f = 0; f < 256; ++f) {
                        //         std::cout << det->feats[f] << " ";
                        //     }
                        //     std::cout << std::endl;
                        // }
                        // ————————————————————————————————

                    }
                    final_display_detections_map[c_results.results[i].image_idx] = cpp_poses_for_display;
                }
            }
        }

        // Free the results returned by the C API for each iteration to prevent memory leaks
        c_free_batched_pose_results(&c_results);
    }

    auto end_total_pipeline_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration_s = end_total_pipeline_time - start_total_pipeline_time;

    // Destroy the context to release all associated resources
    c_destroy_pose_pipeline(context); // Corrected: pass address of context pointer

    // Calculate overall average time based on the number of iterations
    double avg_total_ms = (total_duration_s.count() * 1000) / num_iterations;

    std::cout << "\n--- Threaded Efficient YOLO Pose Benchmark Results (via C API) ---" << std::endl;
    std::cout << "  Total batch iterations: " << num_iterations << std::endl;
    std::cout << "  Input batch size: " << loaded_original_images.size() << " images" << std::endl;
    std::cout << "  Total pipeline time: " << total_duration_s.count() << " seconds" << std::endl;
    std::cout << "  Average total pipeline time per batch iteration: " << avg_total_ms << " ms" << std::endl;

    // Calculate and display average times for each C API call
    auto calculate_average = [](const std::vector<long long>& times) {
        if (times.empty()) return 0.0;
        const long long sum = std::accumulate(times.begin(), times.end(), 0LL);
        const auto times_size = static_cast<double>(times.size());
        return static_cast<double>(sum) / times_size;
    };

    std::cout << "\n--- Average Time Per process_image_batch() Call ---" << std::endl;
    std::cout << "  Process Image Batch (Avg): " << calculate_average(total_process_times) << " ms" << std::endl;


    if (display_results) {
        if (!final_display_detections_map.empty()) {
            for (auto const& [image_idx, detections] : final_display_detections_map) {
                if (image_idx < loaded_original_images.size()) {
                    cv::Mat display_image = loaded_original_images[image_idx].clone(); // Clone for drawing
                    draw_pose_results(display_image, detections); // Use the local draw_pose_results
                    std::string window_name = "Threaded YOLO Pose Detection Results - Image " + std::to_string(image_idx + 1);
                    cv::imshow(window_name, display_image);
                }
            }
            cv::waitKey(0);
            cv::destroyAllWindows();
        } else {
            std::cout << "No pose detections found in the last iteration to display across the batch." << std::endl;
        }
    }
}

int main() {
    // Register models with the C API
    c_register_models();

    // Call the new C API benchmark function
    benchmark_yolo_pose_efficient_api(1, true); // Reduced iterations for quicker testing (e.g., for 8 images in 10 iterations)
    return 0;
}
