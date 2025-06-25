#include <opencv2/opencv.hpp>
#include "trtengine/serverlet/models/inference/model_init_helper.hpp"
#include "trtengine/utils/system.h" // Assuming getCurrentRSS is in this header

// Example usage
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // For std::min, std::max
#include <any>       // For std::any
#include <map>       // For std::map
#include <chrono>    // For std::chrono
#include <numeric>   // For std::accumulate (if calculating average throughput)

// Include OpenMP header
#include <omp.h>

// Helper function to draw pose detection results
void draw_pose_results(cv::Mat& image, const std::vector<YoloPose>& pose_detections) {
    for (const auto& pose : pose_detections) {
        // Choose color based on pose.cls
        cv::Scalar box_color;
        // Assuming cls is a float between 0 and 1
        // We can segment based on cls value to choose colors
        if (pose.cls < 0.2f) {
            box_color = cv::Scalar(0, 0, 255); // Red for low confidence
        } else if (pose.cls < 0.5f) {
            box_color = cv::Scalar(0, 165, 255); // Orange for medium-low confidence
        } else if (pose.cls < 0.8f) {
            box_color = cv::Scalar(0, 255, 255); // Yellow for medium-high confidence
        } else {
            box_color = cv::Scalar(0, 255, 0); // Green for high confidence
        }

        // Draw bounding box
        cv::rectangle(image, cv::Rect(pose.lx, pose.ly, pose.rx - pose.lx, pose.ry - pose.ly), box_color, 2);

        // Draw keypoints (keypoint color can usually be fixed, or also changed based on cls)
        // We'll keep keypoints red here for distinction
        for (const auto& pt : pose.pts) {
            if (pt.x >= 0 && pt.y >= 0) { // Ensure keypoint is valid
                cv::circle(image, cv::Point(pt.x, pt.y), 3, cv::Scalar(0, 0, 255), -1);
            }
        }
        // Draw class score (optional)
        std::string label = "Cls: " + std::to_string(pose.cls);
        cv::putText(image, label, cv::Point(pose.lx, pose.ly - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1); // Text color matches box
    }
}

// Function to benchmark YOLO Pose and EfficientNet inference
void benchmark_yolo_pose_efficient(int num_iterations = 1000,
                                   bool calculate_every_step = false, // This flag now only controls calculation, not printing
                                   bool display_results = true) {

    std::string image_path = "/opt/images/supermarket/customer2.png";
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

    auto start_total_time = std::chrono::high_resolution_clock::now();

    cv::Mat resized_image;
    cv::resize(original_image, resized_image, cv::Size(640, 640));

    std::vector<YoloPose> last_pose_detections;

    // Vectors to store durations for each step
    std::vector<long long> yolo_preprocess_times;
    std::vector<long long> yolo_inference_times;
    std::vector<long long> yolo_postprocess_times;
    std::vector<long long> crop_times;
    std::vector<long long> efficient_preprocess_times;
    std::vector<long long> efficient_inference_times;
    std::vector<long long> efficient_postprocess_times;

    for (int iter = 0; iter < num_iterations; ++iter) {
        auto step_start_time = std::chrono::high_resolution_clock::now();

        // YoloPose Preprocess
        pose_model->preprocess(resized_image, 0);
        auto current_step_end_time = std::chrono::high_resolution_clock::now();
        long long duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_step_end_time - step_start_time).count();
        yolo_preprocess_times.push_back(duration_ms);

        // YoloPose Inference
        step_start_time = std::chrono::high_resolution_clock::now();
        pose_model->inference();
        current_step_end_time = std::chrono::high_resolution_clock::now();
        duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_step_end_time - step_start_time).count();
        yolo_inference_times.push_back(duration_ms);

        // YoloPose Postprocess
        step_start_time = std::chrono::high_resolution_clock::now();
        std::any pose_results;
        pose_model->postprocess(0, params1, pose_results);
        current_step_end_time = std::chrono::high_resolution_clock::now();
        duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_step_end_time - step_start_time).count();
        yolo_postprocess_times.push_back(duration_ms);

        std::vector<YoloPose> current_pose_detections;
        try {
            current_pose_detections = std::any_cast<std::vector<YoloPose>>(pose_results);
        } catch (...) {
            // If cast fails, continue to the next iteration
            // Push 0 to all time vectors for this iteration to maintain size consistency
            crop_times.push_back(0);
            efficient_preprocess_times.push_back(0);
            efficient_inference_times.push_back(0);
            efficient_postprocess_times.push_back(0);
            continue;
        }

        // Image Cropping
        step_start_time = std::chrono::high_resolution_clock::now();
        const float scale_factor = 1.2f;
        std::vector<cv::Mat> cropped_images;
        size_t max_efficient_batch = 4; // From params2["maximum_batch"]
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
        current_step_end_time = std::chrono::high_resolution_clock::now();
        duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_step_end_time - step_start_time).count();
        crop_times.push_back(duration_ms);

        // EfficientNet Preprocess
        step_start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < cropped_images.size(); ++i) {
            efficient_model->preprocess(cropped_images[i], i);
        }
        current_step_end_time = std::chrono::high_resolution_clock::now();
        duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_step_end_time - step_start_time).count();
        efficient_preprocess_times.push_back(duration_ms);

        // EfficientNet Inference
        step_start_time = std::chrono::high_resolution_clock::now();
        efficient_model->inference();
        current_step_end_time = std::chrono::high_resolution_clock::now();
        duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_step_end_time - step_start_time).count();
        efficient_inference_times.push_back(duration_ms);

        // EfficientNet Postprocess
        step_start_time = std::chrono::high_resolution_clock::now();
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
        current_step_end_time = std::chrono::high_resolution_clock::now();
        duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_step_end_time - step_start_time).count();
        efficient_postprocess_times.push_back(duration_ms);
        // Removed: if (calculate_every_step) { std::cout << "..." << std::endl; }

        if (iter == num_iterations - 1) {
            last_pose_detections = current_pose_detections;
        }
    }

    auto end_total_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_total_time - start_total_time;

    double avg_total_ms = (total_duration.count() * 1000) / num_iterations;

    std::cout << "\n--- Efficient YOLO Pose Benchmark Results ---" << std::endl;
    std::cout << "  Total iterations: " << num_iterations << std::endl;
    std::cout << "  Total time: " << total_duration.count() << " seconds" << std::endl;
    std::cout << "  Average total time per iteration: " << avg_total_ms << " ms" << std::endl;

    // Calculate and display average times for each step
    auto calculate_average = [](const std::vector<long long>& times) {
        if (times.empty()) return 0.0;
        long long sum = std::accumulate(times.begin(), times.end(), 0LL);
        return static_cast<double>(sum) / times.size();
    };

    std::cout << "\n--- Average Time Per Step (over " << num_iterations << " iterations) ---" << std::endl;
    std::cout << "  YoloPose Preprocess: " << calculate_average(yolo_preprocess_times) << " ms" << std::endl;
    std::cout << "  YoloPose Inference: " << calculate_average(yolo_inference_times) << " ms" << std::endl;
    std::cout << "  YoloPose Postprocess: " << calculate_average(yolo_postprocess_times) << " ms" << std::endl;
    std::cout << "  Image Cropping: " << calculate_average(crop_times) << " ms" << std::endl;
    std::cout << "  EfficientNet Preprocess: " << calculate_average(efficient_preprocess_times) << " ms" << std::endl;
    std::cout << "  EfficientNet Inference: " << calculate_average(efficient_inference_times) << " ms" << std::endl;
    std::cout << "  EfficientNet Postprocess: " << calculate_average(efficient_postprocess_times) << " ms" << std::endl;


    if (display_results) {
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
    // Setting calculate_every_step to false (or even true, it won't print per-iteration anymore)
    // will now ONLY produce the final average step times.
    benchmark_yolo_pose_efficient(100, true, false); // Changed num_iterations back to 1000 for full benchmark
    return 0;
}