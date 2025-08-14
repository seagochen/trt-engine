#include <opencv2/opencv.hpp>
#include "trtengine/servlet/models/inference/model_init_helper.hpp"
#include "trtengine/servlet/models/common/yolo_drawer.h"
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
            cv::destroyAllWindows();
        } else {
            std::cout << "No pose detections found in the last iteration to display." << std::endl;
        }
    }
}

void benchmark_yolo_detection()
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
        auto detection_results = std::any_cast<std::vector<Yolo>>(raw_results);

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


/**
 * @brief Generates a random OpenCV Mat image of specified dimensions and type.
 * @param width Image width.
 * @param height Image height.
 * @param channels Image channels (e.g., 3 for BGR).
 * @return A cv::Mat object filled with random pixel values.
 */
cv::Mat generate_random_image(int width, int height, int channels) {
    cv::Mat img(height, width, CV_8UC(channels));
    cv::randu(img, cv::Scalar::all(0), cv::Scalar::all(255)); // Fill with random values 0-255
    return img;
}

/**
 * @brief Tests EfficientNet throughput for various batch sizes.
 * Creates random images and measures processing time.
 */
void benchmark_efficientnet_throughput() {
    const std::string engine_path = "/opt/models/efficientnet_b0_feat_logits.engine";
    const int image_width = 224;
    const int image_height = 224;
    const int image_channels = 3;

    // Test batch sizes
    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32}; // Adjusted for more realistic testing
    const int num_iterations = 100; // Number of times to run inference for each batch size

    for (int current_batch_size : batch_sizes) {
        std::cout << "--- Testing EfficientNet with batch size: " << current_batch_size << " ---" << std::endl;

        std::map<std::string, std::any> params;
        params["maximum_batch"] = current_batch_size;

        std::unique_ptr<InferModelBaseMulti> efficient_model = ModelFactory::createModel(
            "EfficientNet", engine_path, params
        );

        if (!efficient_model) {
            std::cerr << "Failed to create EfficientNet Model for batch size " << current_batch_size << ". Skipping." << std::endl;
            continue;
        }

        // Generate random images for the current batch size
        std::vector<cv::Mat> random_images(current_batch_size);
        for (int i = 0; i < current_batch_size; ++i) {
            random_images[i] = generate_random_image(image_width, image_height, image_channels);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < num_iterations; ++iter) {
            // Preprocess batch
            for (int i = 0; i < current_batch_size; ++i) {
                efficient_model->preprocess(random_images[i], i);
            }

            // Inference
            efficient_model->inference();

            // Postprocess (only need to call once per batch, results for each item in batch)
            // We don't need to actually retrieve results for throughput testing, just ensure the call works.
            for (int i = 0; i < current_batch_size; ++i) {
                std::any results_out;
                efficient_model->postprocess(i, params, results_out);
                // Optionally, cast and check results to ensure correctness, but not strictly needed for throughput
                // try {
                //      auto results_vec = std::any_cast<std::vector<float>>(results_out);
                // } catch(...) {}
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;

        double total_images_processed = static_cast<double>(num_iterations) * current_batch_size;
        double throughput_ips = total_images_processed / duration.count();

        std::cout << "   Total images processed: " << total_images_processed << std::endl;
        std::cout << "   Total time taken: " << duration.count() << " seconds" << std::endl;
        std::cout << "   Throughput: " << throughput_ips << " images/second" << std::endl;
        std::cout << std::endl;
    }
}


/**
 * @brief Tests YoloPose throughput for various batch sizes.
 * Creates random images and measures processing time.
 */
void benchmark_yolo_pose_throughput() {
    const std::string engine_path = "/opt/models/yolov8s-pose.engine";
    const int image_width = 640;
    const int image_height = 640;
    const int image_channels = 3;

    // Test batch sizes
    std::vector<int> batch_sizes = {1, 2, 4, 8};
    const int num_iterations = 100; // Number of times to run inference for each batch size

    for (int current_batch_size : batch_sizes) {
        std::cout << "--- Testing YoloPose with batch size: " << current_batch_size << " ---" << std::endl;

        std::map<std::string, std::any> params;
        params["maximum_batch"] = current_batch_size;
        // ADDED: Post-processing parameters for YoloPose
        params["cls"] = 0.4f; // Confidence threshold
        params["iou"] = 0.5f; // IoU threshold
        // 关键修复：添加YoloPose模型后处理所需的参数
        params["maximum_items"] = 100;  // 从test_yolo_pose中获取的参数
        params["infer_features"] = 56;  // 从test_yolo_pose中获取的参数
        params["infer_samples"] = 8400; // 从test_yolo_pose中获取的参数

        std::unique_ptr<InferModelBaseMulti> yolo_pose_model = ModelFactory::createModel(
            "YoloV8_Pose", engine_path, params
        );

        if (!yolo_pose_model) {
            std::cerr << "Failed to create YoloPose model for batch size " << current_batch_size << ". Skipping." << std::endl;
            continue;
        }

        // Generate random images for the current batch size
        std::vector<cv::Mat> random_images(current_batch_size);
        for (int i = 0; i < current_batch_size; ++i) {
            random_images[i] = generate_random_image(image_width, image_height, image_channels);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < num_iterations; ++iter) {
            // Preprocess batch
            for (int i = 0; i < current_batch_size; ++i) {
                yolo_pose_model->preprocess(random_images[i], i);
            }

            // Inference
            yolo_pose_model->inference();

            // Postprocess (results for each item in batch)
            for (int i = 0; i < current_batch_size; ++i) {
                std::any results_out;
                // Pass the `params` map which now includes "cls" and "iou" and other necessary parameters
                yolo_pose_model->postprocess(i, params, results_out);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;

        double total_images_processed = static_cast<double>(num_iterations) * current_batch_size;
        double throughput_ips = total_images_processed / duration.count();

        std::cout << "   Total images processed: " << total_images_processed << std::endl;
        std::cout << "   Total time taken: " << duration.count() << " seconds" << std::endl;
        std::cout << "   Throughput: " << throughput_ips << " images/second" << std::endl;
        std::cout << std::endl;
    }
}


/**
 * @brief Performs a memory leak stress test for a given model.
 * Executes inference requests many times with batch size 1 and prints memory usage.
 * @param model_name Name of the model (for ModelFactory).
 * @param engine_path Path to the TensorRT engine file.
 * @param infer_params Parameters for model creation and postprocessing.
 * @param image_path Path to a sample image for preprocessing.
 * @param is_yolo_model True if it's a YOLO model (to adjust image dimensions).
 */
void benchmark_memory_leak_stress(const std::string& model_name,
                             const std::string& engine_path,
                             const std::map<std::string, std::any>& infer_params,
                             const std::string& image_path,
                             bool is_yolo_model) {
    const int num_iterations = 10000; // 1万次请求

    std::cout << "--- Starting memory leak stress test for: " << model_name << " (" << num_iterations << " iterations) ---" << std::endl;

    // Use a fixed batch size of 1 for this test as requested
    std::map<std::string, std::any> current_params = infer_params;
    current_params["maximum_batch"] = 1; // Force batch size to 1

    std::unique_ptr<InferModelBaseMulti> model = ModelFactory::createModel(
        model_name, engine_path, current_params
    );

    if (!model) {
        std::cerr << "Failed to create model " << model_name << ". Skipping stress test." << std::endl;
        return;
    }

    cv::Mat original_image = cv::imread(image_path);
    if (original_image.empty()) {
        std::cerr << "Failed to load image for stress test: " << image_path << ". Skipping." << std::endl;
        return;
    }

    // Resize image once if it's a YOLO model's input dimension
    cv::Mat processed_image;
    if (is_yolo_model) {
        cv::resize(original_image, processed_image, cv::Size(640, 640));
    } else {
        // Assuming EfficientNet input is 224x224
        cv::resize(original_image, processed_image, cv::Size(224, 224));
    }


    long initial_rss = getCurrentRSS();
    std::cout << "Initial RSS: " << initial_rss << " KB" << std::endl;


    for (int i = 0; i < num_iterations; ++i) {
        model->preprocess(processed_image, 0); // Always batch 0 for single item
        model->inference();
        std::any results_out;
        model->postprocess(0, current_params, results_out); // Pass current_params (contains cls/iou if needed)

        // Optionally, check result validity without processing it much
        if (model_name == "YoloV8_Pose" || model_name == "YoloV8_Detection") {
            auto results_vec = std::any_cast<std::vector<YoloPose>>(results_out); // or std::vector<Yolo>
        } else if (model_name == "EfficientNet") {
            auto results_vec = std::any_cast<std::vector<float>>(results_out);
        }

        // Print memory usage periodically
        if ((i + 1) % (num_iterations / 10) == 0) { // Print 10 times during the test
            const long current_rss = getCurrentRSS();
            std::cout << "   Iteration " << (i + 1) << "/" << num_iterations << ", Current RSS: " << current_rss << " KB, Diff: " << (current_rss - initial_rss) << " KB" << std::endl;
        }
    }

    const long final_rss = getCurrentRSS();
    std::cout << "Final RSS: " << final_rss << " KB" << std::endl;
    std::cout << "Memory usage difference: " << (final_rss - initial_rss) << " KB" << std::endl;
    std::cout << "--- Memory leak stress test for " << model_name << " finished. ---" << std::endl << std::endl;
}

int main() {
    registerModels();

    // 黄色输出前缀和后缀
    const std::string YELLOW = "\033[33m";
    const std::string RESET = "\033[0m";
    const std::string YELLOW_LINE = YELLOW + "-------------------" + RESET;

    // 测试 YOLOv8 检测模型
    std::cout << YELLOW << "Detecting objects using YOLOv8..." << RESET << std::endl;
    benchmark_yolo_detection();
    std::cout << YELLOW_LINE << std::endl;

    // 测试 YOLOv8 + EfficientNet 模型
    std::cout << YELLOW << "Testing YOLOv8 + EfficientNet..." << RESET << std::endl;
    benchmark_yolo_pose_efficient(100, true, true); // Changed num_iterations back to 1000 for full benchmark
    std::cout << YELLOW_LINE << std::endl;

    // EfficientNet 吞吐量测试
    std::cout << YELLOW << "Starting EfficientNet Throughput Test..." << RESET << std::endl;
    benchmark_efficientnet_throughput();
    std::cout << YELLOW_LINE << std::endl;

    // 2. YOLOv8 Pose 吞吐量测试
    std::cout << YELLOW << "Starting YOLOv8 Pose Throughput Test..." << RESET << std::endl;
    benchmark_yolo_pose_throughput();
    std::cout << YELLOW_LINE << std::endl;

    // 2. 内存溢出/泄露测试 (1万次请求)
    std::cout << YELLOW << "Starting Memory Leak Stress Tests (10,000 iterations each)..." << RESET << std::endl;

    // YOLOv8 Pose
    std::map<std::string, std::any> yolo_pose_params;
    yolo_pose_params["maximum_batch"] = 1; // Fixed for this test
    yolo_pose_params["maximum_items"] = 100;
    yolo_pose_params["infer_features"] = 56;
    yolo_pose_params["infer_samples"] = 8400;
    yolo_pose_params["cls"] = 0.4f;
    yolo_pose_params["iou"] = 0.5f;
    benchmark_memory_leak_stress("YoloV8_Pose",
        "/opt/models/yolov8s-pose.engine",
        yolo_pose_params, "/opt/images/human_and_pets.png", true);

    // YOLOv8 Detection
    std::map<std::string, std::any> yolo_detection_params;
    yolo_detection_params["maximum_batch"] = 1; // Fixed for this test
    yolo_detection_params["maximum_items"] = 100;
    yolo_detection_params["infer_features"] = 84;
    yolo_detection_params["infer_samples"] = 8400;
    yolo_detection_params["cls"] = 0.4f;
    yolo_detection_params["iou"] = 0.5f;
    benchmark_memory_leak_stress("YoloV8_Detection",
        "/opt/models/yolov8s.engine",
        yolo_detection_params, "/opt/images/india_road.png", true);

    // EfficientNet
    std::map<std::string, std::any> efficientnet_params_leak_test;
    efficientnet_params_leak_test["maximum_batch"] = 1; // Fixed for this test
    benchmark_memory_leak_stress("EfficientNet",
        "/opt/models/efficientnet_b0_feat_logits.engine",
        efficientnet_params_leak_test, "/opt/images/apples.png", false);

    std::cout << YELLOW_LINE << std::endl;

    return 0;
}