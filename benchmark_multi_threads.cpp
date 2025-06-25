//
// Created by user on 6/25/25.
//

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
#include <thread>          // For std::thread
#include <queue>           // For std::queue
#include <mutex>           // For std::mutex
#include <condition_variable> // For std::condition_variable
#include <atomic>          // For std::atomic_bool

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

// Data structure to pass between threads
struct PoseResultWithID {
    int iteration_id; // Still useful for tracking overall pipeline iterations
    // Map from original image index in the batch to its pose detections
    std::map<int, std::vector<YoloPose>> batched_detections;
    bool is_final_signal; // To signal termination (poison pill)

    // Constructor for batched results
    PoseResultWithID(int id, const std::map<int, std::vector<YoloPose>>& batched_det)
        : iteration_id(id), batched_detections(batched_det), is_final_signal(false) {}

    // Constructor for final signal (poison pill)
    PoseResultWithID() : iteration_id(-1), is_final_signal(true) {}
};

// Thread-safe queue for YoloPose results
std::queue<PoseResultWithID> pose_results_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;
std::atomic<bool> yolo_producer_done(false); // Flag to indicate YoloPose thread has finished producing

// Global vectors to store durations from each thread
// Note: In a larger application, these might be part of a shared context
// or passed as references to avoid global state. For a benchmark, it's acceptable.
std::vector<long long> yolo_preprocess_times_thread;
std::vector<long long> yolo_inference_times_thread;
std::vector<long long> yolo_postprocess_times_thread;

std::vector<long long> crop_times_thread;
std::vector<long long> efficient_preprocess_times_thread;
std::vector<long long> efficient_inference_times_thread;
std::vector<long long> efficient_postprocess_times_thread;


// YoloPose thread function (Producer)
// Now accepts a vector of images for batch processing
void yolo_pose_thread_func(int num_iterations, const std::vector<cv::Mat>& batched_input_images,
                           const std::map<std::string, std::any>& params1) {

    // Initialize model within the thread
    std::unique_ptr<InferModelBaseMulti> pose_model = ModelFactory::createModel("YoloV8_Pose",
        "/opt/models/yolov8n-pose.engine", params1);
    if (!pose_model) {
        std::cerr << "YoloPose Thread: Failed to create pose model. Exiting thread." << std::endl;
        // Ensure consumer is notified to stop if producer fails early
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            pose_results_queue.push(PoseResultWithID()); // Send final signal
            yolo_producer_done = true;
        }
        queue_cv.notify_one();
        return;
    }

    // Get the maximum batch size for YoloPose from params1
    size_t yolo_max_batch = 1; // Default
    if (params1.count("maximum_batch")) {
        try {
            yolo_max_batch = std::any_cast<int>(params1.at("maximum_batch"));
        } catch (const std::bad_any_cast& e) {
            std::cerr << "Warning: 'maximum_batch' in params1 is not an int. Using default of 1." << std::endl;
        }
    }
    // Ensure yolo_max_batch doesn't exceed the number of available input images
    yolo_max_batch = std::min(yolo_max_batch, batched_input_images.size());
    if (yolo_max_batch == 0) {
        std::cerr << "YoloPose Thread: No images to process or invalid batch size. Exiting." << std::endl;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            pose_results_queue.push(PoseResultWithID()); // Send final signal
            yolo_producer_done = true;
        }
        queue_cv.notify_one();
        return;
    }


    for (int iter = 0; iter < num_iterations; ++iter) {
        // YoloPose Preprocess for each image in the batch
        auto step_start_time_preprocess = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < yolo_max_batch; ++i) {
            // Call preprocess for each image with its batch index
            pose_model->preprocess(batched_input_images[i], i);
        }
        yolo_preprocess_times_thread.push_back(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - step_start_time_preprocess).count()
        );

        // YoloPose Inference (after all images for the batch are preprocessed)
        auto step_start_time_inference = std::chrono::high_resolution_clock::now();
        pose_model->inference();
        yolo_inference_times_thread.push_back(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - step_start_time_inference).count()
        );

        // YoloPose Postprocess for the batch - iterate for each item in the batch
        auto step_start_time_postprocess = std::chrono::high_resolution_clock::now();
        std::map<int, std::vector<YoloPose>> current_batched_pose_detections;

        for (size_t i = 0; i < yolo_max_batch; ++i) {
            std::any single_image_pose_results_any;
            // Call postprocess for each image index to get its specific results
            pose_model->postprocess(i, params1, single_image_pose_results_any);

            try {
                // Expecting std::vector<YoloPose> for each individual image
                std::vector<YoloPose> single_image_detections = std::any_cast<std::vector<YoloPose>>(single_image_pose_results_any);
                current_batched_pose_detections[i] = single_image_detections;
            } catch (const std::bad_any_cast& e) {
                std::cerr << "YoloPose Thread: Error casting single image postprocess results for batch index " << i << ": " << e.what() << std::endl;
                // Store an empty vector for this image if casting fails
                current_batched_pose_detections[i] = {}; // Initialize with empty vector
            } catch (...) {
                std::cerr << "YoloPose Thread: Unknown error during single image postprocessing or cast for batch index " << i << "." << std::endl;
                current_batched_pose_detections[i] = {}; // Initialize with empty vector
            }
        }
        yolo_postprocess_times_thread.push_back(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - step_start_time_postprocess).count()
        );

        // Push results to the queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            pose_results_queue.push(PoseResultWithID(iter, current_batched_pose_detections));
        }
        queue_cv.notify_one(); // Notify consumer
    }

    // Send final signal (poison pill)
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        pose_results_queue.push(PoseResultWithID()); // Send final signal
        yolo_producer_done = true; // Set atomic flag
    }
    queue_cv.notify_one(); // Notify consumer one last time
    std::cout << "YoloPose Thread: Finished producing " << num_iterations << " iterations of batches." << std::endl;
}


// EfficientNet thread function (Consumer)
void efficient_net_thread_func(const std::vector<cv::Mat>& original_batched_images,
                               const std::map<std::string, std::any>& params2,
                               std::map<int, std::vector<YoloPose>>& final_display_detections_map_ref) { // Reference to store final detections

    // Initialize model within the thread
    std::unique_ptr<InferModelBaseMulti> efficient_model = ModelFactory::createModel("EfficientNet",
        "/opt/models/efficientnet_b0_feat_logits.engine", params2);
    if (!efficient_model) {
        std::cerr << "EfficientNet Thread: Failed to create efficient model. Exiting thread." << std::endl;
        return;
    }

    size_t efficient_max_batch = 1; // Default
    if (params2.count("maximum_batch")) {
        try {
            efficient_max_batch = std::any_cast<int>(params2.at("maximum_batch"));
        } catch (const std::bad_any_cast& e) {
            std::cerr << "Warning: 'maximum_batch' in params2 is not an int. Using default of 1." << std::endl;
        }
    }

    int processed_count = 0;
    while (true) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_cv.wait(lock, [&]{ return !pose_results_queue.empty() || yolo_producer_done; });

        if (pose_results_queue.empty() && yolo_producer_done) {
            std::cout << "EfficientNet Thread: All items processed. Exiting." << std::endl;
            break;
        }

        PoseResultWithID frame_data = pose_results_queue.front();
        pose_results_queue.pop();
        lock.unlock();

        if (frame_data.is_final_signal) {
            std::cout << "EfficientNet Thread: Received final signal. Exiting." << std::endl;
            break;
        }

        std::map<int, std::vector<YoloPose>>& current_batched_pose_detections = frame_data.batched_detections;

        // Flatten all detections from the current YoloPose batch
        // We also need to keep track of the original image index and the pose's index within that image
        struct FlattenedPose {
            int original_image_idx;
            YoloPose pose;
            size_t original_pose_idx;
        };
        std::vector<FlattenedPose> all_flattened_poses;

        for (auto const& [image_idx, poses_in_image] : current_batched_pose_detections) {
            for (size_t i = 0; i < poses_in_image.size(); ++i) {
                all_flattened_poses.push_back({image_idx, poses_in_image[i], i});
            }
        }

        // --- Cropping and EfficientNet processing for the aggregated batch ---
        const float scale_factor = 1.2f;

        // Process EfficientNet in batches up to efficient_max_batch
        std::vector<std::vector<float>> efficient_net_all_results; // Store all results
        std::vector<FlattenedPose> processed_flat_poses_overall; // Accumulates all FlattenedPose that were actually processed

        auto step_start_time_crop = std::chrono::high_resolution_clock::now();

        for (size_t batch_start_idx = 0; batch_start_idx < all_flattened_poses.size(); batch_start_idx += efficient_max_batch) {
            size_t batch_end_idx = std::min(batch_start_idx + efficient_max_batch, all_flattened_poses.size());

            std::vector<cv::Mat> current_efficient_batch_images;
            std::vector<FlattenedPose> current_efficient_net_batch_flat_poses_local; // Local to this sub-batch

            for (size_t k = batch_start_idx; k < batch_end_idx; ++k) {
                const auto& flat_pose = all_flattened_poses[k];
                const auto& pose = flat_pose.pose;
                // Ensure the original_image_idx is within bounds
                if (flat_pose.original_image_idx >= original_batched_images.size()) {
                    std::cerr << "Error: original_image_idx out of bounds for cropping." << std::endl;
                    continue;
                }
                const cv::Mat& source_image = original_batched_images[flat_pose.original_image_idx];

                if (pose.pts.empty()) continue;

                int min_x = std::min(pose.lx, pose.rx);
                int min_y = std::min(pose.ly, pose.ry);
                int max_x = std::max(pose.lx, pose.rx);
                int max_y = std::max(pose.ly, pose.ry);
                int width = max_x - min_x;
                int height = max_y - min_y;

                int crop_x = std::max(0, static_cast<int>(min_x - width * (scale_factor - 1) / 2));
                int crop_y = std::max(0, static_cast<int>(min_y - height * (scale_factor - 1) / 2));
                int crop_width = std::min(source_image.cols - crop_x, static_cast<int>(width * scale_factor));
                int crop_height = std::min(source_image.rows - crop_y, static_cast<int>(height * scale_factor));

                if (crop_width > 0 && crop_height > 0) {
                    cv::Mat cropped_img = source_image(cv::Rect(crop_x, crop_y, crop_width, crop_height));
                    current_efficient_batch_images.push_back(cropped_img);
                    current_efficient_net_batch_flat_poses_local.push_back(flat_pose); // Store for this sub-batch
                }
            }

            if (current_efficient_batch_images.empty()) {
                continue; // Skip if no valid crops for this sub-batch
            }

            // EfficientNet Preprocess for this sub-batch
            auto step_start_time = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < current_efficient_batch_images.size(); ++i) {
                efficient_model->preprocess(current_efficient_batch_images[i], i); // Process each image with its batch index
            }
            efficient_preprocess_times_thread.push_back(
                 std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - step_start_time).count()
            );

            // EfficientNet Inference for this sub-batch
            step_start_time = std::chrono::high_resolution_clock::now();
            efficient_model->inference();
            efficient_inference_times_thread.push_back(
                 std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - step_start_time).count()
            );

            // EfficientNet Postprocess for this sub-batch
            step_start_time = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<float>> cls_results_sub_batch;
            for(size_t i = 0; i < current_efficient_batch_images.size(); ++i) {
                std::any single_efficient_result_any;
                efficient_model->postprocess(i, params2, single_efficient_result_any);
                try {
                    // Expecting std::vector<float> for each individual crop
                    std::vector<float> cls_result = std::any_cast<std::vector<float>>(single_efficient_result_any);
                    cls_results_sub_batch.push_back(cls_result);
                } catch (const std::bad_any_cast& e) {
                    std::cerr << "EfficientNet Thread: Error casting single crop result for batch index " << i << ": " << e.what() << std::endl;
                    cls_results_sub_batch.push_back({}); // Add empty result
                } catch (...) {
                    std::cerr << "EfficientNet Thread: Unknown error during single crop postprocessing or cast for batch index " << i << "." << std::endl;
                    cls_results_sub_batch.push_back({}); // Add empty result
                }
            }
            efficient_postprocess_times_thread.push_back(
                 std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - step_start_time).count()
            );

            efficient_net_all_results.insert(efficient_net_all_results.end(), cls_results_sub_batch.begin(), cls_results_sub_batch.end());
            processed_flat_poses_overall.insert(processed_flat_poses_overall.end(), current_efficient_net_batch_flat_poses_local.begin(), current_efficient_net_batch_flat_poses_local.end());
        }
        // Accumulate overall crop time after all sub-batches are done
        crop_times_thread.push_back(
             std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - step_start_time_crop).count()
        );


        // Re-associate EfficientNet results back to original YoloPose detections
        std::map<int, std::vector<YoloPose>> updated_batched_detections;
        for (auto const& [image_idx, poses_in_image] : current_batched_pose_detections) {
            updated_batched_detections[image_idx] = poses_in_image; // Copy original poses
        }

        // Apply classification results
        // efficient_net_all_results now contains results for all flattened poses, in the same order as processed_flat_poses_overall
        for (size_t i = 0; i < processed_flat_poses_overall.size(); ++i) {
            const auto& flat_pose = processed_flat_poses_overall[i];
            if (i < efficient_net_all_results.size() && !efficient_net_all_results[i].empty()) {
                if (updated_batched_detections.count(flat_pose.original_image_idx) &&
                    flat_pose.original_pose_idx < updated_batched_detections[flat_pose.original_image_idx].size()) {
                    updated_batched_detections[flat_pose.original_image_idx][flat_pose.original_pose_idx].cls = static_cast<float>(efficient_net_all_results[i][0]);
                } else {
                    std::cerr << "Warning: Could not re-associate result for pose: image_idx=" << flat_pose.original_image_idx
                              << ", pose_idx=" << flat_pose.original_pose_idx << " (target invalid)." << std::endl;
                }
            }
        }

        // Update the reference to store the last processed detections for display in main thread
        // We'll store the entire map for the last processed iteration
        final_display_detections_map_ref = updated_batched_detections;
        processed_count++;
    }
    std::cout << "EfficientNet Thread: Processed a total of " << processed_count << " batches of YoloPose results." << std::endl;
}


// Function to benchmark YOLO Pose and EfficientNet inference with threading
void benchmark_yolo_pose_efficient_threaded(int num_iterations = 1000, bool display_results = true) {
    // Load multiple images for batching
    std::vector<std::string> image_paths = {
        "/opt/images/supermarket/customer1.png",
        "/opt/images/supermarket/customer2.png",
        "/opt/images/supermarket/customer3.png",
        "/opt/images/supermarket/customer4.png",
        "/opt/images/supermarket/customer5.png",
        "/opt/images/supermarket/customer6.png",
        "/opt/images/supermarket/customer7.png",
        "/opt/images/supermarket/customer8.png"
    };

    std::vector<cv::Mat> batched_original_images;
    std::vector<cv::Mat> batched_resized_images; // Renamed for clarity

    for (const std::string& path : image_paths) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << path << std::endl;
            // Handle error, maybe skip or return
            return;
        }
        batched_original_images.push_back(img.clone()); // Keep originals for display later
        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(640, 640));
        batched_resized_images.push_back(resized_img);
    }

    if (batched_resized_images.empty()) {
        std::cerr << "No images loaded for benchmarking." << std::endl;
        return;
    }

    // Parameters for models
    std::map<std::string, std::any> params1{
        {"maximum_batch", 8},
        {"maximum_items", 100},
        {"infer_features", 56},
        {"infer_samples", 8400},
        {"cls", 0.4f},
        {"iou", 0.5f}
    };

    std::map<std::string, std::any> params2{
        {"maximum_batch", 32} // EfficientNet can handle up to 32 crops in a batch
    };

    // To store the very last detection results for display, mapped by original image index
    std::map<int, std::vector<YoloPose>> final_display_detections_map;

    auto start_total_time = std::chrono::high_resolution_clock::now();

    // Start YoloPose (producer) thread
    // Pass the vector of resized images for YoloPose to process as a batch
    std::thread yolo_thread(yolo_pose_thread_func, num_iterations, std::cref(batched_resized_images), std::cref(params1));

    // Start EfficientNet (consumer) thread
    // Pass the *original* images (or resized if drawing on resized makes sense) for cropping reference
    std::thread efficient_thread(efficient_net_thread_func, std::cref(batched_resized_images), std::cref(params2), std::ref(final_display_detections_map));

    // Wait for both threads to complete
    yolo_thread.join();
    efficient_thread.join();

    auto end_total_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_total_time - start_total_time;

    // Calculate overall average time based on the number of iterations initiated by YoloPose
    double avg_total_ms = (total_duration.count() * 1000) / num_iterations;

    std::cout << "\n--- Threaded Efficient YOLO Pose Benchmark Results ---" << std::endl;
    std::cout << "  Total YoloPose batch iterations: " << num_iterations << std::endl;
    std::cout << "  YoloPose batch size: " << batched_resized_images.size() << " images" << std::endl;
    std::cout << "  Total pipeline time: " << total_duration.count() << " seconds" << std::endl;
    std::cout << "  Average total pipeline time per YoloPose batch iteration: " << avg_total_ms << " ms" << std::endl;

    // Calculate and display average times for each step from collected thread vectors
    auto calculate_average = [](const std::vector<long long>& times) {
        if (times.empty()) return 0.0;
        long long sum = std::accumulate(times.begin(), times.end(), 0LL);
        return static_cast<double>(sum) / times.size();
    };

    std::cout << "\n--- Average Time Per Step (from each thread) ---" << std::endl;
    std::cout << "  YoloPose Preprocess (Avg): " << calculate_average(yolo_preprocess_times_thread) << " ms" << std::endl;
    std::cout << "  YoloPose Inference (Avg): " << calculate_average(yolo_inference_times_thread) << " ms" << std::endl;
    std::cout << "  YoloPose Postprocess (Avg): " << calculate_average(yolo_postprocess_times_thread) << " ms" << std::endl;
    std::cout << "  Image Cropping (Avg): " << calculate_average(crop_times_thread) << " ms" << std::endl;
    std::cout << "  EfficientNet Preprocess (Avg): " << calculate_average(efficient_preprocess_times_thread) << " ms" << std::endl;
    std::cout << "  EfficientNet Inference (Avg): " << calculate_average(efficient_inference_times_thread) << " ms" << std::endl;
    std::cout << "  EfficientNet Postprocess (Avg): " << calculate_average(efficient_postprocess_times_thread) << " ms" << std::endl;


    if (display_results) {
        if (!final_display_detections_map.empty()) {
            for (auto const& [image_idx, detections] : final_display_detections_map) {
                if (image_idx < batched_resized_images.size()) {
                    cv::Mat display_image = batched_resized_images[image_idx].clone();
                    draw_pose_results(display_image, detections);
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
    registerModels();
    // Call the new threaded benchmark function
    benchmark_yolo_pose_efficient_threaded(100, true); // Reduced iterations for quicker testing
    return 0;
}
