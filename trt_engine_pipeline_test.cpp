#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <any>
#include <map>
#include <chrono>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp> // For cv::imwrite
#include <sys/stat.h> // For mkdir on Linux/Unix
#ifdef _WIN32
#include <direct.h> // For _mkdir on Windows
#endif

#include "trtengine/c_apis/c_pose_detection.h" // Includes C_Inference_Result and C_Extended_Person_Feats
#include "trtengine/utils/logger.h"


/**
 * @brief Visualizes detected persons by cropping them from the original image,
 * draws keypoints on the cutouts, and saves them as separate image files.
 *
 * @param source_images_640x640 A vector of the images (ALREADY RESIZED TO 640x640)
 * that correspond to the coordinate system of `c_results_array`.
 * @param c_results_array Pointer to the array of C_Inference_Result received from the pipeline.
 * @param num_images_processed The number of elements in c_results_array.
 * @param output_dir The directory where the cutout images will be saved.
 */
void visualize_and_save_person_cutouts(
    const std::vector<cv::Mat>& source_images_640x640, // Changed parameter name to reflect its content
    const C_Inference_Result* c_results_array,
    int num_images_processed,
    const std::string& output_dir = "person_cutouts")
{
    if (c_results_array == nullptr || num_images_processed <= 0) {
        LOG_WARNING("Visualizer", "No results to visualize or invalid input array.");
        return;
    }
    if (source_images_640x640.empty() || source_images_640x640.size() != num_images_processed) {
        LOG_ERROR("Visualizer", "Source images vector (640x640) is empty or its size does not match processed results count. Cannot visualize.");
        return;
    }

    // Create output directory if it doesn't exist
    #ifdef _WIN32
        _mkdir(output_dir.c_str());
    #else
        mkdir(output_dir.c_str(), 0777); // Permissions 0777
    #endif

    LOG_INFO("Visualizer", "Starting visualization and saving of person cutouts to: " + output_dir);

    for (int i = 0; i < num_images_processed; ++i) {
        const C_Inference_Result& image_result = c_results_array[i];
        // Use the already resized 640x640 image directly
        const cv::Mat& current_source_image_640x640 = source_images_640x640[i];

        if (image_result.num_detected > 0 && image_result.detections != nullptr) {
            for (int j = 0; j < image_result.num_detected; ++j) {
                const C_Extended_Person_Feats& person = image_result.detections[j];

                // Scale factor for cropping, use the same as in aux_batch_process.cpp
                const float crop_scale_factor = 1.2f;

                // Calculate bounding box for cropping (from 640x640 coordinate space)
                int min_x_bb = static_cast<int>(person.box.x1);
                int min_y_bb = static_cast<int>(person.box.y1);
                int max_x_bb = static_cast<int>(person.box.x2);
                int max_y_bb = static_cast<int>(person.box.y2);

                int width_bb = max_x_bb - min_x_bb;
                int height_bb = max_y_bb - min_y_bb;

                // Calculate crop region with scale factor and clamp to image boundaries
                int crop_x = std::max(0, static_cast<int>(min_x_bb - width_bb * (crop_scale_factor - 1) / 2));
                int crop_y = std::max(0, static_cast<int>(min_y_bb - height_bb * (crop_scale_factor - 1) / 2));
                int crop_width = static_cast<int>(width_bb * crop_scale_factor);
                int crop_height = static_cast<int>(height_bb * crop_scale_factor);

                // Adjust crop_width and crop_height to not exceed image bounds
                crop_width = std::min(current_source_image_640x640.cols - crop_x, crop_width);
                crop_height = std::min(current_source_image_640x640.rows - crop_y, crop_height);

                if (crop_width <= 0 || crop_height <= 0) {
                    LOG_WARNING("Visualizer", "Invalid crop dimensions for image " + std::to_string(i) + ", person " + std::to_string(j) + ". Skipping cutout.");
                    continue;
                }

                cv::Rect crop_rect(crop_x, crop_y, crop_width, crop_height);
                // Extract the person cutout
                cv::Mat person_cutout = current_source_image_640x640(crop_rect).clone();

                // Draw keypoints on the cutout
                for (int k = 0; k < 17; ++k) {
                    if (person.pts[k].score > 0.0f) { // Only draw valid keypoints
                        // Convert keypoint coordinate from 640x640 space to cutout's local space
                        cv::Point kp_on_cutout(
                            static_cast<int>(person.pts[k].x - crop_x),
                            static_cast<int>(person.pts[k].y - crop_y)
                        );
                        // Draw a red circle for the keypoint
                        cv::circle(person_cutout, kp_on_cutout, 3, cv::Scalar(0, 0, 255), -1); // Red, filled
                    }
                }

                // Add text for class ID and confidence
                std::string label = "Class: " + std::to_string(static_cast<int>(person.class_id)) +
                                    " Conf: " + std::to_string(static_cast<int>(person.confidence * 100)) + "%";
                cv::putText(person_cutout, label, cv::Point(5, 15),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1); // White text

                // Save the cutout image
                std::string output_filename = output_dir + "/image_" + std::to_string(i) + "_person_" + std::to_string(j) + "_class_" + std::to_string(static_cast<int>(person.class_id)) + ".png";
                if (cv::imwrite(output_filename, person_cutout)) {
                    LOG_INFO("Visualizer", "Saved cutout to: " + output_filename);
                } else {
                    LOG_ERROR("Visualizer", "Failed to save cutout to: " + output_filename);
                }
            }
        } else if (image_result.num_detected == -1) {
            LOG_WARNING("Visualizer", "Image " + std::to_string(i) + ": Processing error, no cutouts generated.");
        } else {
            LOG_INFO("Visualizer", "Image " + std::to_string(i) + ": No persons detected, no cutouts generated.");
        }
    }
    LOG_INFO("Visualizer", "Finished visualizing and saving person cutouts.");
}

int main()
{
    // -------------------------------------------- Initialization ----------------------------------------

    // Define the paths to the YOLO and EfficientNet engines
    std::string yolo_engine_path = "/opt/models/yolov8s-pose_extend.engine";
    std::string efficient_engine_path = "/opt/models/efficientnet_b0_feat_logits.engine";

    // Initialize pose_extend detection pipeline
    if (!init_pose_detection_pipeline(
        yolo_engine_path.c_str(),
        efficient_engine_path.c_str(),
        100, 0.4f, 0.3f)) // Using 0.3f for IOU to potentially get more results
    {
        deinit_pose_detection_pipeline();
        LOG_ERROR("TrtEngineDemo", "Initialization failed for pose_extend detection pipeline.");
        return -1;
    }

    // Load test images
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

    // Read images (original sizes)
    std::vector<cv::Mat> original_images_blobs; // Renamed for clarity
    for (const auto& image_path : batch_images_paths)
    {
        cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
        cv::resize(img, img, cv::Size(640, 640)); // Resize to 640x640 for consistency

        if (img.empty())
        {
            LOG_ERROR("TrtEngineDemo", "Failed to read image: " + image_path);
            deinit_pose_detection_pipeline();
            return -1;
        }
        original_images_blobs.push_back(img);
    }

    LOG_DEBUG_V5_TOPIC("TrtEngineDemo", "#1 LOAD_IMAGES", 
        "Loaded " + std::to_string(original_images_blobs.size()) + " images (original size) for pose_extend detection.");

    // ----------------------------------------- Add Images to Pose Detection Pipeline ----------------------------------------

    // Use a timer to measure the duration of the pose detection pipeline run
    auto start_time = std::chrono::high_resolution_clock::now();

    // Add images to the pose_extend detection pipeline queue (using original_images_blobs's data, which will be resized internally)
    for (size_t i = 0; i < original_images_blobs.size(); ++i)
    {
        add_image_to_pose_detection_pipeline(original_images_blobs[i].data, original_images_blobs[i].cols, original_images_blobs[i].rows);
        LOG_DEBUG_V1("TrtEngineDemo",
            "Added image " + batch_images_paths[i] + " to pipeline queue, original size: " +
            std::to_string(original_images_blobs[i].cols) + "x" + std::to_string(original_images_blobs[i].rows));
    }

    // Ensure all images are added to the pipeline before running detection
    auto end_time =  std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> add_duration = end_time - start_time;

    LOG_DEBUG_V5_TOPIC("TrtEngineDemo", "#2 ADD_IMAGES",
        "Added " + std::to_string(original_images_blobs.size()) + " images to the pipeline in " + std::to_string(add_duration.count()) + " ms.");

    // ---------------------------------------- Run Pose Detection Pipeline ----------------------------------------

    C_Inference_Result* c_results_array = nullptr;
    int num_images_processed = 0;

    // Measure the time taken to run the pose detection pipeline
    start_time = std::chrono::high_resolution_clock::now();

    if (!run_pose_detection_pipeline(&c_results_array, &num_images_processed))
    {
        LOG_ERROR("TrtEngineDemo", "Pose detection pipeline failed to run.");
        deinit_pose_detection_pipeline();
        return -1;
    }

    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    LOG_DEBUG_V5_TOPIC("TrtEngineDemo", "#3 RUN_PIPELINE",
        "Pose detection pipeline processed " + std::to_string(num_images_processed) + " images in " + std::to_string(duration.count()) + " ms.");

    // ---------------------------------------- Processing Results ----------------------------------------
    start_time = std::chrono::high_resolution_clock::now();

    if (c_results_array != nullptr && num_images_processed > 0) {
        for (int i = 0; i < num_images_processed; ++i) {
            const C_Inference_Result& image_result = c_results_array[i];

            std::cout << "\n--- Image " << (i + 1) << " (" << batch_images_paths[i] << ") ---\n";
            std::cout << "  Detected " << image_result.num_detected << " persons.\n";

            if (image_result.num_detected > 0 && image_result.detections != nullptr) {
                for (int j = 0; j < image_result.num_detected; ++j) {
                    const C_Extended_Person_Feats& person = image_result.detections[j];
                    std::cout << "    Person " << (j + 1) << ":\n";
                    std::cout << "      Box: (" << person.box.x1 << ", " << person.box.y1 << ") - ("
                              << person.box.x2 << ", " << person.box.y2 << ")\n";
                    std::cout << "      Confidence: " << person.confidence << "\n";
                    std::cout << "      Class ID (EfficientNet): " << person.class_id << "\n";

                    int valid_kps_count = 0;
                    for (int k = 0; k < 17; ++k) {
                        if (person.pts[k].score > 0.0f) {
                            std::cout << "        KP " << k << ": (" << person.pts[k].x << ", "
                                      << person.pts[k].y << "), Score: " << person.pts[k].score << "\n";
                            valid_kps_count++;
                        }
                    }
                    if (valid_kps_count == 0) {
                        std::cout << "        No valid keypoints detected.\n";
                    }
                }
            } else if (image_result.num_detected == -1) {
                 std::cout << "    Processing error occurred for this image.\n";
            } else {
                 std::cout << "    No persons detected in this image.\n";
            }
        }
    } else {
        std::cout << "No results array returned or no images processed.\n";
    }

    // --- Visualize and Save Cutouts ---
    std::string output_directory_name = "visual_cutouts";
    // Pass the already resized 640x640 images for visualization
    visualize_and_save_person_cutouts(original_images_blobs, c_results_array, num_images_processed, output_directory_name);

    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> processing_duration = end_time - start_time;

    LOG_DEBUG_V5_TOPIC("TrtEngineDemo", "#4 PROCESS_RESULTS",
        "Processed results for " + std::to_string(num_images_processed) + " images in " + std::to_string(processing_duration.count()) + " ms.");

    // -------------------------------------------- Release Resources --------------------------------

    // Release inference results
    LOG_INFO("TrtEngineDemo", "Releasing inference results.");
    release_inference_result(c_results_array, num_images_processed);
    LOG_INFO("TrtEngineDemo", "Inference results released successfully.");

    // Deinitialize models
    LOG_INFO("TrtEngineDemo", "Starting to deinitialize pose_extend detection pipeline.");
    deinit_pose_detection_pipeline();
    LOG_INFO("TrtEngineDemo", "Pose detection pipeline deinitialized successfully.");

    return 0;
}