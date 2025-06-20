#ifndef AUX_BATCH_PROCESS_H
#define AUX_BATCH_PROCESS_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <map>    // For std::map
#include <any>    // For std::any

#include "trtengine/c_apis/c_dstruct.h"
#include "trtengine/serverlet/models/infer_model_multi.h"

// --------------------------------- CPP Internal Struct Definition --------------------------
// Used for internal logic processing within TrtEngine.

// Represents the detection result for a single image, including the processed image itself.
struct InferenceResult {
    int num_detected; // Number of persons detected in this specific image
    cv::Mat processed_image; // The image (e.g., 640x640) that was processed for this result
    std::vector<C_Extended_Person_Feats> detections; // Detected persons for this image
};


/**
 * @brief Processes a batch of input images using the pose detection model.
 * This function handles image preprocessing, inference, and postprocessing for pose detection.
 * It consumes images from the input vector.
 *
 * @param images A vector of cv::Mat images to be processed. This vector will be consumed (emptied).
 * @param pose_model A unique_ptr to the initialized YOLOv8 pose estimation model.
 * @param pose_pp_params A map containing post-processing parameters for the YOLOv8 pose model (e.g., cls, iou).
 * @return A vector of InferenceResult, where each element corresponds to one input image.
 * Returns an empty vector if an error occurs or no images are provided.
 */
std::vector<InferenceResult> run_pose_detection_stage(
    std::vector<cv::Mat>& images, // Input images (will be consumed/emptied)
    const std::unique_ptr<InferModelBaseMulti>& pose_model,
    const std::map<std::string, std::any>& pose_pp_params
);

/**
 * @brief Processes the results from the pose detection stage using an EfficientNet model.
 * This function takes the output from the pose detection stage and runs it through an EfficientNet model
 * to classify or extract features from the detected persons.
 *
 * @param pose_results A vector of InferenceResult containing results from the pose detection stage.
 * @param efficient_model A unique_ptr to the initialized EfficientNet model for further processing.
 * @param efficient_pp_params A map containing post-processing parameters for the EfficientNet model.
 * @return A vector of InferenceResult, where each element corresponds to one input image with additional features.
 */
std::vector<InferenceResult> run_efficientnet_stage(
    const std::vector<InferenceResult>& pose_results, // Input results from the pose detection stage
    const std::unique_ptr<InferModelBaseMulti>& efficient_model,
    const std::map<std::string, std::any>& efficient_pp_params
);

#endif // AUX_BATCH_PROCESS_H