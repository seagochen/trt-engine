//
// Created by xtcjj on 2025/08/14.
//

#ifndef COMBINEDPROJECT_C_YOLOPOSE_DETECTION_H
#define COMBINEDPROJECT_C_YOLOPOSE_DETECTION_H

#include "trtengine/c_apis/c_structs.h"
#include "trtengine/c_apis/c_pose_pipeline.h"

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief Registers the YoloPose model for pose detection.
     * This function initializes the necessary resources and configurations
     * for the YoloPose model to be used in pose detection tasks.
     */
    void c_register_yolopose_model();

    /**
     * @brief Creates a context for the YoloPose model.
     * This function allocates and initializes the necessary resources
     * for the YoloPose model to process images.
     * 
     * @param model_path The file path to the YoloPose model.
     * @param yolo_max_batch The maximum batch size for YOLO.
     * @param yolo_cls_threshold The classification threshold for YOLO.
     * @param yolo_iou_threshold The IoU threshold for YOLO.
     * @return A pointer to the created context, or NULL on failure.
     */
    void* c_create_yolopose_context(
        const char* model_path,
        int yolo_max_batch,
        float yolo_cls_threshold,
        float yolo_iou_threshold
    );

    /**
     * @brief Processes a batch of images using the YoloPose model.
     * This function takes a batch of input images and passes them through
     * the YoloPose model for pose detection.
     *
     * @param context The context created for the YoloPose model.
     * @param input_images_data The raw image data for the input images.
     * @param widths The widths of the input images.
     * @param heights The heights of the input images.
     * @param channels The number of channels in the input images.
     * @param num_images The number of images in the batch.
     * @param crop_scale_factor The scale factor for cropping the images.
     * @return A structure containing the results of the pose detection.
     */
    C_BatchedPoseResults c_process_batched_images_with_yolopose(
        void* context,
        const unsigned char* const* input_images_data,
        const int* widths,
        const int* heights,
        const int* channels,
        int num_images,
        float crop_scale_factor
    );

    /**
     * @brief Destroys the context for the YoloPose model.
     * This function releases all resources associated with the YoloPose context.
     *
     * @param context The context to destroy.
     */
    void c_destroy_yolopose_context(void* context);


#ifdef __cplusplus
};
#endif

#endif //COMBINEDPROJECT_C_YOLOPOSE_DETECTION_H