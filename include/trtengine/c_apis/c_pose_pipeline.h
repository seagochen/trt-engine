//
// Created by user on 6/25/25.
//

#ifndef C_POSE_PIPELINE_H
#define C_POSE_PIPELINE_H

#include "trtengine/c_apis/c_structs.h"

#ifdef __cplusplus
extern "C" {
#endif

    // Opaque pointer for the main processing context.
    // Users of this API will treat this as an uninterpretable handle.
    typedef void* YoloEfficientContext;

    /**
     * @brief Registers the models required for the pose detection pipeline.
     *
     * This function must be called before creating a YoloEfficientContext.
     * It initializes the necessary models and resources for the pose detection pipeline.
     */
    void c_register_models();

    /**
     * @brief Initializes the YoloPose and EfficientNet models.
     *
     * @param yolo_engine_path Path to the YoloV8-Pose TensorRT engine file.
     * @param efficient_engine_path Path to the EfficientNet TensorRT engine file.
     * @param yolo_max_batch Maximum batch size for the YoloPose model.
     * @param efficient_max_batch Maximum batch size for the EfficientNet model.
     * @param yolo_cls_thresh Confidence threshold for YoloPose detections.
     * @param yolo_iou_thresh IoU threshold for YoloPose NMS.
     * @return A pointer to the initialized context (YoloEfficientContext*), or NULL on failure.
     */
    YoloEfficientContext* c_create_pose_pipeline(
        const char* yolo_engine_path,
        const char* efficient_engine_path,
        int yolo_max_batch,
        int efficient_max_batch,
        float yolo_cls_thresh,
        float yolo_iou_thresh
    );

    /**
     * @brief Processes a batch of images through the YoloPose and EfficientNet pipeline.
     *
     * This function takes raw image data, performs pose detection with YoloPose,
     * crops detected persons, and then classifies them using EfficientNet.
     *
     * @param context A pointer to a YoloEfficientContext pointer. This allows the API to
     * modify the caller's context pointer (e.g., set to NULL on destruction).
     * Must be a valid pointer to a valid context obtained from create_yolo_efficient_context.
     * @param input_images_data An array of pointers to raw image data (e.g., RGB, HWC, 8-bit unsigned char).
     * Each pointer corresponds to one image in the batch.
     * @param widths An array containing the width of each image in pixels.
     * @param heights An array containing the height of each image in pixels.
     * @param channels An array containing the number of channels for each image (e.g., 3 for RGB).
     * @param num_images The number of images in the input batch.
     * @param crop_scale_factor A scaling factor for cropping the detected persons.
     * @return A C_BatchedPoseResults structure containing the processed pose detections
     * with updated classification scores. The caller is responsible for freeing
     * the memory using free_batched_pose_results().
     * Returns a structure with num_images=0 and results=NULL on error.
     */
    C_BatchedPoseResults c_process_batched_images(
        YoloEfficientContext* context, // Changed to pointer-to-pointer
        const unsigned char* const* input_images_data,
        const int* widths,
        const int* heights,
        const int* channels,
        int num_images,
        float crop_scale_factor
    );

    /**
     * @brief Frees the memory allocated for C_BatchedPoseResults.
     *
     * This function must be called by the API consumer to avoid memory leaks
     * after they are done using the results returned by process_image_batch().
     *
     * @param results A pointer to the C_BatchedPoseResults structure to be freed.
     * The pointer itself will be set to NULL after freeing.
     */
    void c_free_batched_pose_results(C_BatchedPoseResults* results);

    /**
     * @brief Destroys the YoloEfficientContext and releases all associated model resources.
     *
     * This function must be called when the context is no longer needed to free GPU memory
     * and other resources. The context pointer provided will be set to NULL after destruction.
     *
     * @param context A pointer to a YoloEfficientContext pointer to be destroyed.
     * The pointer itself will be set to NULL after destruction.
     */
    void c_destroy_pose_pipeline(YoloEfficientContext* context);

#ifdef __cplusplus
}
#endif

#endif //C_POSE_PIPELINE_H
