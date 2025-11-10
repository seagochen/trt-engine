/**
 * @file c_structures.h
 * @brief Common data structures for YOLO models (Detection, Pose, Segmentation, etc.)
 *
 * This file defines base structures that are shared across different YOLO model types.
 * Model-specific structures should be defined in their respective pipeline directories.
 *
 * @author TrtEngineToolkits
 * @date 2025-11-10
 */

#ifndef TRTENGINE_V2_COMMON_C_STRUCTURES_H
#define TRTENGINE_V2_COMMON_C_STRUCTURES_H

#include <stddef.h>  // for size_t

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Single keypoint with coordinates and confidence
 *
 * Used in pose estimation models (e.g., YOLOv8-pose)
 */
typedef struct {
    float x;        ///< X coordinate
    float y;        ///< Y coordinate
    float conf;     ///< Confidence score (0.0 - 1.0)
} C_KeyPoint;

/**
 * @brief Basic YOLO detection bounding box
 *
 * This is the base structure used by all YOLO models (detection, pose, segmentation)
 */
typedef struct {
    int lx;         ///< Left X coordinate
    int ly;         ///< Top Y coordinate
    int rx;         ///< Right X coordinate
    int ry;         ///< Bottom Y coordinate
    int cls;        ///< Class index
    float conf;     ///< Detection confidence score (0.0 - 1.0)
} C_YoloDetect;

/**
 * @brief Input image structure
 *
 * Common structure for passing images to inference pipelines
 */
typedef struct {
    unsigned char* data;    ///< Image data pointer (RGB format)
    int width;              ///< Image width in pixels
    int height;             ///< Image height in pixels
    int channels;           ///< Number of channels (3 for RGB)
} C_ImageInput;

/**
 * @brief Batch of input images
 *
 * Structure for batch inference
 */
typedef struct {
    C_ImageInput* images;   ///< Array of images
    size_t count;           ///< Number of images in the batch
} C_ImageBatch;

#ifdef __cplusplus
}
#endif

#endif // TRTENGINE_V2_COMMON_C_STRUCTURES_H
