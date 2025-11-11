/**
 * @file c_yolopose_structures.h
 * @brief YOLOv8-Pose specific data structures
 *
 * This file defines structures specific to YOLO pose estimation models,
 * building upon the common structures defined in trtengine_v2/common/c_structures.h
 *
 * @author TrtEngineToolkits
 * @date 2025-11-10
 */

#ifndef TRTENGINE_V2_PIPELINES_YOLOPOSE_C_YOLOPOSE_STRUCTURES_H
#define TRTENGINE_V2_PIPELINES_YOLOPOSE_C_YOLOPOSE_STRUCTURES_H

#include <stddef.h>  // for size_t
#include "trtengine_v2/common/c_structures.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Number of keypoints in COCO pose format
 */
#define YOLO_POSE_NUM_KEYPOINTS 17

/**
 * @brief YOLO Pose detection result
 *
 * This structure combines a base detection bounding box with pose keypoints.
 * It contains a C_YoloDetect for the person bounding box and 17 keypoints
 * for COCO-format human pose estimation.
 */
typedef struct {
    C_YoloDetect detection;                  ///< Base detection (bounding box, class, confidence)
    C_KeyPoint pts[YOLO_POSE_NUM_KEYPOINTS]; ///< 17 keypoints for pose estimation
} C_YoloPose;

/**
 * @brief Calculate IoU between two pose detections (based on bounding boxes)
 *
 * @param a First pose detection
 * @param b Second pose detection
 * @return float IoU value (0.0 ~ 1.0)
 */
float c_iou_pose(const C_YoloPose* a, const C_YoloPose* b);

/**
 * @brief Non-Maximum Suppression (NMS) for pose detections
 *
 * @param poses Input pose detection array
 * @param count Number of input poses
 * @param iou_threshold IoU threshold for suppression
 * @param result Output pose detection array (caller must allocate)
 * @param result_count Actual number of output poses (returned via pointer)
 */
void c_nms_pose(
    const C_YoloPose* poses,
    size_t count,
    float iou_threshold,
    C_YoloPose* result,
    size_t* result_count
);

#ifdef __cplusplus
}
#endif

#endif // TRTENGINE_V2_PIPELINES_YOLOPOSE_C_YOLOPOSE_STRUCTURES_H
