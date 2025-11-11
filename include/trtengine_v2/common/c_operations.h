/**
 * @file c_operations.h
 * @brief Common operations for YOLO models (NMS, IoU, etc.)
 *
 * This file provides generic algorithms that can be used across different
 * YOLO model types, such as Non-Maximum Suppression (NMS) and Intersection
 * over Union (IoU) calculations.
 *
 * @author TrtEngineToolkits
 * @date 2025-11-10
 */

#ifndef TRTENGINE_V2_COMMON_C_OPERATIONS_H
#define TRTENGINE_V2_COMMON_C_OPERATIONS_H

#include "trtengine_v2/common/c_structures.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate Intersection over Union (IoU) between two detection boxes
 *
 * @param a First detection box
 * @param b Second detection box
 * @return float IoU value (0.0 ~ 1.0)
 */
float c_iou_detect(const C_YoloDetect* a, const C_YoloDetect* b);

/**
 * @brief Non-Maximum Suppression (NMS) for detection boxes
 *
 * This function sorts the input detection boxes by confidence and applies
 * NMS algorithm to filter overlapping boxes, keeping only the highest
 * confidence boxes.
 *
 * @param boxes Input detection box array
 * @param count Number of input boxes
 * @param iou_threshold IoU threshold for suppression (typically 0.45-0.7)
 * @param result Output detection box array (caller must allocate, size >= count)
 * @param result_count Actual number of output boxes (returned via pointer)
 *
 * @note Input array is not modified
 * @note Result array must be pre-allocated by caller (same size as count recommended)
 * @note Function automatically handles different classes (NMS per class)
 */
void c_nms_detect(
    const C_YoloDetect* boxes,
    size_t count,
    float iou_threshold,
    C_YoloDetect* result,
    size_t* result_count
);

#ifdef __cplusplus
}
#endif

#endif // TRTENGINE_V2_COMMON_C_OPERATIONS_H
