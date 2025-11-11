/**
 * @file c_yolopose_operations.c
 * @brief Implementation of YOLOv8-Pose specific operations
 *
 * @author TrtEngineToolkits
 * @date 2025-11-10
 */

#include "trtengine_v2/pipelines/yolopose/c_yolopose_structures.h"
#include "trtengine_v2/common/c_operations.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// ============================================================================
//                          Pose-Specific IoU
// ============================================================================

/**
 * @brief Calculate IoU between two pose detections (based on their bounding boxes)
 */
float c_iou_pose(const C_YoloPose* a, const C_YoloPose* b) {
    if (!a || !b) {
        return 0.0f;
    }

    // Use the common IoU function on the detection boxes
    return c_iou_detect(&a->detection, &b->detection);
}

// ============================================================================
//                    Helper Functions for Pose NMS
// ============================================================================

/**
 * @brief Comparison function: sort poses by confidence in descending order
 */
static int compare_pose_by_conf_desc(const void* a, const void* b) {
    const C_YoloPose* pose_a = (const C_YoloPose*)a;
    const C_YoloPose* pose_b = (const C_YoloPose*)b;

    const float conf_a = pose_a->detection.conf;
    const float conf_b = pose_b->detection.conf;

    if (conf_a > conf_b) return -1;
    if (conf_a < conf_b) return 1;
    return 0;
}

// ============================================================================
//                    Pose NMS Implementation
// ============================================================================

/**
 * @brief Non-Maximum Suppression (NMS) for pose detections
 */
void c_nms_pose(
    const C_YoloPose* poses,
    size_t count,
    float iou_threshold,
    C_YoloPose* result,
    size_t* result_count
) {
    // Parameter validation
    if (!poses || !result || !result_count || count == 0) {
        if (result_count) {
            *result_count = 0;
        }
        return;
    }

    // 1. Create temporary array and copy input
    C_YoloPose* sorted_poses = (C_YoloPose*)malloc(count * sizeof(C_YoloPose));
    if (!sorted_poses) {
        *result_count = 0;
        return;
    }
    memcpy(sorted_poses, poses, count * sizeof(C_YoloPose));

    // 2. Sort by confidence in descending order
    qsort(sorted_poses, count, sizeof(C_YoloPose), compare_pose_by_conf_desc);

    // 3. Create suppression flag array
    bool* suppressed = (bool*)calloc(count, sizeof(bool));
    if (!suppressed) {
        free(sorted_poses);
        *result_count = 0;
        return;
    }

    // 4. Execute NMS algorithm
    size_t num_kept = 0;

    for (size_t i = 0; i < count; i++) {
        // Skip if current pose is suppressed
        if (suppressed[i]) {
            continue;
        }

        // Keep current pose
        result[num_kept++] = sorted_poses[i];

        // Check all subsequent poses
        for (size_t j = i + 1; j < count; j++) {
            // Skip if already suppressed
            if (suppressed[j]) {
                continue;
            }

            // Only perform NMS within the same class
            bool same_class = (sorted_poses[i].detection.cls == sorted_poses[j].detection.cls);

            // Calculate IoU (based on bounding boxes)
            if (same_class) {
                float iou_value = c_iou_pose(&sorted_poses[i], &sorted_poses[j]);

                // Suppress if IoU exceeds threshold
                if (iou_value >= iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    // 5. Clean up temporary memory
    free(sorted_poses);
    free(suppressed);

    // 6. Return result count
    *result_count = num_kept;
}
