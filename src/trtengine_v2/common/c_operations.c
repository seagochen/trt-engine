/**
 * @file c_operations.c
 * @brief Implementation of common YOLO operations
 *
 * @author TrtEngineToolkits
 * @date 2025-11-10
 */

#include "trtengine_v2/common/c_operations.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Macro utilities
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// ============================================================================
//                          IoU Calculation
// ============================================================================

/**
 * @brief Calculate Intersection over Union (IoU) between two detection boxes
 */
float c_iou_detect(const C_YoloDetect* a, const C_YoloDetect* b) {
    if (!a || !b) {
        return 0.0f;
    }

    // Calculate intersection boundaries
    const int inter_left   = MAX(a->lx, b->lx);
    const int inter_top    = MAX(a->ly, b->ly);
    const int inter_right  = MIN(a->rx, b->rx);
    const int inter_bottom = MIN(a->ry, b->ry);

    // Calculate intersection width and height
    const int inter_width  = MAX(0, inter_right - inter_left + 1);
    const int inter_height = MAX(0, inter_bottom - inter_top + 1);

    // Calculate intersection area
    const int inter_area = inter_width * inter_height;

    // If no intersection, return 0
    if (inter_area <= 0) {
        return 0.0f;
    }

    // Calculate areas of both boxes
    const int area_a = (a->rx - a->lx + 1) * (a->ry - a->ly + 1);
    const int area_b = (b->rx - b->lx + 1) * (b->ry - b->ly + 1);

    // Calculate union area
    const float union_area = (float)(area_a + area_b - inter_area) + 1e-6f;

    // Return IoU
    return (float)inter_area / union_area;
}

// ============================================================================
//                    Helper Functions for NMS
// ============================================================================

/**
 * @brief Comparison function: sort by confidence in descending order (for qsort)
 */
static int compare_detect_by_conf_desc(const void* a, const void* b) {
    const C_YoloDetect* box_a = (const C_YoloDetect*)a;
    const C_YoloDetect* box_b = (const C_YoloDetect*)b;

    // Descending order: higher confidence first
    if (box_a->conf > box_b->conf) return -1;
    if (box_a->conf < box_b->conf) return 1;
    return 0;
}

// ============================================================================
//                    NMS Implementation
// ============================================================================

/**
 * @brief Non-Maximum Suppression (NMS) for detection boxes
 */
void c_nms_detect(
    const C_YoloDetect* boxes,
    size_t count,
    float iou_threshold,
    C_YoloDetect* result,
    size_t* result_count
) {
    // Parameter validation
    if (!boxes || !result || !result_count || count == 0) {
        if (result_count) {
            *result_count = 0;
        }
        return;
    }

    // 1. Create temporary array and copy input (avoid modifying original)
    C_YoloDetect* sorted_boxes = (C_YoloDetect*)malloc(count * sizeof(C_YoloDetect));
    if (!sorted_boxes) {
        *result_count = 0;
        return;
    }
    memcpy(sorted_boxes, boxes, count * sizeof(C_YoloDetect));

    // 2. Sort by confidence in descending order
    qsort(sorted_boxes, count, sizeof(C_YoloDetect), compare_detect_by_conf_desc);

    // 3. Create suppression flag array
    bool* suppressed = (bool*)calloc(count, sizeof(bool));
    if (!suppressed) {
        free(sorted_boxes);
        *result_count = 0;
        return;
    }

    // 4. Execute NMS algorithm
    size_t num_kept = 0;

    for (size_t i = 0; i < count; i++) {
        // Skip if current box is suppressed
        if (suppressed[i]) {
            continue;
        }

        // Keep current box
        result[num_kept++] = sorted_boxes[i];

        // Check all subsequent boxes
        for (size_t j = i + 1; j < count; j++) {
            // Skip if already suppressed
            if (suppressed[j]) {
                continue;
            }

            // Only perform NMS within the same class
            bool same_class = (sorted_boxes[i].cls == sorted_boxes[j].cls);

            // Calculate IoU
            if (same_class) {
                float iou_value = c_iou_detect(&sorted_boxes[i], &sorted_boxes[j]);

                // Suppress if IoU exceeds threshold
                if (iou_value >= iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    // 5. Clean up temporary memory
    free(sorted_boxes);
    free(suppressed);

    // 6. Return result count
    *result_count = num_kept;
}
