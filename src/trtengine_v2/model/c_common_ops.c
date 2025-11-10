//
// Created by TrtEngineToolkits on 2025-11-10.
//
// C implementation of common computer vision operations
// including IoU calculation and Non-Maximum Suppression (NMS)
//

#include "trtengine_v2/model/c_common_ops.h"
#include <stdlib.h>
#include <string.h>

// Macro utilities
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// ============================================================================
//                          IoU Calculation
// ============================================================================

/**
 * @brief 计算两个检测框的 IoU (Intersection over Union)
 */
float c_iou_detect(const C_YoloDetect* a, const C_YoloDetect* b) {
    if (!a || !b) {
        return 0.0f;
    }

    // 计算交集区域的边界
    const int inter_left   = MAX(a->lx, b->lx);
    const int inter_top    = MAX(a->ly, b->ly);
    const int inter_right  = MIN(a->rx, b->rx);
    const int inter_bottom = MIN(a->ry, b->ry);

    // 计算交集区域的宽度和高度
    const int inter_width  = MAX(0, inter_right - inter_left + 1);
    const int inter_height = MAX(0, inter_bottom - inter_top + 1);

    // 计算交集面积
    const int inter_area = inter_width * inter_height;

    // 如果没有交集，返回 0
    if (inter_area <= 0) {
        return 0.0f;
    }

    // 计算两个框的面积
    const int area_a = (a->rx - a->lx + 1) * (a->ry - a->ly + 1);
    const int area_b = (b->rx - b->lx + 1) * (b->ry - b->ly + 1);

    // 计算并集面积
    const float union_area = (float)(area_a + area_b - inter_area) + 1e-6f;

    // 返回 IoU
    return (float)inter_area / union_area;
}

/**
 * @brief 计算两个姿态检测框的 IoU（基于其检测框）
 */
float c_iou_pose(const C_YoloPose* a, const C_YoloPose* b) {
    if (!a || !b) {
        return 0.0f;
    }

    // 使用内部的检测框计算 IoU
    return c_iou_detect(&a->detection, &b->detection);
}

// ============================================================================
//                    Helper Functions for NMS
// ============================================================================

/**
 * @brief 比较函数：按置信度降序排序（用于 qsort）
 */
static int compare_detect_by_conf_desc(const void* a, const void* b) {
    const C_YoloDetect* box_a = (const C_YoloDetect*)a;
    const C_YoloDetect* box_b = (const C_YoloDetect*)b;

    // 降序排列：conf 大的排在前面
    if (box_a->conf > box_b->conf) return -1;
    if (box_a->conf < box_b->conf) return 1;
    return 0;
}

/**
 * @brief 比较函数：按姿态置信度降序排序
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
//                    NMS Implementation
// ============================================================================

/**
 * @brief 对检测框进行非极大值抑制 (NMS)
 */
void c_nms_detect(
    const C_YoloDetect* boxes,
    size_t count,
    float iou_threshold,
    C_YoloDetect* result,
    size_t* result_count
) {
    // 参数检查
    if (!boxes || !result || !result_count || count == 0) {
        if (result_count) {
            *result_count = 0;
        }
        return;
    }

    // 1. 创建临时数组并复制输入（避免修改原数组）
    C_YoloDetect* sorted_boxes = (C_YoloDetect*)malloc(count * sizeof(C_YoloDetect));
    if (!sorted_boxes) {
        *result_count = 0;
        return;
    }
    memcpy(sorted_boxes, boxes, count * sizeof(C_YoloDetect));

    // 2. 按置信度降序排序
    qsort(sorted_boxes, count, sizeof(C_YoloDetect), compare_detect_by_conf_desc);

    // 3. 创建抑制标记数组
    bool* suppressed = (bool*)calloc(count, sizeof(bool));
    if (!suppressed) {
        free(sorted_boxes);
        *result_count = 0;
        return;
    }

    // 4. 执行 NMS 算法
    size_t num_kept = 0;

    for (size_t i = 0; i < count; i++) {
        // 如果当前框已被抑制，跳过
        if (suppressed[i]) {
            continue;
        }

        // 保留当前框
        result[num_kept++] = sorted_boxes[i];

        // 检查后续所有框
        for (size_t j = i + 1; j < count; j++) {
            // 如果已被抑制，跳过
            if (suppressed[j]) {
                continue;
            }

            // 只在同一类别间进行 NMS
            bool same_class = (sorted_boxes[i].cls == sorted_boxes[j].cls);

            // 计算 IoU
            if (same_class) {
                float iou_value = c_iou_detect(&sorted_boxes[i], &sorted_boxes[j]);

                // 如果 IoU 超过阈值，抑制该框
                if (iou_value >= iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    // 5. 清理临时内存
    free(sorted_boxes);
    free(suppressed);

    // 6. 返回结果数量
    *result_count = num_kept;
}

/**
 * @brief 对姿态检测进行非极大值抑制 (NMS)
 */
void c_nms_pose(
    const C_YoloPose* poses,
    size_t count,
    float iou_threshold,
    C_YoloPose* result,
    size_t* result_count
) {
    // 参数检查
    if (!poses || !result || !result_count || count == 0) {
        if (result_count) {
            *result_count = 0;
        }
        return;
    }

    // 1. 创建临时数组并复制输入
    C_YoloPose* sorted_poses = (C_YoloPose*)malloc(count * sizeof(C_YoloPose));
    if (!sorted_poses) {
        *result_count = 0;
        return;
    }
    memcpy(sorted_poses, poses, count * sizeof(C_YoloPose));

    // 2. 按置信度降序排序
    qsort(sorted_poses, count, sizeof(C_YoloPose), compare_pose_by_conf_desc);

    // 3. 创建抑制标记数组
    bool* suppressed = (bool*)calloc(count, sizeof(bool));
    if (!suppressed) {
        free(sorted_poses);
        *result_count = 0;
        return;
    }

    // 4. 执行 NMS 算法
    size_t num_kept = 0;

    for (size_t i = 0; i < count; i++) {
        // 如果当前姿态已被抑制，跳过
        if (suppressed[i]) {
            continue;
        }

        // 保留当前姿态
        result[num_kept++] = sorted_poses[i];

        // 检查后续所有姿态
        for (size_t j = i + 1; j < count; j++) {
            // 如果已被抑制，跳过
            if (suppressed[j]) {
                continue;
            }

            // 只在同一类别间进行 NMS
            bool same_class = (sorted_poses[i].detection.cls == sorted_poses[j].detection.cls);

            // 计算 IoU（基于检测框）
            if (same_class) {
                float iou_value = c_iou_pose(&sorted_poses[i], &sorted_poses[j]);

                // 如果 IoU 超过阈值，抑制该姿态
                if (iou_value >= iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    // 5. 清理临时内存
    free(sorted_poses);
    free(suppressed);

    // 6. 返回结果数量
    *result_count = num_kept;
}
