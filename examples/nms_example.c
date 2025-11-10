/**
 * @file nms_example.c
 * @brief Example usage of C NMS (Non-Maximum Suppression) implementation
 *
 * This example demonstrates how to use the pure C NMS functions
 * for object detection and pose estimation.
 */

#include "trtengine_v2/common/c_operations.h"
#include "trtengine_v2/pipelines/yolopose/c_yolopose_structures.h"
#include <stdio.h>
#include <stdlib.h>

void print_detect(const C_YoloDetect* box) {
    printf("  Box: [%d,%d,%d,%d] cls=%d conf=%.3f\n",
           box->lx, box->ly, box->rx, box->ry, box->cls, box->conf);
}

void print_pose(const C_YoloPose* pose) {
    printf("  Pose: Box[%d,%d,%d,%d] cls=%d conf=%.3f\n",
           pose->detection.lx, pose->detection.ly,
           pose->detection.rx, pose->detection.ry,
           pose->detection.cls, pose->detection.conf);
    printf("    First keypoint: (%.2f, %.2f, conf=%.3f)\n",
           pose->pts[0].x, pose->pts[0].y, pose->pts[0].conf);
}

int main() {
    printf("=== C NMS Example ===\n\n");

    // ========================================================================
    // Example 1: Object Detection NMS
    // ========================================================================
    printf("--- Example 1: Object Detection NMS ---\n");

    // Create some overlapping detection boxes (person class)
    C_YoloDetect detections[] = {
        {100, 100, 200, 300, 0, 0.95f},  // High confidence
        {105, 105, 205, 305, 0, 0.90f},  // Overlapping with first
        {110, 110, 210, 310, 0, 0.85f},  // Overlapping with first
        {300, 150, 400, 350, 0, 0.92f},  // Different location
        {150, 200, 250, 400, 1, 0.88f},  // Different class (car)
        {155, 205, 255, 405, 1, 0.83f},  // Overlapping with previous car
    };

    size_t num_detections = sizeof(detections) / sizeof(detections[0]);

    printf("Input detections: %zu\n", num_detections);
    for (size_t i = 0; i < num_detections; i++) {
        print_detect(&detections[i]);
    }

    // Apply NMS
    C_YoloDetect* nms_result = (C_YoloDetect*)malloc(num_detections * sizeof(C_YoloDetect));
    size_t nms_count = 0;
    float iou_threshold = 0.45f;

    c_nms_detect(detections, num_detections, iou_threshold, nms_result, &nms_count);

    printf("\nAfter NMS (threshold=%.2f): %zu boxes kept\n", iou_threshold, nms_count);
    for (size_t i = 0; i < nms_count; i++) {
        print_detect(&nms_result[i]);
    }

    free(nms_result);

    // ========================================================================
    // Example 2: Pose Estimation NMS
    // ========================================================================
    printf("\n--- Example 2: Pose Estimation NMS ---\n");

    // Create some pose detections with overlapping bounding boxes
    C_YoloPose poses[3];

    // Pose 1: High confidence
    poses[0].detection = (C_YoloDetect){50, 50, 150, 250, 0, 0.93f};
    for (int i = 0; i < 17; i++) {
        poses[0].pts[i] = (C_KeyPoint){100.0f + i * 5, 150.0f + i * 3, 0.9f};
    }

    // Pose 2: Overlapping with Pose 1, lower confidence
    poses[1].detection = (C_YoloDetect){55, 55, 155, 255, 0, 0.87f};
    for (int i = 0; i < 17; i++) {
        poses[1].pts[i] = (C_KeyPoint){105.0f + i * 5, 155.0f + i * 3, 0.85f};
    }

    // Pose 3: Different location
    poses[2].detection = (C_YoloDetect){200, 100, 300, 300, 0, 0.91f};
    for (int i = 0; i < 17; i++) {
        poses[2].pts[i] = (C_KeyPoint){250.0f + i * 5, 200.0f + i * 3, 0.88f};
    }

    size_t num_poses = 3;

    printf("Input poses: %zu\n", num_poses);
    for (size_t i = 0; i < num_poses; i++) {
        print_pose(&poses[i]);
    }

    // Apply NMS to poses
    C_YoloPose* pose_result = (C_YoloPose*)malloc(num_poses * sizeof(C_YoloPose));
    size_t pose_count = 0;

    c_nms_pose(poses, num_poses, iou_threshold, pose_result, &pose_count);

    printf("\nAfter NMS (threshold=%.2f): %zu poses kept\n", iou_threshold, pose_count);
    for (size_t i = 0; i < pose_count; i++) {
        print_pose(&pose_result[i]);
    }

    free(pose_result);

    // ========================================================================
    // Example 3: IoU Calculation
    // ========================================================================
    printf("\n--- Example 3: IoU Calculation ---\n");

    C_YoloDetect box1 = {100, 100, 200, 200, 0, 0.9f};
    C_YoloDetect box2 = {150, 150, 250, 250, 0, 0.8f};
    C_YoloDetect box3 = {300, 300, 400, 400, 0, 0.85f};

    float iou_12 = c_iou_detect(&box1, &box2);
    float iou_13 = c_iou_detect(&box1, &box3);

    printf("Box 1: [100,100,200,200]\n");
    printf("Box 2: [150,150,250,250]\n");
    printf("Box 3: [300,300,400,400]\n");
    printf("IoU(Box1, Box2) = %.4f (overlapping)\n", iou_12);
    printf("IoU(Box1, Box3) = %.4f (no overlap)\n", iou_13);

    printf("\n=== Example Complete ===\n");

    return 0;
}
