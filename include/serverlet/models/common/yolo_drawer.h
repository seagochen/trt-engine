//
// Created by user on 6/17/25.
//

#ifndef YOLO_DRAWER_H
#define YOLO_DRAWER_H

#include <opencv2/opencv.hpp>
#include "serverlet/models/common/yolo_dstruct.h"


// Define colors in BGR format
inline auto BLUE = cv::Scalar(255, 0, 0);
inline auto GREEN = cv::Scalar(0, 255, 0);
inline auto RED = cv::Scalar(0, 0, 255);
inline auto CYAN = cv::Scalar(255, 255, 0);
inline auto MAGENTA = cv::Scalar(255, 0, 255);
inline auto YELLOW = cv::Scalar(0, 255, 255);
inline auto WHITE = cv::Scalar(255, 255, 255);
inline auto BLACK = cv::Scalar(0, 0, 0);
inline auto ORANGE = cv::Scalar(0, 165, 255);
inline auto PURPLE = cv::Scalar(128, 0, 128);


// Helper struct for KeyPoint information, similar to Python's KeyPoint dataclass
struct KeyPointInfo {
    std::string name;
    cv::Scalar color;
};

// Helper struct for Skeleton information, similar to Python's Skeleton dataclass
struct SkeletonInfo {
    int srt_kpt_id;
    int dst_kpt_id;
    cv::Scalar color;
};

class YoloDrawer {

public:
    explicit YoloDrawer(float object_conf_threshold = 0.25f, float point_conf_threshold = 0.25f);

    // Draw bounding boxes for a vector of Yolo objects
    void drawBoundingBoxes(cv::Mat& frame, const std::vector<Yolo>& detections,
                    const std::map<int, std::string>& class_labels = {});

    // Draw bounding boxes, keypoints, and skeletons for a vector of YoloPose objects
    void drawPoses(cv::Mat& frame, const std::vector<YoloPose>& pose_detections,
                  const std::map<int, std::string>& class_labels = {});

private:
    float object_conf_threshold;
    float point_conf_threshold;

    // Default color palettes and schema definitions
    std::vector<cv::Scalar> bbox_colors;
    std::map<int, KeyPointInfo> kpt_color_map;
    std::vector<SkeletonInfo> skeleton_map;

    void _loadSchema();

    // Helper to get bounding box color based on class ID
    cv::Scalar _getBoundingBoxColorByCls(int class_id) const;

    // Helper to draw a single bounding box with label
    void _drawBoundingBoxAndLabel(cv::Mat& image,
                              const std::string& text,
                              const cv::Rect& bbox_rect,
                              const cv::Scalar& bbox_color,
                              const cv::Scalar& text_color = WHITE,
                              double font_scale = 0.5,
                              int thickness = 1);
};

#endif //YOLO_DRAWER_H
