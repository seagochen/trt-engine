//
// Created by user on 6/17/25.
//

#include "trtengine/servlet/models/common/yolo_drawer.h"

// Constructor
YoloDrawer::YoloDrawer(float object_conf_threshold, float point_conf_threshold)
    : object_conf_threshold(object_conf_threshold),
      point_conf_threshold(point_conf_threshold) {
    _loadSchema();
}

// Private method to load default schema (similar to your Python defaults)
void YoloDrawer::_loadSchema() {
    // Default Bbox colors (matches your Python default_bbox_colors closely, order might differ slightly but covers common colors)
    bbox_colors = {
        RED, GREEN, BLUE, YELLOW, MAGENTA, CYAN,
        PURPLE, ORANGE, WHITE, BLACK,
        // Add more colors if needed
        cv::Scalar(128, 128, 128), // Gray
        cv::Scalar(255, 192, 203)  // Pink
    };

    // Default keypoint definitions (BGR colors)
    kpt_color_map = {
        {0, {"Nose", RED}},
        {1, {"Right Eye", GREEN}},
        {2, {"Left Eye", GREEN}},
        {3, {"Right Ear", BLUE}},
        {4, {"Left Ear", BLUE}},
        {5, {"Right Shoulder", YELLOW}},
        {6, {"Left Shoulder", YELLOW}},
        {7, {"Right Elbow", CYAN}},
        {8, {"Left Elbow", CYAN}},
        {9, {"Right Wrist", MAGENTA}},
        {10, {"Left Wrist", MAGENTA}},
        {11, {"Right Hip", ORANGE}},
        {12, {"Left Hip", ORANGE}},
        {13, {"Right Knee", PURPLE}},
        {14, {"Left Knee", PURPLE}},
        {15, {"Right Ankle", WHITE}},
        {16, {"Left Ankle", WHITE}}
    };

    // Default skeleton definitions (BGR colors)
    skeleton_map = {
        {0, 1, RED}, {0, 2, RED}, // Nose to Eyes
        {1, 3, GREEN}, {2, 4, GREEN}, // Eyes to Ears
        {5, 6, BLUE}, // Shoulders
        {5, 7, YELLOW}, {7, 9, YELLOW}, // Right arm
        {6, 8, CYAN}, {8, 10, CYAN}, // Left arm
        {11, 12, MAGENTA}, // Hips
        {5, 11, ORANGE}, {6, 12, ORANGE}, // Torso
        {11, 13, PURPLE}, {13, 15, PURPLE}, // Right leg
        {12, 14, WHITE}, {14, 16, WHITE}  // Left leg
    };
}

// Helper to get bounding box color based on class ID
cv::Scalar YoloDrawer::_getBoundingBoxColorByCls(int class_id) const {
    if (bbox_colors.empty()) {
        return WHITE; // Fallback
    }
    // Ensure class_id is non-negative before modulo
    int valid_class_id = std::max(0, class_id);
    return bbox_colors[valid_class_id % bbox_colors.size()];
}

// Helper to draw a single bounding box with label
void YoloDrawer::_drawBoundingBoxAndLabel(cv::Mat& image,
                                          const std::string& text,
                                          const cv::Rect& bbox_rect,
                                          const cv::Scalar& bbox_color,
                                          const cv::Scalar& text_color,
                                          double font_scale,
                                          int thickness) {
    // Draw bounding box
    cv::rectangle(image, bbox_rect, bbox_color, thickness);

    if (!text.empty()) {
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);

        // Position text background slightly above the bbox
        int text_bg_lx = bbox_rect.x;
        int text_bg_ly = std::max(0, bbox_rect.y - text_size.height - baseline - 5); // 5 for padding
        int text_bg_rx = std::min(image.cols, bbox_rect.x + text_size.width + 4); // 4 for padding
        int text_bg_ry = bbox_rect.y; // Aligned with the top of the bbox

        cv::rectangle(image, cv::Point(text_bg_lx, text_bg_ly), cv::Point(text_bg_rx, text_bg_ry),
                      bbox_color, cv::FILLED);

        // Draw text
        int text_x = bbox_rect.x + 2; // 2 for padding
        int text_y = std::max(text_size.height + 2, bbox_rect.y - baseline - 2); // 2 for padding
        cv::putText(image, text, cv::Point(text_x, text_y),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv::LINE_AA);
    }
}

// Draw bounding boxes for Yolo objects
void YoloDrawer::drawBoundingBoxes(cv::Mat& frame, const std::vector<Yolo>& detections,
                                const std::map<int, std::string>& class_labels) {
    for (const auto& obj : detections) {
        if (obj.conf < object_conf_threshold) {
            continue; // Skip if confidence is too low
        }

        cv::Rect bbox_rect(obj.lx, obj.ly, obj.rx - obj.lx, obj.ry - obj.ly);
        cv::Scalar bbox_color = _getBoundingBoxColorByCls(obj.cls);

        std::string label_text = "Conf: " + std::to_string(static_cast<int>(obj.conf * 100)) + "%";
        if (class_labels.count(obj.cls)) {
            label_text = class_labels.at(obj.cls) + " - " + label_text;
        } else {
            label_text = "Class " + std::to_string(obj.cls) + " - " + label_text;
        }

        _drawBoundingBoxAndLabel(frame, label_text, bbox_rect, bbox_color);
    }
}

// Draw bounding boxes, keypoints, and skeletons for YoloPose objects
void YoloDrawer::drawPoses(cv::Mat& frame, const std::vector<YoloPose>& pose_detections,
                              const std::map<int, std::string>& class_labels) {
    for (const auto& pose_obj : pose_detections) {
        if (pose_obj.conf < object_conf_threshold) {
            continue; // Skip if overall pose_extend confidence is too low
        }

        // 1. Draw Bounding Box (same as Yolo)
        cv::Rect bbox_rect(pose_obj.lx, pose_obj.ly, pose_obj.rx - pose_obj.lx, pose_obj.ry - pose_obj.ly);
        cv::Scalar bbox_color = _getBoundingBoxColorByCls(pose_obj.cls);

        std::string label_text = "Conf: " + std::to_string(static_cast<int>(pose_obj.conf * 100)) + "%";
        if (class_labels.count(pose_obj.cls)) {
            label_text = class_labels.at(pose_obj.cls) + " - " + label_text;
        } else {
            label_text = "Class " + std::to_string(pose_obj.cls) + " - " + label_text;
        }
        _drawBoundingBoxAndLabel(frame, label_text, bbox_rect, bbox_color);

        // 2. Draw Keypoints
        std::map<int, cv::Point> valid_kpts_coords; // Store valid keypoints for link drawing
        for (size_t i = 0; i < pose_obj.pts.size(); ++i) {
            const auto& kpt = pose_obj.pts[i];
            if (kpt.conf >= point_conf_threshold && kpt_color_map.count(i)) {
                cv::Point kpt_coord(kpt.x, kpt.y);
                cv::Scalar kpt_color = kpt_color_map.at(i).color;
                cv::circle(frame, kpt_coord, 3, kpt_color, cv::FILLED); // Radius 3, filled circle
                valid_kpts_coords[i] = kpt_coord;
            }
        }

        // 3. Draw Skeleton Links (Bones)
        for (const auto& bone : skeleton_map) {
            // Check if both start and end keypoints are valid (above threshold and exist)
            if (valid_kpts_coords.count(bone.srt_kpt_id) && valid_kpts_coords.count(bone.dst_kpt_id)) {
                cv::Point p1 = valid_kpts_coords.at(bone.srt_kpt_id);
                cv::Point p2 = valid_kpts_coords.at(bone.dst_kpt_id);
                cv::line(frame, p1, p2, bone.color, 2, cv::LINE_AA); // Thickness 2
            }
        }
    }
}