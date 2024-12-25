//
// Created by orlando on 9/24/24.
//
#include <map>
#include <vector>
#include <string>

#include "yolo_def.h"
#include "yolo_visualization.h"


// 定义关键点结构
struct KeyPoint {
    std::string name;
    cv::Scalar color;
};

// 定义骨骼连接结构
struct Skeleton {
    int srt_kpt_id;
    int dst_kpt_id;
    cv::Scalar color;
};

// 定义关键点和骨骼映射
std::map<int, KeyPoint> kpt_color_map = {
    {0, {"Nose", cv::Scalar(0, 0, 255)}},          // 鼻尖
    {1, {"Right Eye", cv::Scalar(255, 0, 0)}},     // 右眼
    {2, {"Left Eye", cv::Scalar(255, 0, 0)}},      // 左眼
    {3, {"Right Ear", cv::Scalar(0, 255, 0)}},     // 右耳
    {4, {"Left Ear", cv::Scalar(0, 255, 0)}},      // 左耳
    {5, {"Right Shoulder", cv::Scalar(193, 182, 255)}},  // 右肩膀
    {6, {"Left Shoulder", cv::Scalar(193, 182, 255)}},   // 左肩膀
    {7, {"Right Elbow", cv::Scalar(16, 144, 247)}},      // 右肘
    {8, {"Left Elbow", cv::Scalar(16, 144, 247)}},       // 左肘
    {9, {"Right Wrist", cv::Scalar(1, 240, 255)}},       // 右手腕
    {10, {"Left Wrist", cv::Scalar(1, 240, 255)}},       // 左手腕
    {11, {"Right Hip", cv::Scalar(140, 47, 240)}},       // 右胯
    {12, {"Left Hip", cv::Scalar(140, 47, 240)}},        // 左胯
    {13, {"Right Knee", cv::Scalar(223, 155, 60)}},      // 右膝
    {14, {"Left Knee", cv::Scalar(223, 155, 60)}},       // 左膝
    {15, {"Right Ankle", cv::Scalar(139, 0, 0)}},        // 右脚踝
    {16, {"Left Ankle", cv::Scalar(139, 0, 0)}}          // 左脚踝
};

std::vector<Skeleton> skeleton_map = {
    {0, 1, cv::Scalar(0, 0, 255)},    // 鼻尖-右眼
    {0, 2, cv::Scalar(0, 0, 255)},    // 鼻尖-左眼
    {1, 3, cv::Scalar(0, 0, 255)},    // 右眼-右耳
    {2, 4, cv::Scalar(0, 0, 255)},    // 左眼-左耳
    {15, 13, cv::Scalar(0, 100, 255)},// 右脚踝-右膝
    {13, 11, cv::Scalar(0, 255, 0)},  // 右膝-右胯
    {16, 14, cv::Scalar(255, 0, 0)},  // 左脚踝-左膝
    {14, 12, cv::Scalar(0, 0, 255)},  // 左膝-左胯
    {11, 12, cv::Scalar(122, 160, 255)},  // 右胯-左胯
    {5, 11, cv::Scalar(139, 0, 139)}, // 右肩膀-右胯
    {6, 12, cv::Scalar(237, 149, 100)},   // 左肩膀-左胯
    {5, 6, cv::Scalar(152, 251, 152)},    // 右肩膀-左肩膀
    {5, 7, cv::Scalar(148, 0, 69)},       // 右肩膀-右肘
    {6, 8, cv::Scalar(0, 75, 255)},       // 左肩膀-左肘
    {7, 9, cv::Scalar(56, 230, 25)},      // 右肘-右手腕
    {8, 10, cv::Scalar(0, 240, 240)}      // 左肘-左手腕
};


// Define a list of colors for different classes (for simplicity, we assume 10 classes)
std::vector<cv::Scalar> bbox_colors = {
    cv::Scalar(255, 0, 0),    // Class 0: Blue
    cv::Scalar(0, 255, 0),    // Class 1: Green
    cv::Scalar(0, 0, 255),    // Class 2: Red
    cv::Scalar(255, 255, 0),  // Class 3: Cyan
    cv::Scalar(255, 0, 255),  // Class 4: Magenta
    cv::Scalar(0, 255, 255),  // Class 5: Yellow
    cv::Scalar(128, 0, 128),  // Class 6: Purple
    cv::Scalar(128, 128, 0),  // Class 7: Olive
    cv::Scalar(128, 128, 128),// Class 8: Gray
    cv::Scalar(0, 128, 255)   // Class 9: Orange
};


void drawSkeletons(cv::Mat& image, const std::vector<YoloPose>& poses, bool showPts, bool showNames) {
    for (const auto& pose : poses) {
        // 绘制关键点
        if (showPts) {
            for (size_t i = 0; i < pose.pts.size(); ++i) {
                const YoloPoint& pt = pose.pts[i];
                if (pt.conf > 0.2 && kpt_color_map.find(i) != kpt_color_map.end()) {
                    const KeyPoint& kp = kpt_color_map[i];
                    cv::circle(image, cv::Point(pt.x, pt.y), 3, kp.color, -1);

                    if (showNames) {
                        cv::putText(image, kp.name, cv::Point(pt.x, pt.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, kp.color, 1);
                    }
                }
            }
        }

        // 绘制骨骼
        for (const auto& bone : skeleton_map) {
            if (pose.pts[bone.srt_kpt_id].conf > 0.2 && pose.pts[bone.dst_kpt_id].conf > 0.2) {
                cv::line(image,
                    cv::Point(pose.pts[bone.srt_kpt_id].x, pose.pts[bone.srt_kpt_id].y),
                    cv::Point(pose.pts[bone.dst_kpt_id].x, pose.pts[bone.dst_kpt_id].y),
                    bone.color, 2);
            }
        }
    }
}



void drawBoxes(cv::Mat& image, const std::vector<YoloPose>& results) {

    for (const auto& result : results) {
        // Get bounding box coordinates and confidence
        auto lx = result.lx;
        auto ly = result.ly;
        auto rx = result.rx;
        auto ry = result.ry;
        float confidence = result.conf;

        // Select color based on class (using mod to cycle through colors)
        cv::Scalar box_color = bbox_colors[0];

        // Draw the bounding box
        cv::rectangle(image, cv::Point(lx, ly), cv::Point(rx, ry), box_color, 2);

        // Create the label text with confidence (formatted to two decimal places)
        std::string label = "person: " + cv::format("%.2f", confidence);

        // Get text size for background size
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        // Draw a filled rectangle as background for text (same color as the box)
        cv::rectangle(image, cv::Point(lx, ly - textSize.height - 5),
                      cv::Point(lx + textSize.width, ly),
                      box_color, cv::FILLED);

        // Put the label text on the image (white text)
        cv::putText(image, label, cv::Point(lx, ly - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}


void drawBoxes(cv::Mat& image, const std::vector<Yolo>& results, std::vector<std::string> labels) {

    for (const auto& result : results) {
        // Get bounding box coordinates and confidence
        auto lx = result.lx;
        auto ly = result.ly;
        auto rx = result.rx;
        auto ry = result.ry;
        auto cls = result.cls;
        float confidence = result.conf;

        // Select color based on class (using mod to cycle through colors)
        cv::Scalar box_color = bbox_colors[cls % bbox_colors.size()];

        // Draw the bounding box
        cv::rectangle(image, cv::Point(lx, ly), cv::Point(rx, ry), box_color, 2);

        // Create the label text with confidence (formatted to two decimal places)
        std::string label = labels[cls] + ": " + cv::format("%.2f", confidence);

        // Get text size for background size
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        // Draw a filled rectangle as background for text (same color as the box)
        cv::rectangle(image, cv::Point(lx, ly - textSize.height - 5),
                      cv::Point(lx + textSize.width, ly),
                      box_color, cv::FILLED);

        // Put the label text on the image (white text)
        cv::putText(image, label, cv::Point(lx, ly - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

