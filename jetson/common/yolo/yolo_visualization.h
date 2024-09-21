//
// Created by orlando on 9/24/24.
//

#ifndef YOLO_POSE_H
#define YOLO_POSE_H

#include <opencv2/opencv.hpp>

#include "yolo_def.h"

// 使用这些映射来绘制关键点和骨骼
void drawSkeletons(cv::Mat& image, const std::vector<YoloPose>& poses, bool showPts=true, bool showNames=true);

// Draw the bounding box - For pose detection
void drawBoxes(cv::Mat& image, const std::vector<YoloPose>& results);

// Draw the bounding box - For object detection
void drawBoxes(cv::Mat& image, const std::vector<Yolo>& results, std::vector<std::string> labels);

#endif //YOLO_POSE_H
