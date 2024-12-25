#ifndef NMS_H
#define NMS_H

#include <vector>
#include <algorithm>
#include <cmath>

#include "common/yolo/yolo_def.h"


// 快速计算IoU
template<typename T>
float IoU(const T& a, const T& b) {
    int interLeft = std::max(a.lx, b.lx);
    int interTop = std::max(a.ly, b.ly);
    int interRight = std::min(a.rx, b.rx);
    int interBottom = std::min(a.ry, b.ry);

    int interArea = std::max(0, interRight - interLeft + 1) * std::max(0, interBottom - interTop + 1);

    int areaA = (a.rx - a.lx + 1) * (a.ry - a.ly + 1);
    int areaB = (b.rx - b.lx + 1) * (b.ry - b.ly + 1);

    float unionArea = static_cast<float>(areaA + areaB - interArea) + 1e-6f;
    return unionArea > 0 ? static_cast<float>(interArea) / unionArea : 0.0f;
}


// NMS 核心函数
std::vector<Yolo> NMS(const std::vector<Yolo>& boxes, float iouThreshold) {
    // 1. 根据置信度排序
    std::vector<Yolo> sortedBoxes = boxes;
    std::sort(sortedBoxes.begin(), sortedBoxes.end(), [](const Yolo& a, const Yolo& b) {
        return a.conf > b.conf;
    });

    std::vector<Yolo> result;

    // 2. 遍历候选框
    std::vector<bool> suppressed(sortedBoxes.size(), false);
    for (size_t i = 0; i < sortedBoxes.size(); ++i) {
        if (suppressed[i]) continue;

        result.push_back(sortedBoxes[i]);

        for (size_t j = i + 1; j < sortedBoxes.size(); ++j) {
            if (suppressed[j]) continue;

            if (sortedBoxes[i].cls == sortedBoxes[j].cls && IoU<Yolo>(sortedBoxes[i], sortedBoxes[j]) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

// 对于 YoloPose 的 NMS
std::vector<YoloPose> NMS(const std::vector<YoloPose>& poses, float iouThreshold) {
    // 1. 根据置信度排序
    std::vector<YoloPose> sortedPoses = poses;
    std::sort(sortedPoses.begin(), sortedPoses.end(), [](const YoloPose& a, const YoloPose& b) {
        return a.conf > b.conf;
    });

    std::vector<YoloPose> result;

    // 2. 遍历候选框
    std::vector<bool> suppressed(sortedPoses.size(), false);
    for (size_t i = 0; i < sortedPoses.size(); ++i) {
        if (suppressed[i]) continue;

        result.push_back(sortedPoses[i]);

        for (size_t j = i + 1; j < sortedPoses.size(); ++j) {
            if (suppressed[j]) continue;

            if (IoU<YoloPose>(sortedPoses[i], sortedPoses[j]) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}


#endif