//
// Created by user on 3/21/25.
//

#ifndef INFER_NMS_HPP
#define INFER_NMS_HPP


#include <algorithm>
#include <vector>

// 快速计算IoU
template<typename T>
float iou(const T& a, const T& b) {

    // 分别计算两个矩形的交集区域
    const int inter_left = std::max(a.lx, b.lx);
    const int inter_top = std::max(a.ly, b.ly);
    const int inter_right = std::min(a.rx, b.rx);
    const int inter_bottom = std::min(a.ry, b.ry);

    // 计算交集区域的面积
    const int inter_area = std::max(0, inter_right - inter_left + 1) * std::max(0, inter_bottom - inter_top + 1);

    // 如果交集区域的面积为0，则返回0
    if (inter_area <= 0) {
        return 0.0f;
    }

    // 计算两个矩形的面积
    const int area_a = (a.rx - a.lx + 1) * (a.ry - a.ly + 1);
    const int area_b = (b.rx - b.lx + 1) * (b.ry - b.ly + 1);

    // 计算并集区域的面积
    const float union_area = static_cast<float>(area_a + area_b - inter_area) + 1e-6f;
    
    // 返回交并比
    return static_cast<float>(inter_area) / union_area;
}


// 检查类是否具有成员变量 cls
template <typename T, typename = void>
struct HasClassMember : std::false_type {};

template <typename T>
struct HasClassMember<T, std::void_t<decltype(std::declval<T>().cls)>> : std::true_type {};


// NMS 核心函数
template<typename T>
std::vector<T> nms(const std::vector<T>& boxes, float iouThreshold) {
    // 1. 根据置信度排序
    std::vector<T> sortedBoxes = boxes;
    std::sort(sortedBoxes.begin(), sortedBoxes.end(), [](const T& a, const T& b) {
        return a.conf > b.conf;
    });

    std::vector<T> result;

    // 2. 遍历候选框
    std::vector<bool> suppressed(sortedBoxes.size(), false); // 创建一个长度为 sortedBoxes.size() 的布尔向量，初始值为 false

    // 3. 遍历已排序的候选框，对每个框进行非极大值抑制
    for (size_t i = 0; i < sortedBoxes.size(); ++i) {

        // 如果当前框已经被抑制，则跳过
        if (suppressed[i]) continue;

        // 将当前框添加到结果中
        result.push_back(sortedBoxes[i]);

        for (size_t j = i + 1; j < sortedBoxes.size(); ++j) {

            // 如果当前框已经被抑制，则跳过
            if (suppressed[j]) continue;

            // 检查当前框和下一个框的 IoU
            // 如果是同一类，则计算 IoU
            // 否则直接跳过
            // 检查是否有类成员
            // 如果没有类成员，则默认所有框都是同一类
            // 如果有类成员，则检查类是否相同
            // 这里假设 T 有成员变量 cls，表示类别
            // 如果没有类成员，则默认所有框都是同一类
            bool same_class = true;
            if constexpr (HasClassMember<T>::value) {
                same_class = (sortedBoxes[i].cls == sortedBoxes[j].cls);
            }

            // 如果是同一类且 且当前框和下一个框的 IoU 大于等于阈值，则抑制下一个框
            // 否则直接跳过
            if (same_class && iou<T>(sortedBoxes[i], sortedBoxes[j]) >= iouThreshold) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

#endif //INFER_NMS_HPP
