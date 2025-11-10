#ifndef COMBINEDPROJECT_TRTENGINE_V2_C_COMMON_OPS_H
#define COMBINEDPROJECT_TRTENGINE_V2_C_COMMON_OPS_H

#include "c_structures.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算两个检测框的 IoU (Intersection over Union)
 *
 * @param a 第一个检测框
 * @param b 第二个检测框
 * @return float IoU 值 (0.0 ~ 1.0)
 */
float c_iou_detect(const C_YoloDetect* a, const C_YoloDetect* b);

/**
 * @brief 计算两个姿态检测框的 IoU
 *
 * @param a 第一个姿态检测
 * @param b 第二个姿态检测
 * @return float IoU 值 (0.0 ~ 1.0)
 */
float c_iou_pose(const C_YoloPose* a, const C_YoloPose* b);

/**
 * @brief 对检测框进行非极大值抑制 (NMS)
 *
 * 该函数会根据置信度对输入的检测框进行排序，然后使用 NMS 算法
 * 过滤重叠的框，保留置信度最高的框。
 *
 * @param boxes 输入的检测框数组
 * @param count 输入框的数量
 * @param iou_threshold IoU 阈值，超过该值的重叠框会被抑制 (通常 0.45-0.7)
 * @param result 输出的检测框数组 (由调用者分配，大小至少为 count)
 * @param result_count 输出框的实际数量 (通过指针返回)
 *
 * @note 输入数组不会被修改
 * @note result 数组必须由调用者预先分配足够的空间 (建议与 count 相同)
 * @note 函数会自动处理不同类别的框 (只在同类别间进行 NMS)
 */
void c_nms_detect(
    const C_YoloDetect* boxes,
    size_t count,
    float iou_threshold,
    C_YoloDetect* result,
    size_t* result_count
);

/**
 * @brief 对姿态检测进行非极大值抑制 (NMS)
 *
 * @param poses 输入的姿态检测数组
 * @param count 输入姿态的数量
 * @param iou_threshold IoU 阈值
 * @param result 输出的姿态检测数组 (由调用者分配)
 * @param result_count 输出姿态的实际数量 (通过指针返回)
 */
void c_nms_pose(
    const C_YoloPose* poses,
    size_t count,
    float iou_threshold,
    C_YoloPose* result,
    size_t* result_count
);

#ifdef __cplusplus
}
#endif

#endif // COMBINEDPROJECT_TRTENGINE_V2_C_COMMON_OPS_H 