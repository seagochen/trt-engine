//
// Created by user on 6/18/25.
//

#ifndef COMBINEDPROJECT_YOLO_POSTPROCESS_H
#define COMBINEDPROJECT_YOLO_POSTPROCESS_H

#include <vector>

/**
 * @brief Yolo系列模型处理完毕后，需要对数据进行特殊处理，提取出边界框、类别等信息，执行NMS等操作。
 * @param ptr_device 指向CUDA设备上输出数据的指针，通常是float类型的数组。
 * @param output 输出的结果向量，包含处理后的边界框、类别等信息。
 * @param features 每个样本的特征数量，通常是模型输出的维度。
 * @param samples 样本总数，表示有多少个边界框需要处理。
 * @param cls 分类阈值，通常用于过滤低置信度的检测结果。
 * @param maximum 最大边界框数量，通常是8400或其他预设值，用于限制输出的边界框数量。
 * @param use_pose 是否使用姿态估计，如果为true，则处理结果将包含关键点信息。
 * @return
 */
int inferPostProcForYolo(
    const float* ptr_device,
    std::vector<float>& output,
    int features,
    int samples,
    float cls,
    int maximum = 8400,
    bool use_pose = false
);

#endif // COMBINEDPROJECT_YOLO_POSTPROCESS_H
