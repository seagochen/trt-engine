#ifndef YOLO_COORDS_H
#define YOLO_COORDS_H

#include <vector>
#include "serverlet/models/common/yolo_dstruct.h" // For Yolo and Yolo

/**
 * @brief 将数据的坐标格式(cx,cy,w,h)转换为(lx,ly,rx,ry)，解析后并返回为Yolo对象向量
 * @param input 输入的浮点数数组
 * @param output 输出的Yolo对象向量
 * @param features 每个样本的特征数量
 * @param samples 样本总数
 * @param target_width 目标图像宽度
 * @param target_height 目标图像高度
 */
void cvtXYWHCoordsToYolo(const std::vector<float>& input, std::vector<Yolo>& output,
                            int features, int samples, float target_width = 1.0, float target_height = 1.0);


/**
 * @brief 将数据的坐标格式(cx,cy,w,h)转换为(lx,ly,rx,ry)，解析后并返回为YoloPose对象向量
 * @param input 输入的浮点数数组
 * @param output 输出的YoloPose对象向量
 * @param features 每个样本的特征数量
 * @param samples 样本总数
 * @param target_width 目标图像宽度
 * @param target_height 目标图像高度
 */
void cvtXYWHCoordsToYoloPose(const std::vector<float>& input, std::vector<YoloPose>& output,
                            int features, int samples, float target_width = 1.0, float target_height = 1.0);


#endif //YOLO_COORDS_H