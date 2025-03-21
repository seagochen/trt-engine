//
// Created by user on 3/21/25.
//

#ifndef COMBINEDPROJECT_C_YOLO_APIS_H
#define COMBINEDPROJECT_C_YOLO_APIS_H

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief 初始化模型资源
     * @param model_path 模型路径
     * @param b_use_pose 是否使用姿态估计
    */
    void c_yolo_init(const char *model_path, bool b_use_pose=false);

    /**
     * @brief 释放模型资源
     * @return 是否释放成功
    */
    bool c_yolo_release();

    /**
     * @brief 添加图片至模型中
     * @param n_index 索引
     * @param cstr 图片数据指针
     * @param n_channels 通道数
     * @param n_width 宽度
     * @param n_height 高度
     * @return 是否添加成功
    */
    bool c_yolo_add_image(int n_index, unsigned char* cstr, int n_channels, int n_width, int n_height);

    /**
     * @brief 执行推理
     * @return 是否推理成功
    */
    bool c_yolo_inference();

    /**
     * @brief 获取可用结果数量
     * @param n_index 索引
     * @param f_clsThreshold 置信度阈值
     * @param f_nmsThreshold nms阈值
     * @return 返回 n_index 索引的可用结果数量
     */
    int c_yolo_available_results(int n_index, float f_clsThreshold, float f_nmsThreshold);

    /**
     * @brief 获取yolo8模型的推理结果
     * @param n_itemIndex
     * @param n_size
     * @return
     */
    float* c_yolo_get_result(int n_itemIndex, int& n_size);

#ifdef __cplusplus
};
#endif

#endif //COMBINEDPROJECT_C_YOLO_APIS_H
