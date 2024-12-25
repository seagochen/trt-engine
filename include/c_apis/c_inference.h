//
// Created by vipuser on 25-1-8.
//

#ifndef C_INFERENCE_H
#define C_INFERENCE_H

#include "include/c_vit_infer.h"
#include "include/c_yolo_infer.h"

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief 初始化模型资源
     * @param model_path
     * @return
     */
    void* c_yolo_init(const char *model_path);

    /**
     * @brief 初始化模型资源
     * @param model_path
     * @return
     */
    void* c_pose_init(const char *model_path);

    /**
     * @brief 初始化模型资源
     * @param model_path
     * @return
     */
    void* c_vit_init(const char *model_path);
    

    /**
    * @brief 释放模型资源
    * @return 是否释放成功
    */
    bool c_release_model();

    /**
    * @brief 添加图片至模型中
    * @param n_index 索引
    * @param cstr 图片数据指针
    * @param n_channels 通道数
    * @param n_width 宽度
    * @param n_height 高度
    * @return 是否添加成功
    */
    bool c_add_image(int n_index, unsigned char* cstr, int n_channels, int n_width, int n_height);

    /**
    * @brief 执行推理
    * @return 是否推理成功
    */
    bool c_do_inference();


#ifdef __cplusplus
};
#endif

#endif //C_INFERENCE_H
