#ifndef C_POSE_DETECTION_H
#define C_POSE_DETECTION_H

#include "trtengine/c_apis/c_dstruct.h"

// 对于目前的Jetson平台来说
// YoloV8的模型，一次性推理8张图片最合适的，EfficientNet的模型，一次性推理32张图片是合适的
// 因此，对于目前采用硬编码，指定了YoloPose最多一次性处理8张图片，而EfficientNet为32张图片
// 以后如果还有时间，这个参数的配置情况将全部挪到配置文件中

#ifdef __cplusplus
extern "C" {
#endif
    
    /**
     * @brief 初始化姿态检测引擎
     *
     * @param yolo_engine_path yolo pose引擎路径 (C字符串)
     * @param efficient_engine_path efficientnet引擎路径 (C字符串)
     * @param max_items 最大检测物体数量 (per batch)
     * @param cls 置信度阈值
     * @param iou IOU阈值
     * @return bool 是否成功初始化 (1 for true, 0 for false)
     */
    bool init_pose_detection_pipeline(const char* yolo_engine_path, const char* efficient_engine_path,
                                      int max_items, float cls, float iou);

    /**
     * @brief 将一张图片添加到姿态检测管道中。
     * 这个函数会将图片数据添加到内部的处理队列中，供后续处理。
     * 注意：此函数不会立即执行检测，而是将图片数据存储起来，
     * 供 run_pose_detection_pipeline 函数调用时使用。
     * @param image_data_in_bgr 输入的BGR格式图片数据指针
     * @param width 图片宽度
     * @param height 图片高度
     */
    void add_image_to_pose_detection_pipeline(const unsigned char* image_data_in_bgr, int width, int height);

    /**
     * @brief 对单张图片进行姿态检测和信息扩充。
     * 这是一个阻塞调用，内部会执行预处理、推理和后处理。
     *
     * @param out_results 指向 C_InferenceResult 数组的指针，用于存储检测结果。另外注意，调用者需要手工释放内存。
     * @param out_num_results 指向整数的指针，用于存储检测结果数量, 如果输入10张图片，一定会返回10个结果，
     * @return bool 是否成功执行整个流程 (1 for true, 0 for false)
     */
    bool run_pose_detection_pipeline(void **out_results, int *out_num_results);

    /**
     * @brief 销毁所有已加载的模型。
     */
    void deinit_pose_detection_pipeline();

    /**
     * @brief 释放 C_InferenceResult 结构体的内存。
     * 注意：调用此函数后，result 指向的内存将被释放，不能再访问。
     *
     * @param result 指向 C_InferenceResult 的指针
     */
    void release_inference_result(void* result);

#ifdef __cplusplus
};
#endif

#endif // C_POSE_DETECTION_H