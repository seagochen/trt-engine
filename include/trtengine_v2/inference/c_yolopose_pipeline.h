/**
 * @file c_yolopose_pipeline.h
 * @brief V2 版本的 YOLO Pose 推理管线（纯 C 实现）
 *
 * 这是专为 v2 架构设计的 YOLO Pose 检测管线的完全重写版本，使用纯 C 实现。
 * 它消除了 C++ 依赖，提供了可以从多种语言调用的干净 C API。
 *
 * 与 v1 的主要区别：
 * - 纯 C 实现（接口无 C++/OpenCV 依赖）
 * - 使用 v2/core 中的 TrtEngineMultiTs
 * - 集成 c_common_ops.h 的 NMS 函数
 * - 简化的内存管理
 * - 更好的跨平台兼容性
 */

#ifndef TRTENGINE_V2_C_YOLOPOSE_PIPELINE_H
#define TRTENGINE_V2_C_YOLOPOSE_PIPELINE_H

#include "trtengine_v2/model/c_structures.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
//                         推理管线配置
// ============================================================================

/**
 * @brief YOLO Pose 推理管线配置
 */
typedef struct {
    // 模型配置
    const char* engine_path;        ///< TensorRT 引擎文件路径
    int input_width;                ///< 模型输入宽度（例如 640）
    int input_height;               ///< 模型输入高度（例如 640）
    int max_batch_size;             ///< 最大批处理大小

    // 检测阈值
    float conf_threshold;           ///< 置信度阈值（0.0-1.0，通常 0.25）
    float iou_threshold;            ///< NMS 的 IoU 阈值（0.0-1.0，通常 0.45）

    // 模型参数
    int num_keypoints;              ///< 关键点数量（例如 COCO 的 17 个）
    int num_classes;                ///< 检测类别数量
} C_YoloPosePipelineConfig;

/**
 * @brief YOLO Pose 推理管线上下文的不透明句柄
 */
typedef struct C_YoloPosePipelineContext C_YoloPosePipelineContext;

// ============================================================================
//                         输入图像结构
// ============================================================================

/**
 * @brief 单张图像输入
 */
typedef struct {
    const unsigned char* data;      ///< RGB 图像数据（HWC 格式）
    int width;                      ///< 图像宽度
    int height;                     ///< 图像高度
    int channels;                   ///< 通道数（RGB 必须为 3）
} C_ImageInput;

/**
 * @brief 批量图像输入
 */
typedef struct {
    C_ImageInput* images;           ///< 输入图像数组
    size_t count;                   ///< 批次中的图像数量
} C_ImageBatch;

// ============================================================================
//                         输出结果结构
// ============================================================================

/**
 * @brief 单张图像的检测结果
 */
typedef struct {
    int image_index;                ///< 图像在输入批次中的索引
    C_YoloPose* poses;              ///< 检测到的姿态数组
    size_t num_poses;               ///< 检测到的姿态数量
} C_YoloPoseImageResult;

/**
 * @brief 批量图像的检测结果
 */
typedef struct {
    C_YoloPoseImageResult* results; ///< 每张图像的结果数组
    size_t num_images;              ///< 批次中的图像数量
} C_YoloPoseBatchResult;

// ============================================================================
//                         管线生命周期函数
// ============================================================================

/**
 * @brief 创建 YOLO Pose 推理管线上下文
 *
 * 该函数初始化 TensorRT 引擎并准备推理所需的所有资源。
 *
 * @param config 管线配置
 * @return 创建的上下文指针，失败时返回 NULL
 *
 * @note 返回的上下文必须使用 c_yolopose_pipeline_destroy() 释放
 */
C_YoloPosePipelineContext* c_yolopose_pipeline_create(
    const C_YoloPosePipelineConfig* config
);

/**
 * @brief 销毁 YOLO Pose 推理管线上下文
 *
 * 该函数释放与管线相关的所有资源，包括 TensorRT 引擎和内部缓冲区。
 *
 * @param context 要销毁的管线上下文
 */
void c_yolopose_pipeline_destroy(C_YoloPosePipelineContext* context);

// ============================================================================
//                         推理函数
// ============================================================================

/**
 * @brief 对单张图像进行推理
 *
 * @param context 管线上下文
 * @param image 输入图像
 * @param result 输出检测结果（调用者负责释放）
 * @return 成功返回 true，失败返回 false
 *
 * @note 调用者必须使用 free() 释放 result->poses
 */
bool c_yolopose_infer_single(
    C_YoloPosePipelineContext* context,
    const C_ImageInput* image,
    C_YoloPoseImageResult* result
);

/**
 * @brief 对批量图像进行推理
 *
 * 该函数在单个批次中处理多张图像，比逐个处理更高效。
 *
 * @param context 管线上下文
 * @param batch 输入图像批次
 * @param result 输出检测结果（调用者负责释放）
 * @return 成功返回 true，失败返回 false
 *
 * @note 调用者必须使用 c_yolopose_batch_result_free() 释放结果
 */
bool c_yolopose_infer_batch(
    C_YoloPosePipelineContext* context,
    const C_ImageBatch* batch,
    C_YoloPoseBatchResult* result
);

// ============================================================================
//                         内存管理函数
// ============================================================================

/**
 * @brief 释放单张图像结果分配的内存
 *
 * @param result 要释放的结果
 */
void c_yolopose_image_result_free(C_YoloPoseImageResult* result);

/**
 * @brief 释放批量结果分配的内存
 *
 * 该函数释放与批量结果相关的所有内存，包括所有单张图像的结果和检测到的姿态。
 *
 * @param result 要释放的批量结果
 */
void c_yolopose_batch_result_free(C_YoloPoseBatchResult* result);

// ============================================================================
//                         实用工具函数
// ============================================================================

/**
 * @brief 获取默认管线配置
 *
 * 该函数返回具有合理默认值的配置。
 * 用户可以在创建管线之前修改返回的配置。
 *
 * @return 默认配置
 */
C_YoloPosePipelineConfig c_yolopose_pipeline_get_default_config(void);

/**
 * @brief 验证管线配置
 *
 * @param config 要验证的配置
 * @return 配置有效返回 true，否则返回 false
 */
bool c_yolopose_pipeline_validate_config(const C_YoloPosePipelineConfig* config);

/**
 * @brief 获取上次操作的错误消息
 *
 * @param context 管线上下文
 * @return 错误消息字符串，无错误时返回 NULL
 *
 * @note 返回的字符串由上下文拥有，不应被释放
 */
const char* c_yolopose_pipeline_get_last_error(C_YoloPosePipelineContext* context);

#ifdef __cplusplus
}
#endif

#endif // TRTENGINE_V2_C_YOLOPOSE_PIPELINE_H
