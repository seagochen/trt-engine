/**
 * @file c_yolopose_pipeline.h
 * @brief YOLOv8-Pose inference pipeline (Pure C implementation)
 *
 * This is a complete rewrite of the YOLO Pose detection pipeline designed for
 * the v2 architecture, using pure C implementation. It eliminates C++ dependencies
 * and provides a clean C API that can be called from multiple languages.
 *
 * Key features:
 * - Pure C implementation (no C++/OpenCV dependencies in interface)
 * - Uses TrtEngineMultiTs from v2/core
 * - Integrates with common NMS functions
 * - Simplified memory management
 * - Better cross-platform compatibility
 *
 * @author TrtEngineToolkits
 * @date 2025-11-10
 */

#ifndef TRTENGINE_V2_PIPELINES_YOLOPOSE_C_YOLOPOSE_PIPELINE_H
#define TRTENGINE_V2_PIPELINES_YOLOPOSE_C_YOLOPOSE_PIPELINE_H

#include "trtengine_v2/pipelines/yolopose/c_yolopose_structures.h"
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
//                         输出结果结构
// ============================================================================
// Note: C_ImageInput and C_ImageBatch are defined in trtengine_v2/common/c_structures.h

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

#endif // TRTENGINE_V2_PIPELINES_YOLOPOSE_C_YOLOPOSE_PIPELINE_H
