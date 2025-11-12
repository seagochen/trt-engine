/**
 * @file c_yolopose_pipeline.c
 * @brief Pure C implementation of YOLO Pose inference pipeline
 *
 * This file implements a complete YOLO Pose detection pipeline in pure C,
 * integrating with TensorRT engine and providing image preprocessing,
 * inference, and postprocessing functionality.
 *
 * Author: TrtEngineToolkits
 * Date: 2025-11-10
 */

#include "trtengine_v2/pipelines/yolopose/c_yolopose_pipeline.h"
#include "trtengine_v2/pipelines/yolopose/c_yolopose_structures.h"
#include "trtengine_v2/core/trt_engine_multi.h"
#include "trtengine_v2/utils/logger.h"

// SimpleCudaToolkits CUDA核函数
#include "simple_cuda_toolkits/matrix/matrix.h"
#include "simple_cuda_toolkits/tsutils/maxmin.h"
#include "simple_cuda_toolkits/tsutils/filter.h"
#include "simple_cuda_toolkits/tsutils/sort.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

// ============================================================================
//                         Internal Structures
// ============================================================================

/**
 * @brief Internal context structure for YOLO Pose pipeline
 */
struct C_YoloPosePipelineContext {
    // Configuration
    C_YoloPosePipelineConfig config;

    // TensorRT engine
    TrtEngineMultiTs* trt_engine;

    // Input/Output tensors (CUDA device memory)
    Tensor<float>* input_tensor;
    Tensor<float>* output_tensor;

    // Host memory buffers for data transfer
    float* host_input_buffer;
    float* host_output_buffer;

    // Tensor dimensions
    int output_features;     // Number of features per detection (e.g., 56)
    int output_samples;      // Number of detection samples (e.g., 8400)

    // Error message
    char error_msg[256];
};

// ============================================================================
//                         Constants and Macros
// ============================================================================

#define YOLO_POSE_NUM_KEYPOINTS 17
#define YOLO_POSE_BOX_FEATURES  4   // x, y, w, h
#define YOLO_POSE_CONF_FEATURES 1   // objectness
#define YOLO_POSE_CLS_FEATURES  1   // class (usually just "person")
#define YOLO_POSE_KPT_FEATURES  (YOLO_POSE_NUM_KEYPOINTS * 3)  // x, y, conf for each

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// ============================================================================
//                         Image Preprocessing
// ============================================================================

/**
 * @brief 对图像进行 letterbox 预处理，并将其转换为 CHW 格式的 float 缓冲区。
 *
 * 此函数实现以下操作：
 * 1. 使用 letterbox 方法计算缩放比例和填充大小，以保持原始图像的宽高比。
 * 2. 将整个输出缓冲区初始化为灰色 (114) 并归一化。
 * 3. 使用双线性插值将输入图像缩放到目标尺寸。
 * 4. 将缩放后的图像（归一化为 [0, 1]）复制到输出缓冲区的中心位置。
 * 5. 最终的输出格式为 CHW (Channel-Height-Width)，适用于 TensorRT 等推理引擎。
 *
 * @param input_data     [in] 指向原始输入图像数据（HWC 格式, 8-bit unsigned char）的指针。
 * @param input_width    [in] 输入图像的宽度。
 * @param input_height   [in] 输入图像的高度。
 * @param input_channels [in] 输入图像的通道数（必须为 3）。
 * @param output_buffer  [out] 指向目标 float 缓冲区（CHW 格式）的指针。
 * @param output_width   [in] 输出缓冲区的宽度（模型输入宽度）。
 * @param output_height  [in] 输出缓冲区的高度（模型输入高度）。
 * @param scale_x        [out] 用于坐标映射的 X 轴缩放比例（在此实现中等于 Y 轴）。
 * @param scale_y        [out] 用于坐标映射的 Y 轴缩放比例（在此实现中等于 X 轴）。
 * @param pad_x          [out] X 轴（宽度）上的填充大小（单边，通常是左侧）。
 * @param pad_y          [out] Y 轴（高度）上的填充大小（单边，通常是顶部）。
 * @return true 如果预处理成功， false 如果输入参数无效。
 */
static bool preprocess_image(
    const unsigned char* input_data,
    int input_width,
    int input_height,
    int input_channels,
    float* output_buffer,
    int output_width,
    int output_height,
    float* scale_x,
    float* scale_y,
    int* pad_x,
    int* pad_y
) {
    // 1. 检查输入参数是否合法
    // 确保输入输出指针有效，并且输入图像是 3 通道 (RGB)
    if (!input_data || !output_buffer || input_channels != 3) {
        return false;
    }

    // 2. 计算 letterbox 缩放参数
    // Letterbox：保持宽高比缩放，空白处填充
    // 计算缩放比例，取 宽缩放比 和 高缩放比 中的较小值，以确保整个图像都能放入目标尺寸
    float scale = MIN((float)output_width / input_width,
                      (float)output_height / input_height);

    // 计算缩放后的实际宽高
    int scaled_width = (int)(input_width * scale);
    int scaled_height = (int)(input_height * scale);

    // 计算居中所需的填充（padding）
    // (目标宽度 - 缩放后宽度) / 2
    *pad_x = (output_width - scaled_width) / 2;
    // (目标高度 - 缩放后高度) / 2
    *pad_y = (output_height - scaled_height) / 2;

    // 将计算出的缩放和填充值通过指针返回
    // 这些值在后处理（将检测框映射回原始图像）时非常重要
    *scale_x = scale;
    *scale_y = scale; // 在 letterbox 中，x 和 y 缩放比例相同

    // 3. 初始化输出缓冲区
    // 将整个输出缓冲区填充为灰色 (114)，并归一化到 [0, 1]
    // 114 是 YOLOX 等模型常用的填充值
    for (int i = 0; i < output_width * output_height * 3; i++) {
        output_buffer[i] = 114.0f / 255.0f;
    }

    // 4. 执行双线性插值缩放和 CHW 格式转换
    // 目标格式: CHW (Channel-Height-Width)
    // 外层循环遍历通道 (C)
    for (int c = 0; c < 3; c++) {
        // 中层循环遍历缩放后图像的高度 (H)
        for (int y = 0; y < scaled_height; y++) {
            // 内层循环遍历缩放后图像的宽度 (W)
            for (int x = 0; x < scaled_width; x++) {
                
                // --- 双线性插值计算 ---

                // 将当前目标坐标 (x, y) 映射回原始输入图像的坐标 (src_x, src_y)
                // 注意：这里 (x, y) 是 *缩放后* 图像内的坐标，不是 *最终输出* 缓冲区的坐标
                float src_x = x / scale;
                float src_y = y / scale;

                // 找到用于插值的四个最近邻像素的左上角坐标 (x0, y0)
                int x0 = (int)src_x;
                int y0 = (int)src_y;

                // 找到右下角坐标 (x1, y1)，并处理边界情况，防止越界
                int x1 = MIN(x0 + 1, input_width - 1);
                int y1 = MIN(y0 + 1, input_height - 1);

                // 计算 (src_x, src_y) 相对于 (x0, y0) 的小数部分（偏移量）
                // 这将用作插值的权重
                float dx = src_x - x0;
                float dy = src_y - y0;

                // 获取四个邻近像素在 *输入* 缓冲区中的索引
                // 输入格式是 HWC: (行 * 宽度 + 列) * 通道数 + 通道索引
                int src_idx_00 = (y0 * input_width + x0) * input_channels + c; // 左上
                int src_idx_01 = (y0 * input_width + x1) * input_channels + c; // 右上
                int src_idx_10 = (y1 * input_width + x0) * input_channels + c; // 左下
                int src_idx_11 = (y1 * input_width + x1) * input_channels + c; // 右下

                // 获取这四个像素的值，并归一化到 [0, 1]
                float val00 = input_data[src_idx_00] / 255.0f;
                float val01 = input_data[src_idx_01] / 255.0f;
                float val10 = input_data[src_idx_10] / 255.0f;
                float val11 = input_data[src_idx_11] / 255.0f;

                // 执行双线性插值：
                // f(x,y) ≈ (1-dx)(1-dy)f(0,0) + dx(1-dy)f(1,0) + (1-dx)dyf(0,1) + dxdyf(1,1)
                float val = val00 * (1 - dx) * (1 - dy) +
                            val01 * dx * (1 - dy) +
                            val10 * (1 - dx) * dy +
                            val11 * dx * dy;

                // --- 写入输出缓冲区 ---

                // 计算插值结果在 *输出* 缓冲区中的目标位置 (dst_x, dst_y)
                // 需要加上之前计算的填充 (padding)
                int dst_y = y + *pad_y;
                int dst_x = x + *pad_x;

                // 计算目标像素在 *输出* 缓冲区中的索引
                // 输出格式是 CHW: (通道索引 * 高度 * 宽度) + (行 * 宽度) + 列
                int dst_idx = c * output_height * output_width +
                              dst_y * output_width + dst_x;

                // 将计算出的插值（已归一化）写入输出缓冲区
                output_buffer[dst_idx] = val;
            }
        }
    }

    // 所有操作完成，返回成功
    return true;
}


// ============================================================================
//                         Output Postprocessing (CUDA-accelerated)
// ============================================================================

/**
 * @brief 使用 CUDA 核函数对 YOLO-Pose 模型的原始输出进行高效后处理
 *
 * 此函数实现了与 V1.0 版本相同的 CUDA 加速后处理流程：
 * 1. 转置：将 [56, 8400] 转换为 [8400, 56]
 * 2. 过滤：根据置信度阈值过滤结果
 * 3. 排序：按置信度降序排序
 * 4. 限制：限制最大结果数量
 *
 * 对于 YOLOv8-Pose，我们跳过 argmax 步骤，因为它是单类别检测（person）
 *
 * @param ptr_device      [in] 设备内存中的模型原始输出指针 [56, 8400]
 * @param output          [out] 主机内存的输出缓冲区，存储处理后的结果
 * @param features        [in] 特征数量 (56)
 * @param samples         [in] 样本数量 (8400)
 * @param conf_threshold  [in] 置信度阈值
 * @param maximum         [in] 最大返回结果数量
 * @return int            实际有效结果数量，失败返回 -1
 */
static int kernel_decode_for_yolopose(
    const float* ptr_device,
    std::vector<float>& output,
    const int features,
    const int samples,
    const float conf_threshold,
    const int maximum
) {
    // 1. 分配 CUDA 设备上的临时张量
    float* ptr_device_temp0 = nullptr;
    float* ptr_device_temp1 = nullptr;
    size_t total_size = (size_t)features * samples * sizeof(float);

    cudaError_t err0 = cudaMalloc(&ptr_device_temp0, total_size);
    if (err0 != cudaSuccess) {
        LOG_ERROR("YoloPosePipeline",
            std::string("Failed to allocate CUDA memory for temp tensor 0: ") +
            cudaGetErrorString(err0));
        return -1;
    }

    cudaError_t err1 = cudaMalloc(&ptr_device_temp1, total_size);
    if (err1 != cudaSuccess) {
        cudaFree(ptr_device_temp0);
        LOG_ERROR("YoloPosePipeline",
            std::string("Failed to allocate CUDA memory for temp tensor 1: ") +
            cudaGetErrorString(err1));
        return -1;
    }

    // 2. 复制原始输出到临时缓冲区
    cudaMemcpy(ptr_device_temp0, ptr_device, total_size, cudaMemcpyDeviceToDevice);

    // 3. 转置操作：将 [features=56, samples=8400] 转换为 [samples=8400, features=56]
    //    这样每一行就代表一个检测框的完整数据：[cx, cy, w, h, conf, kpt0_x, kpt0_y, kpt0_conf, ...]
    sctMatrixTranspose(ptr_device_temp0, ptr_device_temp1, features, samples);

    // 4. 根据置信度阈值过滤结果
    //    对于 YOLOv8-Pose：置信度在索引 4 (即第 5 列)
    //    过滤后，不符合要求的行被全部置为 0
    int results = sctFilterGreater_dim1(
        ptr_device_temp1,   // 输入：转置后的数据 [8400, 56]
        ptr_device_temp0,   // 输出：过滤后的数据
        4,                  // 置信度所在的列索引
        conf_threshold,     // 置信度阈值
        samples,            // dim0 - 样本数量 (8400)
        features            // dim1 - 特征数量 (56)
    );

    if (results > 0) {
        // 5. 按置信度降序排序
        //    排序后，有效结果排在前面
        sctSortTensor_dim1_descending(
            ptr_device_temp0,   // 输入：过滤后的数据
            ptr_device_temp1,   // 输出：排序后的数据
            samples,            // dim0 - 样本数量
            features,           // dim1 - 特征数量
            4                   // 排序依据：置信度列索引
        );

        // 6. 限制结果数量不超过最大值
        if (results > maximum) {
            results = maximum;
        }

        // 7. 确保输出向量大小足够
        if (output.size() < (size_t)results * features) {
            output.resize((size_t)results * features);
        }

        // 8. 将结果从设备内存拷贝到主机内存
        //    只拷贝有效的 results 行
        cudaMemcpy(output.data(), ptr_device_temp1,
                   results * features * sizeof(float),
                   cudaMemcpyDeviceToHost);
    } else {
        results = -1; // 没有找到有效结果
    }

    // 9. 清理 CUDA 内存
    cudaFree(ptr_device_temp0);
    cudaFree(ptr_device_temp1);

    return results;
}

/**
 * @brief 解析 CUDA 后处理的结果，转换为 C_YoloPose 结构
 *
 * CUDA 后处理返回的数据格式为 [N, 56]，每一行包含：
 * [cx, cy, w, h, conf, kpt0_x, kpt0_y, kpt0_conf, kpt1_x, kpt1_y, kpt1_conf, ...]
 *
 * @param processed_data  [in] CUDA 后处理的输出数据
 * @param num_detections  [in] 检测到的目标数量
 * @param num_features    [in] 每个检测的特征数量 (56)
 * @param scale_x         [in] 预处理的 x 缩放因子
 * @param scale_y         [in] 预处理的 y 缩放因子
 * @param pad_x           [in] 预处理的 x 填充
 * @param pad_y           [in] 预处理的 y 填充
 * @param original_width  [in] 原始图像宽度
 * @param original_height [in] 原始图像高度
 * @param detections      [out] 输出检测结果的数组
 */
static void parse_cuda_postproc_results(
    const float* processed_data,
    int num_detections,
    int num_features,
    float scale_x,
    float scale_y,
    int pad_x,
    int pad_y,
    int original_width,
    int original_height,
    C_YoloPose* detections
) {
    for (int i = 0; i < num_detections; i++) {
        const float* row = &processed_data[i * num_features];
        C_YoloPose* det = &detections[i];

        // 提取边界框 (cx, cy, w, h)
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];
        float confidence = row[4];

        // 坐标反算：模型坐标 -> 原始图像坐标
        cx = (cx - pad_x) / scale_x;
        cy = (cy - pad_y) / scale_y;
        w = w / scale_x;
        h = h / scale_y;

        // 将 (cx, cy, w, h) 转换为 (lx, ly, rx, ry)
        int lx = (int)((cx - w / 2.0f));
        int ly = (int)((cy - h / 2.0f));
        int rx = (int)((cx + w / 2.0f));
        int ry = (int)((cy + h / 2.0f));

        // 坐标裁剪
        lx = MAX(0, MIN(lx, original_width - 1));
        ly = MAX(0, MIN(ly, original_height - 1));
        rx = MAX(0, MIN(rx, original_width - 1));
        ry = MAX(0, MIN(ry, original_height - 1));

        // 存储边界框
        det->detection.lx = lx;
        det->detection.ly = ly;
        det->detection.rx = rx;
        det->detection.ry = ry;
        det->detection.cls = 0;  // YOLOv8-Pose 只检测 person
        det->detection.conf = confidence;

        // 解析关键点：从索引 5 开始，每 3 个值为一组 (x, y, conf)
        for (int k = 0; k < YOLO_POSE_NUM_KEYPOINTS; k++) {
            int kpt_offset = 5 + k * 3;
            float kpt_x = row[kpt_offset + 0];
            float kpt_y = row[kpt_offset + 1];
            float kpt_conf = row[kpt_offset + 2];

            // 坐标反算
            kpt_x = (kpt_x - pad_x) / scale_x;
            kpt_y = (kpt_y - pad_y) / scale_y;

            // 存储关键点
            det->pts[k].x = kpt_x;
            det->pts[k].y = kpt_y;
            det->pts[k].conf = kpt_conf;
        }
    }
}


// ============================================================================
//                         Pipeline Lifecycle
// ============================================================================

/**
 * @brief 获取默认的 YoloPose 管线配置。
 * @return 一个包含默认参数的 C_YoloPosePipelineConfig 结构体。
 */
C_YoloPosePipelineConfig c_yolopose_pipeline_get_default_config(void) {
    C_YoloPosePipelineConfig config = {
        .engine_path = NULL,             // 引擎路径（默认为空，必须由用户设置）
        .input_width = 640,              // 模型输入宽度
        .input_height = 640,             // 模型输入高度
        .max_batch_size = 1,             // 最大批处理大小
        .conf_threshold = 0.25f,         // 解码时使用的置信度阈值
        .iou_threshold = 0.45f,          // NMS（非极大值抑制）时使用的 IOU 阈值
        .num_keypoints = YOLO_POSE_NUM_KEYPOINTS, // 关键点数量 (例如 17)
    };
    return config;
}

/**
 * @brief 验证 YoloPose 管线配置是否有效。
 *
 * 检查配置指针是否为空、引擎路径是否已设置、
 * 输入尺寸和批量大小是否为正数，以及阈值是否在 [0, 1] 范围内。
 *
 * @param config [in] 指向待验证配置的指针。
 * @return 如果配置有效，返回 true；否则返回 false。
 */
bool c_yolopose_pipeline_validate_config(const C_YoloPosePipelineConfig* config) {
    // 检查空指针
    if (!config) return false;
    // 必须提供引擎路径
    if (!config->engine_path || strlen(config->engine_path) == 0) return false;
    // 输入尺寸必须为正
    if (config->input_width <= 0 || config->input_height <= 0) return false;
    // 批量大小必须为正
    if (config->max_batch_size <= 0) return false;
    // 置信度阈值必须在 [0, 1] 之间
    if (config->conf_threshold < 0.0f || config->conf_threshold > 1.0f) return false;
    // IOU 阈值必须在 [0, 1] 之间
    if (config->iou_threshold < 0.0f || config->iou_threshold > 1.0f) return false;
    
    // 所有检查通过
    return true;
}

/**
 * @brief 创建并初始化 YoloPose 管线上下文。
 *
 * 此函数执行以下操作：
 * 1. 验证配置。
 * 2. 分配上下文 (Context) 内存。
 * 3. 复制配置（包括深度复制 engine_path）。
 * 4. 创建并加载 TensorRT 引擎。
 * 5. 设置输入和输出维度。
 * 6. 分配主机（CPU）的输入/输出缓冲区。
 * 7. 创建 TensorRT 执行上下文 (Execution Context)。
 *
 * @param config [in] 指向用于初始化的配置的指针。
 * @return 成功时返回指向 C_YoloPosePipelineContext 的指针；失败时返回 NULL。
 */
C_YoloPosePipelineContext* c_yolopose_pipeline_create(
    const C_YoloPosePipelineConfig* config
) {
    // 1. 验证配置
    if (!c_yolopose_pipeline_validate_config(config)) {
        LOG_ERROR("YoloPosePipeline", "Invalid configuration");
        return NULL;
    }

    // 2. 分配上下文结构体内存
    // 使用 calloc 确保内存被初始化为零
    C_YoloPosePipelineContext* ctx = (C_YoloPosePipelineContext*)calloc(
        1, sizeof(C_YoloPosePipelineContext)
    );
    if (!ctx) {
        LOG_ERROR("YoloPosePipeline", "Failed to allocate context");
        return NULL;
    }

    // 3. 复制配置
    ctx->config = *config;
    // 深度复制 engine_path 字符串，因为原始 config 可能会被释放
    ctx->config.engine_path = strdup(config->engine_path);

    // 4. 创建 TensorRT 引擎实例
    ctx->trt_engine = new TrtEngineMultiTs(); // TrtEngineMultiTs 是一个自定义的 TRT 封装类
    if (!ctx->trt_engine) {
        LOG_ERROR("YoloPosePipeline", "Failed to create TRT engine");
        free((void*)ctx->config.engine_path);
        free(ctx);
        return NULL;
    }

    // 5. 从文件加载 TRT 引擎
    if (!ctx->trt_engine->loadFromFile(ctx->config.engine_path)) {
        LOG_ERROR("YoloPosePipeline", "Failed to load TRT engine");
        delete ctx->trt_engine;
        free((void*)ctx->config.engine_path);
        free(ctx);
        return NULL;
    }

    // 6. 计算/设置输出维度
    // YOLOv8-pose: 56 = 4(box) + 1(obj) + 51(17*3 kpts)
    // (注意：这里硬编码了 56 和 8400，这特定于 640x640 输入的 YOLOv8-Pose)
    ctx->output_features = 56;
    ctx->output_samples = 8400;  // 对应 640x640 输入的 anchor/sample 数量

    // 7. 创建输入张量 (Tensor) 的封装 (NCHW 格式)
    std::vector<int> input_dims = {
        ctx->config.max_batch_size, // N (Batch size)
        3,                          // C (Channels)
        ctx->config.input_height,   // H (Height)
        ctx->config.input_width     // W (Width)
    };
    ctx->input_tensor = new Tensor<float>(TensorType::FLOAT32, input_dims);

    // 8. 创建输出张量 (Tensor) 的封装
    // 格式: [N, Features, Samples] -> [N, 56, 8400]
    std::vector<int> output_dims = {
        ctx->config.max_batch_size, // N
        ctx->output_features,       // 56
        ctx->output_samples         // 8400
    };
    ctx->output_tensor = new Tensor<float>(TensorType::FLOAT32, output_dims);

    // 9. 分配主机（CPU）内存缓冲区
    int input_size = ctx->config.max_batch_size * 3 *
                       ctx->config.input_height * ctx->config.input_width;
    int output_size = ctx->config.max_batch_size *
                        ctx->output_features * ctx->output_samples;

    ctx->host_input_buffer = (float*)malloc(input_size * sizeof(float));
    ctx->host_output_buffer = (float*)malloc(output_size * sizeof(float));

    if (!ctx->host_input_buffer || !ctx->host_output_buffer) {
        LOG_ERROR("YoloPosePipeline", "Failed to allocate host buffers");
        c_yolopose_pipeline_destroy(ctx); // 调用 destroy 清理已分配的资源
        return NULL;
    }

    // 10. 创建 TensorRT 执行上下文 (Context)
    // 指定输入/输出节点的名称和输入形状
    std::vector<std::string> input_names = {"images"}; // 必须与模型 ONNX 中的名称一致
    std::vector<nvinfer1::Dims4> input_shapes = {
        {ctx->config.max_batch_size, 3, ctx->config.input_height, ctx->config.input_width}
    };
    std::vector<std::string> output_names = {"output0"}; // 必须与模型 ONNX 中的名称一致

    if (!ctx->trt_engine->createContext(input_names, input_shapes, output_names)) {
        LOG_ERROR("YoloPosePipeline", "Failed to create TRT context");
        c_yolopose_pipeline_destroy(ctx); // 清理
        return NULL;
    }

    LOG_INFO("YoloPosePipeline", "Pipeline created successfully");
    return ctx; // 返回初始化完毕的上下文
}

/**
 * @brief 销毁并释放 YoloPose 管线上下文。
 *
 * 此函数安全地释放 `c_yolopose_pipeline_create` 分配的所有资源，
 * 包括 C++ 对象 (delete) 和 C 内存 (free)。
 *
 * @param context [in] 指向要销毁的上下文的指针。
 */
void c_yolopose_pipeline_destroy(C_YoloPosePipelineContext* context) {
    if (!context) return; // 空指针检查

    // 释放 C++ 对象 (使用 delete)
    if (context->trt_engine) {
        delete context->trt_engine;
    }
    if (context->input_tensor) {
        delete context->input_tensor;
    }
    if (context->output_tensor) {
        delete context->output_tensor;
    }

    // 释放 C 内存 (使用 free)
    if (context->host_input_buffer) {
        free(context->host_input_buffer);
    }
    if (context->host_output_buffer) {
        free(context->host_output_buffer);
    }
    // 释放 strdup 分配的内存
    if (context->config.engine_path) {
        free((void*)context->config.engine_path);
    }

    // 最后释放上下文结构体本身
    free(context);
    LOG_INFO("YoloPosePipeline", "Pipeline destroyed");
}

// ============================================================================
//                         Inference Functions
// ============================================================================
/**
 * @brief 对单张图像执行 YoloPose 推理。
 *
 * 此函数完成完整的单图推理流程：
 * 1. 验证输入。
 * 2. 图像预处理 (letterbox, CHW, 归一化)。
 * 3. 将输入数据从 CPU 拷贝到 GPU。
 * 4. 执行 TensorRT 推理。
 * 5. 将输出数据从 GPU 拷贝回 CPU。
 * 6. 解码模型输出 (decode_yolopose_output)。
 * 7. 执行姿态非极大值抑制 (NMS)。
 * 8. 填充并返回最终结果。
 *
 * @param context [in] 指向 YoloPose 管线上下文的指针。
 * @param image   [in] 指向 C_ImageInput 结构（原始图像）的指针。
 * @param result  [out] 指向 C_YoloPoseImageResult 结构（用于存储结果）的指针。
 * @return 成功返回 true，失败返回 false。
 */
bool c_yolopose_infer_single(
    C_YoloPosePipelineContext* context,
    const C_ImageInput* image,
    C_YoloPoseImageResult* result
) {
    // 1. 基本的空指针检查
    if (!context || !image || !result) {
        return false;
    }

    // 2. 验证输入图像数据的有效性
    if (!image->data || image->width <= 0 || image->height <= 0 ||
        image->channels != 3) {
        snprintf(context->error_msg, sizeof(context->error_msg),
                 "Invalid input image");
        return false;
    }

    // 3. 图像预处理
    float scale_x, scale_y; // 用于存储反算坐标所需的缩放/填充值
    int pad_x, pad_y;

    // 调用 preprocess_image (之前已注释)
    // 将 image->data 预处理后放入 context->host_input_buffer
    if (!preprocess_image(
        image->data, image->width, image->height, image->channels,
        context->host_input_buffer, // 目标：主机输入缓冲区
        context->config.input_width, context->config.input_height,
        &scale_x, &scale_y, &pad_x, &pad_y
    )) {
        snprintf(context->error_msg, sizeof(context->error_msg),
                 "Image preprocessing failed");
        return false;
    }

    // 4. 将输入数据从 CPU 拷贝到 GPU
    // 注意：即使只推理一张图片，tensor 也是按 max_batch_size 分配的
    // 所以我们需要传入完整大小的数据，只有第一张图片是有效的
    int single_input_size = 3 * context->config.input_height * context->config.input_width;
    int total_input_size = context->config.max_batch_size * single_input_size;

    std::vector<float> input_vec(context->host_input_buffer,
                                 context->host_input_buffer + total_input_size);

    // input_tensor 是 GPU 上的数据封装
    context->input_tensor->copyFromVector(input_vec);

    // 5. 运行 TensorRT 推理
    std::vector<Tensor<float>*> inputs = {context->input_tensor};
    std::vector<Tensor<float>*> outputs = {context->output_tensor};

    if (!context->trt_engine->infer(inputs, outputs)) {
        snprintf(context->error_msg, sizeof(context->error_msg),
                 "TensorRT inference failed");
        return false;
    }

    // 6. 使用 CUDA 核函数进行后处理
    //    注意：我们需要获取 GPU 上的输出数据指针，而不是立即拷贝到 CPU
    const size_t max_detections = 300;
    std::vector<float> processed_output;

    // 获取 GPU 设备上的输出张量指针
    float* device_output_ptr = context->output_tensor->ptr();

    // 调用 CUDA 后处理函数（在 GPU 上执行转置、过滤、排序）
    int num_valid = kernel_decode_for_yolopose(
        device_output_ptr,              // GPU 上的原始输出 [56, 8400]
        processed_output,               // 输出缓冲区（会被填充）
        context->output_features,       // 56
        context->output_samples,        // 8400
        context->config.conf_threshold, // 置信度阈值
        max_detections                  // 最大返回数量
    );

    // 检查 CUDA 后处理是否成功
    if (num_valid <= 0) {
        snprintf(context->error_msg, sizeof(context->error_msg),
                 "CUDA post-processing failed or no detections found");
        result->image_index = 0;
        result->num_poses = 0;
        result->poses = nullptr;
        return true; // 没有检测到目标不算错误
    }

    // 7. 分配临时内存存储解析后的检测结果
    C_YoloPose* temp_detections = (C_YoloPose*)malloc(
        num_valid * sizeof(C_YoloPose)
    );
    if (!temp_detections) {
        snprintf(context->error_msg, sizeof(context->error_msg),
                 "Memory allocation failed");
        return false;
    }

    // 8. 解析 CUDA 后处理的结果
    parse_cuda_postproc_results(
        processed_output.data(),        // CUDA 后处理的输出
        num_valid,                      // 有效检测数量
        context->output_features,       // 56
        scale_x, scale_y, pad_x, pad_y, // 预处理参数
        image->width, image->height,    // 原始图像尺寸
        temp_detections                 // 输出：解析后的检测结果
    );

    // 9. 应用 NMS (非极大值抑制)
    C_YoloPose* nms_result = (C_YoloPose*)malloc(
        num_valid * sizeof(C_YoloPose)
    );
    if (!nms_result) {
        free(temp_detections);
        snprintf(context->error_msg, sizeof(context->error_msg),
                 "Memory allocation failed");
        return false;
    }

    size_t num_after_nms = 0;
    c_nms_pose(
        temp_detections, num_valid,
        context->config.iou_threshold,
        nms_result, &num_after_nms
    );

    // 10. 填充最终结果
    result->image_index = 0;
    result->num_poses = num_after_nms;
    result->poses = (C_YoloPose*)malloc(num_after_nms * sizeof(C_YoloPose));

    if (result->poses) {
        memcpy(result->poses, nms_result, num_after_nms * sizeof(C_YoloPose));
    }

    // 11. 清理临时内存
    free(temp_detections);
    free(nms_result);

    return result->poses != NULL;
}

/**
 * @brief 对一批图像执行 YoloPose 推理（真正的批处理实现）。
 *
 * 此函数实现了真正的批处理：
 * 1. 将所有图像一次性预处理并填充到批处理缓冲区
 * 2. 一次性执行 TensorRT 批量推理
 * 3. 分别处理每张图像的输出结果
 *
 * @param context [in] 指向 YoloPose 管线上下文的指针。
 * @param batch   [in] 指向 C_ImageBatch 结构（包含多张图像）的指针。
 * @param result  [out] 指向 C_YoloPoseBatchResult 结构（用于存储批量结果）的指针。
 * @return 成功返回 true，失败返回 false。
 */
bool c_yolopose_infer_batch(
    C_YoloPosePipelineContext* context,
    const C_ImageBatch* batch,
    C_YoloPoseBatchResult* result
) {
    // 1. 基本的空指针和有效性检查
    if (!context || !batch || !result || batch->count == 0) {
        return false;
    }

    // 2. 检查批次大小是否超过配置的最大批处理大小
    if (batch->count > (size_t)context->config.max_batch_size) {
        snprintf(context->error_msg, sizeof(context->error_msg),
                 "Batch size %zu exceeds max_batch_size %d",
                 batch->count, context->config.max_batch_size);
        return false;
    }

    // 3. 为每张图像准备预处理参数（scale 和 padding）
    float* scales_x = (float*)malloc(batch->count * sizeof(float));
    float* scales_y = (float*)malloc(batch->count * sizeof(float));
    int* pads_x = (int*)malloc(batch->count * sizeof(int));
    int* pads_y = (int*)malloc(batch->count * sizeof(int));

    if (!scales_x || !scales_y || !pads_x || !pads_y) {
        free(scales_x); free(scales_y); free(pads_x); free(pads_y);
        snprintf(context->error_msg, sizeof(context->error_msg),
                 "Memory allocation failed for preprocessing params");
        return false;
    }

    // 4. 批量预处理所有图像
    // 将每张图像预处理到 host_input_buffer 的对应位置
    int single_input_size = 3 * context->config.input_height * context->config.input_width;

    for (size_t i = 0; i < batch->count; i++) {
        const C_ImageInput* image = &batch->images[i];

        // 验证输入图像
        if (!image->data || image->width <= 0 || image->height <= 0 ||
            image->channels != 3) {
            snprintf(context->error_msg, sizeof(context->error_msg),
                     "Invalid input image at index %zu", i);
            free(scales_x); free(scales_y); free(pads_x); free(pads_y);
            return false;
        }

        // 预处理到批处理缓冲区的第 i 个位置
        float* buffer_offset = context->host_input_buffer + (i * single_input_size);

        if (!preprocess_image(
            image->data, image->width, image->height, image->channels,
            buffer_offset, // 每张图像写入对应的缓冲区位置
            context->config.input_width, context->config.input_height,
            &scales_x[i], &scales_y[i], &pads_x[i], &pads_y[i]
        )) {
            snprintf(context->error_msg, sizeof(context->error_msg),
                     "Image preprocessing failed at index %zu", i);
            free(scales_x); free(scales_y); free(pads_x); free(pads_y);
            return false;
        }
    }

    // 5. 将整个批次的输入数据从 CPU 拷贝到 GPU
    int total_input_size = batch->count * single_input_size;
    std::vector<float> input_vec(context->host_input_buffer,
                                  context->host_input_buffer + total_input_size);
    context->input_tensor->copyFromVector(input_vec);

    // 6. 执行批量推理
    std::vector<Tensor<float>*> inputs = {context->input_tensor};
    std::vector<Tensor<float>*> outputs = {context->output_tensor};

    if (!context->trt_engine->infer(inputs, outputs)) {
        snprintf(context->error_msg, sizeof(context->error_msg),
                 "TensorRT batch inference failed");
        free(scales_x); free(scales_y); free(pads_x); free(pads_y);
        return false;
    }

    // 7. 为批量结果分配内存
    result->num_images = batch->count;
    result->results = (C_YoloPoseImageResult*)calloc(
        batch->count, sizeof(C_YoloPoseImageResult)
    );

    if (!result->results) {
        free(scales_x); free(scales_y); free(pads_x); free(pads_y);
        return false;
    }

    // 8. 处理每张图像的输出结果
    const size_t max_detections = 300;
    float* device_output_ptr = context->output_tensor->ptr();

    for (size_t i = 0; i < batch->count; i++) {
        // 获取当前图像在批次中的输出位置
        // 输出格式: [batch, features, samples] -> [N, 56, 8400]
        int single_output_size = context->output_features * context->output_samples;
        float* device_image_output = device_output_ptr + (i * single_output_size);

        // 使用 CUDA 后处理当前图像的输出
        std::vector<float> processed_output;
        int num_valid = kernel_decode_for_yolopose(
            device_image_output,                // 当前图像的 GPU 输出
            processed_output,
            context->output_features,
            context->output_samples,
            context->config.conf_threshold,
            max_detections
        );

        // 如果没有检测到目标，设置为空结果
        if (num_valid <= 0) {
            result->results[i].image_index = i;
            result->results[i].num_poses = 0;
            result->results[i].poses = NULL;
            continue;
        }

        // 分配临时内存存储解析后的检测结果
        C_YoloPose* temp_detections = (C_YoloPose*)malloc(
            num_valid * sizeof(C_YoloPose)
        );
        if (!temp_detections) {
            snprintf(context->error_msg, sizeof(context->error_msg),
                     "Memory allocation failed for image %zu", i);
            // 清理之前成功的结果
            for (size_t j = 0; j < i; j++) {
                c_yolopose_image_result_free(&result->results[j]);
            }
            free(result->results);
            result->results = NULL;
            result->num_images = 0;
            free(scales_x); free(scales_y); free(pads_x); free(pads_y);
            return false;
        }

        // 解析 CUDA 后处理的结果
        parse_cuda_postproc_results(
            processed_output.data(),
            num_valid,
            context->output_features,
            scales_x[i], scales_y[i], pads_x[i], pads_y[i],
            batch->images[i].width, batch->images[i].height,
            temp_detections
        );

        // 应用 NMS
        C_YoloPose* nms_result = (C_YoloPose*)malloc(
            num_valid * sizeof(C_YoloPose)
        );
        if (!nms_result) {
            free(temp_detections);
            snprintf(context->error_msg, sizeof(context->error_msg),
                     "Memory allocation failed for NMS at image %zu", i);
            // 清理之前成功的结果
            for (size_t j = 0; j < i; j++) {
                c_yolopose_image_result_free(&result->results[j]);
            }
            free(result->results);
            result->results = NULL;
            result->num_images = 0;
            free(scales_x); free(scales_y); free(pads_x); free(pads_y);
            return false;
        }

        size_t num_after_nms = 0;
        c_nms_pose(
            temp_detections, num_valid,
            context->config.iou_threshold,
            nms_result, &num_after_nms
        );

        // 填充当前图像的结果
        result->results[i].image_index = i;
        result->results[i].num_poses = num_after_nms;
        result->results[i].poses = (C_YoloPose*)malloc(
            num_after_nms * sizeof(C_YoloPose)
        );

        if (result->results[i].poses) {
            memcpy(result->results[i].poses, nms_result,
                   num_after_nms * sizeof(C_YoloPose));
        }

        // 清理临时内存
        free(temp_detections);
        free(nms_result);
    }

    // 9. 清理预处理参数
    free(scales_x);
    free(scales_y);
    free(pads_x);
    free(pads_y);

    return true;
}

// ============================================================================
//                         Memory Management
// ============================================================================

/**
 * @brief 释放 C_YoloPoseImageResult 结构体中动态分配的内存。
 *
 * 此函数主要释放 `poses` 数组（该数组是在 c_yolopose_infer_single 中
 * 通过 malloc 分配的）。
 * 它不会释放 `result` 结构体本身（如果它也是动态分配的）。
 *
 * @param result [in/out] 指向要清理的图像结果结构体的指针。
 */
void c_yolopose_image_result_free(C_YoloPoseImageResult* result) {
    // 1. 空指针检查
    if (!result) return;

    // 2. 释放 poses 数组
    if (result->poses) {
        free(result->poses);      // 释放 C_YoloPose 数组
        result->poses = NULL;     // 置空指针，防止悬挂引用
    }
    
    // 3. 重置计数器
    result->num_poses = 0;
}

/**
 * @brief 释放 C_YoloPoseBatchResult 结构体中动态分配的内存。
 *
 * 此函数执行两级释放：
 * 1. 遍历 `results` 数组中的每一个 C_YoloPoseImageResult，
 * 并调用 `c_yolopose_image_result_free` 来释放它们各自的 `poses` 数组。
 * 2. 释放 `results` 数组本身（该数组是在 c_yolopose_infer_batch 中
 * 通过 calloc/malloc 分配的）。
 *
 * @param result [in/out] 指向要清理的批量结果结构体的指针。
 */
void c_yolopose_batch_result_free(C_YoloPoseBatchResult* result) {
    // 1. 空指针检查
    if (!result) return;

    // 2. 检查 results 数组是否有效
    if (result->results) {
        // 3. 遍历数组中的每个元素
        for (size_t i = 0; i < result->num_images; i++) {
            // 4. 对每个元素调用单图释放函数，释放其内部的 poses 数组
            c_yolopose_image_result_free(&result->results[i]);
        }
        
        // 5. 释放 C_YoloPoseImageResult 数组本身
        free(result->results);
        result->results = NULL; // 置空指针
    }
    
    // 6. 重置计数器
    result->num_images = 0;
}

// ============================================================================
//                         Utility Functions
// ============================================================================

/**
 * @brief 获取管线上下文中记录的最后一条错误信息。
 *
 * 此函数用于在推理或其他操作失败后，检索具体错误原因。
 *
 * @param context [in] 指向 YoloPose 管线上下文的指针。
 * @return 如果上下文中存在错误信息（即 error_msg[0] != '\0'），则返回该错误字符串的指针。
 * 如果 context 为 NULL 或没有设置错误信息，则返回 NULL。
 */
const char* c_yolopose_pipeline_get_last_error(
    C_YoloPosePipelineContext* context
) {
    // 1. 检查上下文是否有效
    if (!context) {
        return NULL; // 如果上下文为空，无法获取错误
    }

    // 2. 检查错误消息是否被设置（是否为空字符串）
    //    如果 error_msg[0] 不是空终止符 '\0'，说明有错误消息
    //    如果 error_msg[0] 是 '\0'，说明没有错误，返回 NULL
    return context->error_msg[0] != '\0' ? context->error_msg : NULL;
}
