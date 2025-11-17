/**
 * @file test_batch_processing.cpp
 * @brief 批处理功能测试程序
 *
 * 此程序用于验证 YOLOPose 和 EfficientNet pipeline 的真正批处理功能。
 * 测试场景:
 * 1. 单张图片推理 vs 批量推理的性能对比
 * 2. 不同批次大小的性能测试
 * 3. 结果准确性验证（批量结果应与单张结果一致）
 *
 * @author TrtEngineToolkits
 * @date 2025-11-12
 */

#include "trtengine_v2/pipelines/yolopose/c_yolopose_pipeline.h"
#include "trtengine_v2/pipelines/efficientnet/c_efficientnet_pipeline.h"
#include "trtengine_v2/utils/logger.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>

// ============================================================================
//                         性能测试工具
// ============================================================================

class PerformanceTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;

public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double stop_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;  // 转换为毫秒
    }
};

// ============================================================================
//                         辅助函数
// ============================================================================

/**
 * @brief 加载测试图像
 */
std::vector<cv::Mat> load_test_images(const std::vector<std::string>& image_paths) {
    std::vector<cv::Mat> images;

    for (const auto& path : image_paths) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            char buffer[512];
            snprintf(buffer, sizeof(buffer), "无法加载图像: %s", path.c_str());
            LOG_ERROR("BatchTest", buffer);
            continue;
        }

        // 转换为 RGB
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        images.push_back(img);
    }

    return images;
}

/**
 * @brief 准备 C_ImageBatch 结构
 */
C_ImageBatch prepare_image_batch(const std::vector<cv::Mat>& images,
                                  std::vector<C_ImageInput>& c_images) {
    c_images.clear();
    c_images.reserve(images.size());

    for (const auto& img : images) {
        C_ImageInput c_img;
        c_img.data = img.data;
        c_img.width = img.cols;
        c_img.height = img.rows;
        c_img.channels = img.channels();
        c_images.push_back(c_img);
    }

    C_ImageBatch batch;
    batch.images = c_images.data();
    batch.count = c_images.size();

    return batch;
}

// ============================================================================
//                         YOLOPose 批处理测试
// ============================================================================

void test_yolopose_batch_processing(const std::string& engine_path,
                                     const std::vector<cv::Mat>& test_images,
                                     int batch_size) {
    char buffer[512];
    LOG_INFO("BatchTest", "");
    LOG_INFO("BatchTest", "========================================");
    snprintf(buffer, sizeof(buffer), "YOLOPose 批处理测试 (Batch Size: %d)", batch_size);
    LOG_INFO("BatchTest", buffer);
    LOG_INFO("BatchTest", "========================================");

    // 创建 pipeline
    C_YoloPosePipelineConfig config = c_yolopose_pipeline_get_default_config();
    config.engine_path = engine_path.c_str();
    config.max_batch_size = batch_size;
    config.input_width = 640;
    config.input_height = 640;
    config.conf_threshold = 0.25f;
    config.iou_threshold = 0.45f;

    C_YoloPosePipelineContext* ctx = c_yolopose_pipeline_create(&config);
    if (!ctx) {
        LOG_ERROR("BatchTest", "创建 YOLOPose pipeline 失败");
        return;
    }

    LOG_INFO("BatchTest", "YOLOPose pipeline 创建成功");

    // 准备测试图像（取前 batch_size 张）
    size_t num_images = std::min((size_t)batch_size, test_images.size());
    std::vector<cv::Mat> batch_images(test_images.begin(),
                                       test_images.begin() + num_images);

    snprintf(buffer, sizeof(buffer), "测试图像数量: %zu", num_images);
    LOG_INFO("BatchTest", buffer);

    // ========================================================================
    // 测试 1: 单张图片循环推理
    // ========================================================================
    LOG_INFO("BatchTest", "");
    LOG_INFO("BatchTest", "[测试 1] 单张图片循环推理...");
    PerformanceTimer timer;
    std::vector<C_YoloPoseImageResult> single_results(num_images);

    timer.start();
    for (size_t i = 0; i < num_images; i++) {
        C_ImageInput c_img;
        c_img.data = batch_images[i].data;
        c_img.width = batch_images[i].cols;
        c_img.height = batch_images[i].rows;
        c_img.channels = batch_images[i].channels();

        bool success = c_yolopose_infer_single(ctx, &c_img, &single_results[i]);
        if (!success) {
            snprintf(buffer, sizeof(buffer), "单张推理失败 (图像 %zu)", i);
            LOG_ERROR("BatchTest", buffer);
        }
    }
    double single_time = timer.stop_ms();
    snprintf(buffer, sizeof(buffer), "[结果] 单张循环推理总时间: %.2f ms (平均每张: %.2f ms)",
           single_time, single_time / num_images);
    LOG_INFO("BatchTest", buffer);

    // 打印单张推理结果摘要
    for (size_t i = 0; i < num_images; i++) {
        snprintf(buffer, sizeof(buffer), "  图像 %zu: 检测到 %zu 个姿态", i, single_results[i].num_poses);
        LOG_INFO("BatchTest", buffer);
    }

    // ========================================================================
    // 测试 2: 真正的批量推理
    // ========================================================================
    LOG_INFO("BatchTest", "");
    LOG_INFO("BatchTest", "[测试 2] 批量推理...");
    std::vector<C_ImageInput> c_images;
    C_ImageBatch batch = prepare_image_batch(batch_images, c_images);

    C_YoloPoseBatchResult batch_result;
    batch_result.results = NULL;
    batch_result.num_images = 0;

    timer.start();
    bool success = c_yolopose_infer_batch(ctx, &batch, &batch_result);
    double batch_time = timer.stop_ms();

    if (!success) {
        LOG_ERROR("BatchTest", "批量推理失败");
        const char* error = c_yolopose_pipeline_get_last_error(ctx);
        if (error) {
            snprintf(buffer, sizeof(buffer), "错误信息: %s", error);
            LOG_ERROR("BatchTest", buffer);
        }
    } else {
        snprintf(buffer, sizeof(buffer), "[结果] 批量推理总时间: %.2f ms (平均每张: %.2f ms)",
               batch_time, batch_time / num_images);
        LOG_INFO("BatchTest", buffer);
        snprintf(buffer, sizeof(buffer), "[性能] 加速比: %.2fx", single_time / batch_time);
        LOG_INFO("BatchTest", buffer);

        // 打印批量推理结果摘要
        for (size_t i = 0; i < batch_result.num_images; i++) {
            snprintf(buffer, sizeof(buffer), "  图像 %zu: 检测到 %zu 个姿态",
                   i, batch_result.results[i].num_poses);
            LOG_INFO("BatchTest", buffer);
        }

        // ====================================================================
        // 测试 3: 验证结果一致性
        // ====================================================================
        LOG_INFO("BatchTest", "");
        LOG_INFO("BatchTest", "[测试 3] 验证批量推理与单张推理结果一致性...");
        bool results_match = true;

        for (size_t i = 0; i < num_images; i++) {
            if (single_results[i].num_poses != batch_result.results[i].num_poses) {
                snprintf(buffer, sizeof(buffer), "图像 %zu 检测数量不一致: 单张=%zu, 批量=%zu",
                       i, single_results[i].num_poses,
                       batch_result.results[i].num_poses);
                LOG_WARNING("BatchTest", buffer);
                results_match = false;
            }
        }

        if (results_match) {
            LOG_INFO("BatchTest", "✓ 批量推理结果与单张推理一致！");
        } else {
            LOG_WARNING("BatchTest", "✗ 批量推理结果与单张推理存在差异");
        }

        // 释放批量结果
        c_yolopose_batch_result_free(&batch_result);
    }

    // 释放单张结果
    for (auto& result : single_results) {
        c_yolopose_image_result_free(&result);
    }

    // 销毁 pipeline
    c_yolopose_pipeline_destroy(ctx);
    LOG_INFO("BatchTest", "");
    LOG_INFO("BatchTest", "YOLOPose 批处理测试完成");
}

// ============================================================================
//                         EfficientNet 批处理测试
// ============================================================================

void test_efficientnet_batch_processing(const std::string& engine_path,
                                         const std::vector<cv::Mat>& test_images,
                                         int batch_size) {
    char buffer[512];
    LOG_INFO("BatchTest", "");
    LOG_INFO("BatchTest", "========================================");
    snprintf(buffer, sizeof(buffer), "EfficientNet 批处理测试 (Batch Size: %d)", batch_size);
    LOG_INFO("BatchTest", buffer);
    LOG_INFO("BatchTest", "========================================");

    // 创建 pipeline
    C_EfficientNetPipelineConfig config = c_efficientnet_pipeline_get_default_config();
    config.engine_path = engine_path.c_str();
    config.max_batch_size = batch_size;
    config.input_width = 224;
    config.input_height = 224;

    C_EfficientNetPipelineContext* ctx = c_efficientnet_pipeline_create(&config);
    if (!ctx) {
        LOG_ERROR("BatchTest", "创建 EfficientNet pipeline 失败");
        return;
    }

    LOG_INFO("BatchTest", "EfficientNet pipeline 创建成功");

    // 准备测试图像（取前 batch_size 张）
    size_t num_images = std::min((size_t)batch_size, test_images.size());
    std::vector<cv::Mat> batch_images(test_images.begin(),
                                       test_images.begin() + num_images);

    snprintf(buffer, sizeof(buffer), "测试图像数量: %zu", num_images);
    LOG_INFO("BatchTest", buffer);

    // ========================================================================
    // 测试 1: 单张图片循环推理
    // ========================================================================
    LOG_INFO("BatchTest", "");
    LOG_INFO("BatchTest", "[测试 1] 单张图片循环推理...");
    PerformanceTimer timer;
    std::vector<C_EfficientNetResult> single_results(num_images);

    timer.start();
    for (size_t i = 0; i < num_images; i++) {
        C_ImageInput c_img;
        c_img.data = batch_images[i].data;
        c_img.width = batch_images[i].cols;
        c_img.height = batch_images[i].rows;
        c_img.channels = batch_images[i].channels();

        bool success = c_efficientnet_infer_single(ctx, &c_img, &single_results[i]);
        if (!success) {
            snprintf(buffer, sizeof(buffer), "单张推理失败 (图像 %zu)", i);
            LOG_ERROR("BatchTest", buffer);
        }
    }
    double single_time = timer.stop_ms();
    snprintf(buffer, sizeof(buffer), "[结果] 单张循环推理总时间: %.2f ms (平均每张: %.2f ms)",
           single_time, single_time / num_images);
    LOG_INFO("BatchTest", buffer);

    // 打印单张推理结果摘要
    for (size_t i = 0; i < num_images; i++) {
        snprintf(buffer, sizeof(buffer), "  图像 %zu: 类别=%d, 置信度=%.4f",
               i, single_results[i].class_id, single_results[i].confidence);
        LOG_INFO("BatchTest", buffer);
    }

    // ========================================================================
    // 测试 2: 真正的批量推理
    // ========================================================================
    LOG_INFO("BatchTest", "");
    LOG_INFO("BatchTest", "[测试 2] 批量推理...");
    std::vector<C_ImageInput> c_images;
    C_ImageBatch batch = prepare_image_batch(batch_images, c_images);

    C_EfficientNetBatchResult batch_result;
    batch_result.results = NULL;
    batch_result.count = 0;

    timer.start();
    bool success = c_efficientnet_infer_batch(ctx, &batch, &batch_result);
    double batch_time = timer.stop_ms();

    if (!success) {
        LOG_ERROR("BatchTest", "批量推理失败");
        const char* error = c_efficientnet_pipeline_get_last_error(ctx);
        if (error) {
            snprintf(buffer, sizeof(buffer), "错误信息: %s", error);
            LOG_ERROR("BatchTest", buffer);
        }
    } else {
        snprintf(buffer, sizeof(buffer), "[结果] 批量推理总时间: %.2f ms (平均每张: %.2f ms)",
               batch_time, batch_time / num_images);
        LOG_INFO("BatchTest", buffer);
        snprintf(buffer, sizeof(buffer), "[性能] 加速比: %.2fx", single_time / batch_time);
        LOG_INFO("BatchTest", buffer);

        // 打印批量推理结果摘要
        for (size_t i = 0; i < batch_result.count; i++) {
            snprintf(buffer, sizeof(buffer), "  图像 %zu: 类别=%d, 置信度=%.4f",
                   i, batch_result.results[i].class_id,
                   batch_result.results[i].confidence);
            LOG_INFO("BatchTest", buffer);
        }

        // ====================================================================
        // 测试 3: 验证结果一致性
        // ====================================================================
        LOG_INFO("BatchTest", "");
        LOG_INFO("BatchTest", "[测试 3] 验证批量推理与单张推理结果一致性...");
        bool results_match = true;

        for (size_t i = 0; i < num_images; i++) {
            if (single_results[i].class_id != batch_result.results[i].class_id) {
                snprintf(buffer, sizeof(buffer), "图像 %zu 分类结果不一致: 单张=%d, 批量=%d",
                       i, single_results[i].class_id,
                       batch_result.results[i].class_id);
                LOG_WARNING("BatchTest", buffer);
                results_match = false;
            }

            float conf_diff = std::abs(single_results[i].confidence -
                                       batch_result.results[i].confidence);
            if (conf_diff > 0.001f) {
                snprintf(buffer, sizeof(buffer), "图像 %zu 置信度存在差异: 单张=%.4f, 批量=%.4f",
                       i, single_results[i].confidence,
                       batch_result.results[i].confidence);
                LOG_WARNING("BatchTest", buffer);
                results_match = false;
            }
        }

        if (results_match) {
            LOG_INFO("BatchTest", "✓ 批量推理结果与单张推理一致！");
        } else {
            LOG_WARNING("BatchTest", "✗ 批量推理结果与单张推理存在差异");
        }

        // 释放批量结果
        c_efficientnet_batch_result_free(&batch_result);
    }

    // 释放单张结果
    for (auto& result : single_results) {
        c_efficientnet_result_free(&result);
    }

    // 销毁 pipeline
    c_efficientnet_pipeline_destroy(ctx);
    LOG_INFO("BatchTest", "");
    LOG_INFO("BatchTest", "EfficientNet 批处理测试完成");
}

// ============================================================================
//                         主函数
// ============================================================================

int main(int argc, char** argv) {
    char buffer[512];
    LOG_INFO("BatchTest", "========================================");
    LOG_INFO("BatchTest", "批处理功能测试程序");
    LOG_INFO("BatchTest", "========================================");

    // 检查命令行参数
    if (argc < 4) {
        snprintf(buffer, sizeof(buffer), "用法: %s <model_type> <engine_path> <image1> [image2] [image3] ...", argv[0]);
        LOG_INFO("BatchTest", buffer);
        LOG_INFO("BatchTest", "  model_type: yolopose 或 efficientnet");
        LOG_INFO("BatchTest", "  engine_path: TensorRT 引擎文件路径");
        LOG_INFO("BatchTest", "  image*: 测试图像路径（至少1张）");
        LOG_INFO("BatchTest", "");
        LOG_INFO("BatchTest", "示例:");
        snprintf(buffer, sizeof(buffer), "  %s yolopose ./yolopose.engine ./test1.jpg ./test2.jpg ./test3.jpg", argv[0]);
        LOG_INFO("BatchTest", buffer);
        snprintf(buffer, sizeof(buffer), "  %s efficientnet ./efficientnet.engine ./img1.jpg ./img2.jpg", argv[0]);
        LOG_INFO("BatchTest", buffer);
        return 1;
    }

    std::string model_type = argv[1];
    std::string engine_path = argv[2];

    // 加载测试图像
    std::vector<std::string> image_paths;
    for (int i = 3; i < argc; i++) {
        image_paths.push_back(argv[i]);
    }

    snprintf(buffer, sizeof(buffer), "加载 %zu 张测试图像...", image_paths.size());
    LOG_INFO("BatchTest", buffer);
    std::vector<cv::Mat> test_images = load_test_images(image_paths);

    if (test_images.empty()) {
        LOG_ERROR("BatchTest", "没有成功加载任何图像");
        return 1;
    }

    snprintf(buffer, sizeof(buffer), "成功加载 %zu 张图像", test_images.size());
    LOG_INFO("BatchTest", buffer);

    // 执行测试
    if (model_type == "yolopose") {
        // 测试不同的批次大小
        std::vector<int> batch_sizes = {1, 2, 4};
        for (int batch_size : batch_sizes) {
            if ((size_t)batch_size <= test_images.size()) {
                test_yolopose_batch_processing(engine_path, test_images, batch_size);
            }
        }
    } else if (model_type == "efficientnet") {
        // 测试不同的批次大小
        std::vector<int> batch_sizes = {1, 2, 4};
        for (int batch_size : batch_sizes) {
            if ((size_t)batch_size <= test_images.size()) {
                test_efficientnet_batch_processing(engine_path, test_images, batch_size);
            }
        }
    } else {
        snprintf(buffer, sizeof(buffer), "不支持的模型类型: %s", model_type.c_str());
        LOG_ERROR("BatchTest", buffer);
        LOG_ERROR("BatchTest", "        支持的类型: yolopose, efficientnet");
        return 1;
    }

    LOG_INFO("BatchTest", "");
    LOG_INFO("BatchTest", "========================================");
    LOG_INFO("BatchTest", "所有测试完成！");
    LOG_INFO("BatchTest", "========================================");

    return 0;
}
