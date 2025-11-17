/**
 * @file functional_test_v2_cascade.cpp
 * @brief V2 级联推理测试程序（含性能基准测试）
 *
 * 功能流程:
 * 1. YOLOv8-Pose 检测人物和关键点
 * 2. 绘制关键点和骨架连线
 * 3. 裁剪检测到的人物区域
 * 4. EfficientNet 对裁剪区域进行分类和特征提取
 *
 * 性能测试功能:
 * - 模型预热（避免首次推理的初始化开销）
 * - 100次循环测试取平均值
 * - 每个pipeline的详细时间统计
 * - 内存使用监控
 * - 内存泄漏检测
 *
 * @author TrtEngineToolkits
 * @date 2025-11-10
 */

#include "trtengine_v2/pipelines/yolopose/c_yolopose_pipeline.h"
#include "trtengine_v2/pipelines/efficientnet/c_efficientnet_pipeline.h"
#include "trtengine_v2/utils/logger.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sys/resource.h>
#include <unistd.h>

// ============================================================================
//                         性能测试工具
// ============================================================================

/**
 * @brief 性能计时器类
 */
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

/**
 * @brief 性能统计数据
 */
struct PerformanceStats {
    std::vector<double> times;

    void add(double time_ms) {
        times.push_back(time_ms);
    }

    double get_mean() const {
        if (times.empty()) return 0.0;
        double sum = 0.0;
        for (double t : times) sum += t;
        return sum / times.size();
    }

    double get_min() const {
        if (times.empty()) return 0.0;
        return *std::min_element(times.begin(), times.end());
    }

    double get_max() const {
        if (times.empty()) return 0.0;
        return *std::max_element(times.begin(), times.end());
    }

    double get_std() const {
        if (times.size() < 2) return 0.0;
        double mean = get_mean();
        double variance = 0.0;
        for (double t : times) {
            double diff = t - mean;
            variance += diff * diff;
        }
        return sqrt(variance / times.size());
    }

    void clear() {
        times.clear();
    }
};

/**
 * @brief 内存使用监控器
 */
class MemoryMonitor {
private:
    size_t initial_rss_kb;

    size_t get_current_rss_kb() {
        std::ifstream status_file("/proc/self/status");
        std::string line;
        while (std::getline(status_file, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                size_t rss_kb;
                sscanf(line.c_str(), "VmRSS: %zu", &rss_kb);
                return rss_kb;
            }
        }
        return 0;
    }

public:
    void start() {
        initial_rss_kb = get_current_rss_kb();
    }

    size_t get_current_usage_mb() {
        return get_current_rss_kb() / 1024;
    }

    long get_delta_mb() {
        size_t current_rss_kb = get_current_rss_kb();
        return (long)(current_rss_kb - initial_rss_kb) / 1024;
    }
};

// ============================================================================
//                         COCO 骨架连接定义
// ============================================================================

// COCO 骨架连接定义 (17个关键点)
static const int SKELETON_CONNECTIONS[][2] = {
    {0, 1}, {0, 2},   // 鼻子到眼睛
    {1, 3}, {2, 4},   // 眼睛到耳朵
    {0, 5}, {0, 6},   // 鼻子到肩膀
    {5, 7}, {7, 9},   // 左臂
    {6, 8}, {8, 10},  // 右臂
    {5, 6},           // 肩膀连接
    {5, 11}, {6, 12}, // 肩膀到髋部
    {11, 12},         // 髋部连接
    {11, 13}, {13, 15}, // 左腿
    {12, 14}, {14, 16}  // 右腿
};

static const int NUM_SKELETON_CONNECTIONS = sizeof(SKELETON_CONNECTIONS) / sizeof(SKELETON_CONNECTIONS[0]);

// ============================================================================
//                         图像处理函数
// ============================================================================

/**
 * @brief 根据关键点索引获取颜色
 */
cv::Scalar get_keypoint_color(int kpt_idx) {
    // OpenCV 使用 BGR 格式
    if (kpt_idx == 0) {
        return cv::Scalar(0, 0, 255);        // 红色 - 鼻子
    } else if (kpt_idx >= 1 && kpt_idx <= 4) {
        return cv::Scalar(255, 0, 0);        // 蓝色 - 眼睛和耳朵
    } else if (kpt_idx >= 5 && kpt_idx <= 6) {
        return cv::Scalar(0, 255, 0);        // 绿色 - 肩膀
    } else if (kpt_idx >= 7 && kpt_idx <= 10) {
        return cv::Scalar(255, 255, 0);      // 青色 - 手臂
    } else {
        return cv::Scalar(255, 0, 255);      // 品红色 - 腿部
    }
}

/**
 * @brief 在图像上绘制单个姿态 (边界框 + 关键点 + 骨架)
 */
void draw_pose(cv::Mat& image, const C_YoloPose* pose, float conf_threshold = 0.3f) {
    // 1. 绘制边界框
    cv::rectangle(image,
                  cv::Point(pose->detection.lx, pose->detection.ly),
                  cv::Point(pose->detection.rx, pose->detection.ry),
                  cv::Scalar(0, 255, 0),  // 绿色
                  2);

    // 2. 绘制骨架连线
    for (int i = 0; i < NUM_SKELETON_CONNECTIONS; i++) {
        int idx1 = SKELETON_CONNECTIONS[i][0];
        int idx2 = SKELETON_CONNECTIONS[i][1];

        const C_KeyPoint* p1 = &pose->pts[idx1];
        const C_KeyPoint* p2 = &pose->pts[idx2];

        // 只绘制置信度足够高的连线
        if (p1->conf > conf_threshold && p2->conf > conf_threshold) {
            cv::line(image,
                     cv::Point((int)p1->x, (int)p1->y),
                     cv::Point((int)p2->x, (int)p2->y),
                     cv::Scalar(0, 255, 255),  // 黄色 (BGR)
                     2);
        }
    }

    // 3. 绘制关键点 (在连线之上)
    for (int i = 0; i < 17; i++) {
        const C_KeyPoint* kpt = &pose->pts[i];
        if (kpt->conf > conf_threshold) {
            cv::Scalar color = get_keypoint_color(i);
            cv::circle(image,
                       cv::Point((int)kpt->x, (int)kpt->y),
                       5,
                       color,
                       -1);  // 填充圆
        }
    }
}

// ============================================================================
//                         主程序
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        char buffer[512];
        snprintf(buffer, sizeof(buffer), "Usage: %s <yolopose_engine> <efficientnet_engine> <input_image> [--benchmark]", argv[0]);
        LOG_INFO("CascadeTest", buffer);
        LOG_INFO("CascadeTest", "");
        LOG_INFO("CascadeTest", "Modes:");
        LOG_INFO("CascadeTest", "  Normal mode:    Run once and save visualizations");
        LOG_INFO("CascadeTest", "  Benchmark mode: Run 100 iterations with performance statistics (use --benchmark)");
        LOG_INFO("CascadeTest", "");
        LOG_INFO("CascadeTest", "Example:");
        snprintf(buffer, sizeof(buffer), "  %s yolov8n-pose.engine efficientnet_b0.engine people.jpg", argv[0]);
        LOG_INFO("CascadeTest", buffer);
        snprintf(buffer, sizeof(buffer), "  %s yolov8n-pose.engine efficientnet_b0.engine people.jpg --benchmark", argv[0]);
        LOG_INFO("CascadeTest", buffer);
        return 1;
    }

    const char* yolopose_engine_path = argv[1];
    const char* efficientnet_engine_path = argv[2];
    const char* input_image_path = argv[3];
    bool benchmark_mode = (argc == 5 && strcmp(argv[4], "--benchmark") == 0);

    LOG_INFO("CascadeTest", "========================================");
    char buffer[512];
    snprintf(buffer, sizeof(buffer), "V2 级联推理测试程序%s", benchmark_mode ? "（性能测试模式）" : "");
    LOG_INFO("CascadeTest", buffer);
    LOG_INFO("CascadeTest", "========================================");
    snprintf(buffer, sizeof(buffer), "YOLOv8-Pose Engine: %s", yolopose_engine_path);
    LOG_INFO("CascadeTest", buffer);
    snprintf(buffer, sizeof(buffer), "EfficientNet Engine: %s", efficientnet_engine_path);
    LOG_INFO("CascadeTest", buffer);
    snprintf(buffer, sizeof(buffer), "Input Image: %s", input_image_path);
    LOG_INFO("CascadeTest", buffer);
    if (benchmark_mode) {
        LOG_INFO("CascadeTest", "模式: 性能基准测试 (100次迭代)");
    }
    LOG_INFO("CascadeTest", "========================================");
    LOG_INFO("CascadeTest", "");

    // ========================================================================
    // [1/6] 加载输入图像
    // ========================================================================
    LOG_INFO("CascadeTest", "[1/6] 加载输入图像...");
    cv::Mat input_image = cv::imread(input_image_path);
    if (input_image.empty()) {
        snprintf(buffer, sizeof(buffer), "  错误: 无法加载图像 '%s'", input_image_path);
        LOG_ERROR("CascadeTest", buffer);
        return 1;
    }

    // 转换为 RGB (OpenCV 默认是 BGR)
    cv::Mat input_image_rgb;
    cv::cvtColor(input_image, input_image_rgb, cv::COLOR_BGR2RGB);

    snprintf(buffer, sizeof(buffer), "  图像尺寸: %dx%d, 通道数: %d",
           input_image_rgb.cols, input_image_rgb.rows, input_image_rgb.channels());
    LOG_INFO("CascadeTest", buffer);

    // ========================================================================
    // [2/6] 创建 YOLOv8-Pose Pipeline
    // ========================================================================
    LOG_INFO("CascadeTest", "");
    LOG_INFO("CascadeTest", "[2/6] 创建 YOLOv8-Pose Pipeline...");
    C_YoloPosePipelineConfig pose_config = c_yolopose_pipeline_get_default_config();
    pose_config.engine_path = yolopose_engine_path;
    pose_config.conf_threshold = 0.25f;
    pose_config.iou_threshold = 0.45f;

    C_YoloPosePipelineContext* pose_pipeline = c_yolopose_pipeline_create(&pose_config);
    if (!pose_pipeline) {
        LOG_ERROR("CascadeTest", "  错误: 无法创建 YOLOv8-Pose Pipeline");
        return 1;
    }
    LOG_INFO("CascadeTest", "  Pipeline 创建成功");

    // ========================================================================
    // [3/6] 运行姿态检测
    // ========================================================================
    LOG_INFO("CascadeTest", "");
    LOG_INFO("CascadeTest", "[3/6] 运行姿态检测...");

    C_ImageInput image_input;
    image_input.data = input_image_rgb.data;
    image_input.width = input_image_rgb.cols;
    image_input.height = input_image_rgb.rows;
    image_input.channels = input_image_rgb.channels();

    C_YoloPoseImageResult pose_result = {0};

    // 如果是性能测试模式，执行预热和基准测试
    if (benchmark_mode) {
        LOG_INFO("CascadeTest", "");
        LOG_INFO("CascadeTest", ">>> 开始性能基准测试 <<<");
        LOG_INFO("CascadeTest", "");

        // 初始化内存监控
        MemoryMonitor mem_monitor;
        mem_monitor.start();
        snprintf(buffer, sizeof(buffer), "[内存] 初始内存使用: %zu MB", mem_monitor.get_current_usage_mb());
        LOG_INFO("CascadeTest", buffer);
        LOG_INFO("CascadeTest", "");

        // 预热阶段
        LOG_INFO("CascadeTest", "[预热] 执行 10 次推理进行模型预热...");
        for (int i = 0; i < 10; i++) {
            C_YoloPoseImageResult warmup_result = {0};
            c_yolopose_infer_single(pose_pipeline, &image_input, &warmup_result);
            c_yolopose_image_result_free(&warmup_result);
            printf(".");
            fflush(stdout);
        }
        printf("\n");
        LOG_INFO("CascadeTest", " 完成");
        snprintf(buffer, sizeof(buffer), "[内存] 预热后内存使用: %zu MB (增加 %ld MB)",
               mem_monitor.get_current_usage_mb(), mem_monitor.get_delta_mb());
        LOG_INFO("CascadeTest", buffer);
        LOG_INFO("CascadeTest", "");

        // 性能测试阶段
        LOG_INFO("CascadeTest", "[测试] 执行 100 次推理并统计性能...");
        PerformanceStats pose_stats;

        size_t initial_mem = mem_monitor.get_current_usage_mb();

        for (int i = 0; i < 100; i++) {
            C_YoloPoseImageResult bench_result = {0};

            // 每次循环创建新的计时器实例
            PerformanceTimer timer;
            timer.start();
            c_yolopose_infer_single(pose_pipeline, &image_input, &bench_result);
            double elapsed = timer.stop_ms();

            pose_stats.add(elapsed);
            c_yolopose_image_result_free(&bench_result);

            if ((i + 1) % 20 == 0) {
                snprintf(buffer, sizeof(buffer), "  完成 %d/100 次...", i + 1);
                LOG_INFO("CascadeTest", buffer);
            }
        }

        size_t final_mem = mem_monitor.get_current_usage_mb();
        long mem_leak = final_mem - initial_mem;

        // 输出性能统计
        LOG_INFO("CascadeTest", "");
        LOG_INFO("CascadeTest", "========================================");
        LOG_INFO("CascadeTest", "YOLOv8-Pose 性能统计 (100次)");
        LOG_INFO("CascadeTest", "========================================");
        snprintf(buffer, sizeof(buffer), "  平均耗时: %.3f ms", pose_stats.get_mean());
        LOG_INFO("CascadeTest", buffer);
        snprintf(buffer, sizeof(buffer), "  最小耗时: %.3f ms", pose_stats.get_min());
        LOG_INFO("CascadeTest", buffer);
        snprintf(buffer, sizeof(buffer), "  最大耗时: %.3f ms", pose_stats.get_max());
        LOG_INFO("CascadeTest", buffer);
        snprintf(buffer, sizeof(buffer), "  标准差:   %.3f ms", pose_stats.get_std());
        LOG_INFO("CascadeTest", buffer);
        snprintf(buffer, sizeof(buffer), "  吞吐量:   %.2f FPS", 1000.0 / pose_stats.get_mean());
        LOG_INFO("CascadeTest", buffer);
        LOG_INFO("CascadeTest", "");
        snprintf(buffer, sizeof(buffer), "[内存] 测试后内存使用: %zu MB", final_mem);
        LOG_INFO("CascadeTest", buffer);
        snprintf(buffer, sizeof(buffer), "[内存] 内存变化: %s%ld MB %s",
               mem_leak > 0 ? "+" : "",
               mem_leak,
               mem_leak > 5 ? "(可能存在内存泄漏)" : "(正常)");
        LOG_INFO("CascadeTest", buffer);
        LOG_INFO("CascadeTest", "========================================");
        LOG_INFO("CascadeTest", "");

        // 最后执行一次获取结果用于后续可视化
        if (!c_yolopose_infer_single(pose_pipeline, &image_input, &pose_result)) {
            snprintf(buffer, sizeof(buffer), "  错误: 姿态检测失败: %s",
                   c_yolopose_pipeline_get_last_error(pose_pipeline));
            LOG_ERROR("CascadeTest", buffer);
            c_yolopose_pipeline_destroy(pose_pipeline);
            return 1;
        }
    } else {
        // 正常模式：只执行一次
        if (!c_yolopose_infer_single(pose_pipeline, &image_input, &pose_result)) {
            snprintf(buffer, sizeof(buffer), "  错误: 姿态检测失败: %s",
                   c_yolopose_pipeline_get_last_error(pose_pipeline));
            LOG_ERROR("CascadeTest", buffer);
            c_yolopose_pipeline_destroy(pose_pipeline);
            return 1;
        }
    }

    snprintf(buffer, sizeof(buffer), "  检测到 %zu 个人", pose_result.num_poses);
    LOG_INFO("CascadeTest", buffer);

    if (pose_result.num_poses == 0) {
        LOG_WARNING("CascadeTest", "  警告: 未检测到任何人物，程序退出");
        c_yolopose_image_result_free(&pose_result);
        c_yolopose_pipeline_destroy(pose_pipeline);
        return 0;
    }

    // 输出检测结果
    for (size_t i = 0; i < pose_result.num_poses; i++) {
        const C_YoloPose* pose = &pose_result.poses[i];
        snprintf(buffer, sizeof(buffer), "  Person %zu: bbox=[%d,%d,%d,%d], conf=%.2f",
               i, pose->detection.lx, pose->detection.ly,
               pose->detection.rx, pose->detection.ry,
               pose->detection.conf);
        LOG_INFO("CascadeTest", buffer);
    }

    // 绘制姿态到图像上 (BGR 格式)
    cv::Mat pose_image = input_image.clone();
    for (size_t i = 0; i < pose_result.num_poses; i++) {
        draw_pose(pose_image, &pose_result.poses[i]);
    }

    // 保存姿态图像
    const char* pose_output_path = "output_pose.jpg";
    cv::imwrite(pose_output_path, pose_image);
    snprintf(buffer, sizeof(buffer), "  已保存姿态图像: %s", pose_output_path);
    LOG_INFO("CascadeTest", buffer);

    // ========================================================================
    // [4/6] 创建 EfficientNet Pipeline
    // ========================================================================
    LOG_INFO("CascadeTest", "");
    LOG_INFO("CascadeTest", "[4/6] 创建 EfficientNet Pipeline...");
    C_EfficientNetPipelineConfig eff_config = c_efficientnet_pipeline_get_default_config();
    eff_config.engine_path = efficientnet_engine_path;

    C_EfficientNetPipelineContext* eff_pipeline = c_efficientnet_pipeline_create(&eff_config);
    if (!eff_pipeline) {
        LOG_ERROR("CascadeTest", "  错误: 无法创建 EfficientNet Pipeline");
        c_yolopose_image_result_free(&pose_result);
        c_yolopose_pipeline_destroy(pose_pipeline);
        return 1;
    }
    LOG_INFO("CascadeTest", "  Pipeline 创建成功");

    // ========================================================================
    // [5/6] 裁剪人物区域并进行分类
    // ========================================================================
    LOG_INFO("CascadeTest", "");
    LOG_INFO("CascadeTest", "[5/6] 裁剪人物区域并进行分类...");

    // 准备裁剪区域（用于性能测试）
    std::vector<cv::Mat> cropped_images;
    std::vector<C_ImageInput> crop_inputs;

    for (size_t i = 0; i < pose_result.num_poses; i++) {
        const C_YoloPose* pose = &pose_result.poses[i];

        int crop_lx = std::max(0, pose->detection.lx);
        int crop_ly = std::max(0, pose->detection.ly);
        int crop_rx = std::min(input_image_rgb.cols, pose->detection.rx);
        int crop_ry = std::min(input_image_rgb.rows, pose->detection.ry);

        cv::Rect crop_rect(crop_lx, crop_ly, crop_rx - crop_lx, crop_ry - crop_ly);
        cv::Mat cropped_rgb = input_image_rgb(crop_rect).clone();
        cropped_images.push_back(cropped_rgb);

        C_ImageInput crop_input;
        crop_input.data = cropped_rgb.data;
        crop_input.width = cropped_rgb.cols;
        crop_input.height = cropped_rgb.rows;
        crop_input.channels = cropped_rgb.channels();
        crop_inputs.push_back(crop_input);
    }

    // 如果是性能测试模式，对第一个裁剪区域进行测试
    if (benchmark_mode && !cropped_images.empty()) {
        LOG_INFO("CascadeTest", "");
        LOG_INFO("CascadeTest", ">>> EfficientNet 性能基准测试 <<<");
        LOG_INFO("CascadeTest", "");

        MemoryMonitor eff_mem_monitor;
        eff_mem_monitor.start();

        // 预热
        LOG_INFO("CascadeTest", "[预热] EfficientNet 执行 10 次推理...");
        for (int i = 0; i < 10; i++) {
            C_EfficientNetResult warmup_result = {0};
            c_efficientnet_infer_single(eff_pipeline, &crop_inputs[0], &warmup_result);
            c_efficientnet_result_free(&warmup_result);
            printf(".");
            fflush(stdout);
        }
        printf("\n");
        LOG_INFO("CascadeTest", " 完成");
        LOG_INFO("CascadeTest", "");

        // 性能测试
        LOG_INFO("CascadeTest", "[测试] EfficientNet 执行 100 次推理...");
        PerformanceStats eff_stats;

        size_t eff_initial_mem = eff_mem_monitor.get_current_usage_mb();

        for (int i = 0; i < 100; i++) {
            C_EfficientNetResult bench_result = {0};

            // 每次循环创建新的计时器实例
            PerformanceTimer eff_timer;
            eff_timer.start();
            c_efficientnet_infer_single(eff_pipeline, &crop_inputs[0], &bench_result);
            double elapsed = eff_timer.stop_ms();

            eff_stats.add(elapsed);
            c_efficientnet_result_free(&bench_result);

            if ((i + 1) % 20 == 0) {
                snprintf(buffer, sizeof(buffer), "  完成 %d/100 次...", i + 1);
                LOG_INFO("CascadeTest", buffer);
            }
        }

        size_t eff_final_mem = eff_mem_monitor.get_current_usage_mb();
        long eff_mem_leak = eff_final_mem - eff_initial_mem;

        // 输出性能统计
        LOG_INFO("CascadeTest", "");
        LOG_INFO("CascadeTest", "========================================");
        LOG_INFO("CascadeTest", "EfficientNet 性能统计 (100次)");
        LOG_INFO("CascadeTest", "========================================");
        snprintf(buffer, sizeof(buffer), "  平均耗时: %.3f ms", eff_stats.get_mean());
        LOG_INFO("CascadeTest", buffer);
        snprintf(buffer, sizeof(buffer), "  最小耗时: %.3f ms", eff_stats.get_min());
        LOG_INFO("CascadeTest", buffer);
        snprintf(buffer, sizeof(buffer), "  最大耗时: %.3f ms", eff_stats.get_max());
        LOG_INFO("CascadeTest", buffer);
        snprintf(buffer, sizeof(buffer), "  标准差:   %.3f ms", eff_stats.get_std());
        LOG_INFO("CascadeTest", buffer);
        snprintf(buffer, sizeof(buffer), "  吞吐量:   %.2f FPS", 1000.0 / eff_stats.get_mean());
        LOG_INFO("CascadeTest", buffer);
        LOG_INFO("CascadeTest", "");
        snprintf(buffer, sizeof(buffer), "[内存] 内存变化: %s%ld MB %s",
               eff_mem_leak > 0 ? "+" : "",
               eff_mem_leak,
               eff_mem_leak > 5 ? "(可能存在内存泄漏)" : "(正常)");
        LOG_INFO("CascadeTest", buffer);
        LOG_INFO("CascadeTest", "========================================");
        LOG_INFO("CascadeTest", "");
    }

    // 正常处理流程：对所有检测到的人进行分类
    for (size_t i = 0; i < pose_result.num_poses; i++) {
        LOG_INFO("CascadeTest", "");
        snprintf(buffer, sizeof(buffer), "--- Person %zu ---", i);
        LOG_INFO("CascadeTest", buffer);
        snprintf(buffer, sizeof(buffer), "  裁剪区域: %dx%d", cropped_images[i].cols, cropped_images[i].rows);
        LOG_INFO("CascadeTest", buffer);

        // 保存裁剪图像 (BGR 格式)
        if (!benchmark_mode) {  // 性能测试模式下不保存图像
            cv::Mat cropped_bgr;
            cv::cvtColor(cropped_images[i], cropped_bgr, cv::COLOR_RGB2BGR);
            char crop_filename[256];
            snprintf(crop_filename, sizeof(crop_filename), "output_crop_%zu.jpg", i);
            cv::imwrite(crop_filename, cropped_bgr);
            snprintf(buffer, sizeof(buffer), "  已保存裁剪图像: %s", crop_filename);
            LOG_INFO("CascadeTest", buffer);
        }

        // EfficientNet 推理
        C_EfficientNetResult eff_result = {0};
        if (!c_efficientnet_infer_single(eff_pipeline, &crop_inputs[i], &eff_result)) {
            snprintf(buffer, sizeof(buffer), "  错误: EfficientNet 推理失败: %s",
                   c_efficientnet_pipeline_get_last_error(eff_pipeline));
            LOG_ERROR("CascadeTest", buffer);
            continue;
        }

        // 输出分类结果
        LOG_INFO("CascadeTest", "  分类结果:");
        snprintf(buffer, sizeof(buffer), "    预测类别: %d", eff_result.class_id);
        LOG_INFO("CascadeTest", buffer);
        snprintf(buffer, sizeof(buffer), "    置信度: %.4f", eff_result.confidence);
        LOG_INFO("CascadeTest", buffer);

        if (!benchmark_mode) {  // 详细信息只在非性能测试模式下显示
            std::string logits_str = "    Logits: ";
            for (size_t j = 0; j < eff_result.num_classes; j++) {
                char val_buf[32];
                snprintf(val_buf, sizeof(val_buf), "%.4f ", eff_result.logits[j]);
                logits_str += val_buf;
            }
            LOG_INFO("CascadeTest", logits_str.c_str());

            // 输出特征向量信息
            LOG_INFO("CascadeTest", "  特征向量信息:");
            snprintf(buffer, sizeof(buffer), "    维度: %zu", eff_result.feature_size);
            LOG_INFO("CascadeTest", buffer);

            // 计算 L2 范数
            float l2_norm = 0.0f;
            for (size_t j = 0; j < eff_result.feature_size; j++) {
                l2_norm += eff_result.features[j] * eff_result.features[j];
            }
            l2_norm = sqrtf(l2_norm);
            snprintf(buffer, sizeof(buffer), "    L2 范数: %.4f", l2_norm);
            LOG_INFO("CascadeTest", buffer);

            // 输出前 10 个特征值
            std::string features_str = "    前 10 个特征值: ";
            for (size_t j = 0; j < 10 && j < eff_result.feature_size; j++) {
                char feat_buf[32];
                snprintf(feat_buf, sizeof(feat_buf), "%.4f ", eff_result.features[j]);
                features_str += feat_buf;
            }
            features_str += "...";
            LOG_INFO("CascadeTest", features_str.c_str());
        }

        // 清理结果
        c_efficientnet_result_free(&eff_result);
    }

    // ========================================================================
    // [6/6] 清理资源
    // ========================================================================
    LOG_INFO("CascadeTest", "");
    LOG_INFO("CascadeTest", "[6/6] 清理资源...");
    c_yolopose_image_result_free(&pose_result);
    c_yolopose_pipeline_destroy(pose_pipeline);
    c_efficientnet_pipeline_destroy(eff_pipeline);
    LOG_INFO("CascadeTest", "  完成");

    LOG_INFO("CascadeTest", "");
    LOG_INFO("CascadeTest", "========================================");
    LOG_INFO("CascadeTest", "处理完成！");
    LOG_INFO("CascadeTest", "========================================");
    LOG_INFO("CascadeTest", "输出文件:");
    LOG_INFO("CascadeTest", "  - output_pose.jpg       : 带关键点和骨架的图像");
    LOG_INFO("CascadeTest", "  - output_crop_*.jpg     : 裁剪的人物区域");
    LOG_INFO("CascadeTest", "========================================");

    return 0;
}
