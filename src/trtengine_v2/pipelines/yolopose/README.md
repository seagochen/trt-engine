# YOLOv8-Pose Pipeline

## 概述

这是一个基于 TrtEngine V2 架构的 YOLOv8-Pose 姿态检测管线，提供纯 C 接口用于人体姿态估计。该管线可以检测图像中的人物并输出 17 个 COCO 格式的关键点。

## 功能特性

- **纯 C API**: 无 C++/OpenCV 依赖的干净接口
- **完整的端到端推理**: 包含预处理、推理、后处理
- **批量推理**: 支持单张或批量图像处理
- **内置 NMS**: 自动过滤重叠检测框
- **Letterbox 预处理**: 保持宽高比的图像缩放
- **内存安全**: 完整的资源管理和错误处理

## 模型规格

### 输入
- **格式**: RGB图像
- **尺寸**: 640x640 (可配置)
- **预处理**:
  - Letterbox resize (保持宽高比)
  - 归一化到 [0, 1]
  - 灰色填充 (114/255)

### 输出
- **检测框**: 人物边界框 (lx, ly, rx, ry)
- **关键点**: 17个COCO格式关键点
  - 每个关键点: (x, y, confidence)
- **置信度**: 检测置信度分数

### COCO 关键点定义

17 个关键点按以下顺序排列：

```
0:  鼻子 (Nose)
1:  左眼 (Left Eye)
2:  右眼 (Right Eye)
3:  左耳 (Left Ear)
4:  右耳 (Right Ear)
5:  左肩 (Left Shoulder)
6:  右肩 (Right Shoulder)
7:  左肘 (Left Elbow)
8:  右肘 (Right Elbow)
9:  左腕 (Left Wrist)
10: 右腕 (Right Wrist)
11: 左髋 (Left Hip)
12: 右髋 (Right Hip)
13: 左膝 (Left Knee)
14: 右膝 (Right Knee)
15: 左踝 (Left Ankle)
16: 右踝 (Right Ankle)
```

## 使用方法

### 1. 基础使用 - 单张图片

```c
#include "trtengine_v2/pipelines/yolopose/c_yolopose_pipeline.h"

// 1. 获取默认配置
C_YoloPosePipelineConfig config = c_yolopose_pipeline_get_default_config();
config.engine_path = "/path/to/yolov8_pose.engine";
config.conf_threshold = 0.25f;  // 置信度阈值
config.iou_threshold = 0.45f;   // NMS IoU 阈值

// 2. 创建管线
C_YoloPosePipelineContext* pipeline = c_yolopose_pipeline_create(&config);
if (!pipeline) {
    printf("Failed to create pipeline!\n");
    return;
}

// 3. 准备输入图像
C_ImageInput image;
image.data = your_rgb_image_data;  // RGB格式
image.width = 1920;
image.height = 1080;
image.channels = 3;

// 4. 运行推理
C_YoloPoseImageResult result = {0};
if (!c_yolopose_infer_single(pipeline, &image, &result)) {
    printf("Inference failed: %s\n",
           c_yolopose_pipeline_get_last_error(pipeline));
    c_yolopose_pipeline_destroy(pipeline);
    return;
}

// 5. 处理检测结果
printf("Detected %zu person(s)\n", result.num_poses);

for (size_t i = 0; i < result.num_poses; i++) {
    C_YoloPose* pose = &result.poses[i];

    // 边界框
    printf("Person %zu: bbox=[%d,%d,%d,%d], conf=%.2f\n",
           i, pose->detection.lx, pose->detection.ly,
           pose->detection.rx, pose->detection.ry,
           pose->detection.conf);

    // 关键点
    for (int j = 0; j < 17; j++) {
        if (pose->pts[j].conf > 0.5f) {  // 只显示可信的关键点
            printf("  Keypoint %d: (%.1f, %.1f) conf=%.2f\n",
                   j, pose->pts[j].x, pose->pts[j].y, pose->pts[j].conf);
        }
    }
}

// 6. 清理资源
c_yolopose_image_result_free(&result);
c_yolopose_pipeline_destroy(pipeline);
```

### 2. 批量推理

```c
// 1. 配置更大的 batch size
C_YoloPosePipelineConfig config = c_yolopose_pipeline_get_default_config();
config.engine_path = "/path/to/yolov8_pose.engine";
config.max_batch_size = 4;  // 一次处理 4 张图片

C_YoloPosePipelineContext* pipeline = c_yolopose_pipeline_create(&config);

// 2. 准备图像批次
C_ImageBatch batch;
batch.count = 3;  // 实际图片数量
batch.images = (C_ImageInput*)malloc(batch.count * sizeof(C_ImageInput));

// 填充图像数据...
for (int i = 0; i < batch.count; i++) {
    batch.images[i].data = ...;
    batch.images[i].width = 1920;
    batch.images[i].height = 1080;
    batch.images[i].channels = 3;
}

// 3. 批量推理
C_YoloPoseBatchResult batch_result = {0};
if (!c_yolopose_infer_batch(pipeline, &batch, &batch_result)) {
    printf("Batch inference failed!\n");
}

// 4. 处理每张图片的结果
for (size_t i = 0; i < batch_result.num_images; i++) {
    C_YoloPoseImageResult* img_result = &batch_result.results[i];
    printf("Image %zu: detected %zu person(s)\n",
           i, img_result->num_poses);

    for (size_t j = 0; j < img_result->num_poses; j++) {
        // 处理每个检测到的人...
    }
}

// 5. 清理
c_yolopose_batch_result_free(&batch_result);
free(batch.images);
c_yolopose_pipeline_destroy(pipeline);
```

### 3. 实用工具函数

#### 检查关键点可见性

```c
// 判断某个关键点是否可见
bool is_keypoint_visible(const C_KeyPoint* kpt, float threshold) {
    return kpt->conf > threshold;
}

// 示例: 检查左肩是否可见
C_YoloPose* pose = &result.poses[0];
if (is_keypoint_visible(&pose->pts[5], 0.5f)) {  // 5 = 左肩
    printf("Left shoulder: (%.1f, %.1f)\n",
           pose->pts[5].x, pose->pts[5].y);
}
```

#### 计算两点距离

```c
// 计算两个关键点之间的欧氏距离
float calculate_distance(const C_KeyPoint* p1, const C_KeyPoint* p2) {
    float dx = p1->x - p2->x;
    float dy = p1->y - p2->y;
    return sqrtf(dx * dx + dy * dy);
}

// 示例: 计算肩宽
float shoulder_width = calculate_distance(
    &pose->pts[5],  // 左肩
    &pose->pts[6]   // 右肩
);
printf("Shoulder width: %.1f pixels\n", shoulder_width);
```

#### 判断姿势

```c
// 简单的姿势分类示例
void classify_pose(const C_YoloPose* pose) {
    // 检查手臂是否举起 (腕高于肩)
    bool left_arm_up = pose->pts[9].y < pose->pts[5].y &&   // 左腕高于左肩
                       pose->pts[9].conf > 0.5f &&
                       pose->pts[5].conf > 0.5f;

    bool right_arm_up = pose->pts[10].y < pose->pts[6].y &&  // 右腕高于右肩
                        pose->pts[10].conf > 0.5f &&
                        pose->pts[6].conf > 0.5f;

    if (left_arm_up && right_arm_up) {
        printf("Pose: Both arms raised\n");
    } else if (left_arm_up) {
        printf("Pose: Left arm raised\n");
    } else if (right_arm_up) {
        printf("Pose: Right arm raised\n");
    } else {
        printf("Pose: Normal standing\n");
    }
}
```

## 配置选项

```c
typedef struct {
    // 模型配置
    const char* engine_path;        // TensorRT 引擎路径
    int input_width;                // 输入宽度 (默认: 640)
    int input_height;               // 输入高度 (默认: 640)
    int max_batch_size;             // 最大批次大小 (默认: 1)

    // 检测阈值
    float conf_threshold;           // 置信度阈值 (默认: 0.25)
    float iou_threshold;            // NMS IoU阈值 (默认: 0.45)

    // 模型参数
    int num_keypoints;              // 关键点数量 (默认: 17)
    int num_classes;                // 类别数量 (默认: 1)
} C_YoloPosePipelineConfig;
```

## 数据结构

### C_KeyPoint - 关键点

```c
typedef struct {
    float x;        // X 坐标
    float y;        // Y 坐标
    float conf;     // 置信度 (0.0 - 1.0)
} C_KeyPoint;
```

### C_YoloDetect - 检测框

```c
typedef struct {
    int lx, ly, rx, ry;     // 边界框坐标 (左上, 右下)
    int cls;                // 类别索引 (通常为0: person)
    float conf;             // 检测置信度
} C_YoloDetect;
```

### C_YoloPose - 姿态检测结果

```c
typedef struct {
    C_YoloDetect detection;    // 边界框
    C_KeyPoint pts[17];        // 17个COCO关键点
} C_YoloPose;
```

## 性能优化建议

### 1. 批量处理
```c
// ✅ 推荐: 批量处理多张图片
config.max_batch_size = 8;
c_yolopose_infer_batch(pipeline, &batch, &results);

// ❌ 不推荐: 循环调用单张推理
for (int i = 0; i < 8; i++) {
    c_yolopose_infer_single(pipeline, &images[i], &results[i]);
}
```

### 2. 阈值调整
```c
// 快速检测 (可能有误检)
config.conf_threshold = 0.15f;
config.iou_threshold = 0.65f;

// 精确检测 (可能漏检)
config.conf_threshold = 0.45f;
config.iou_threshold = 0.35f;

// 平衡设置 (推荐)
config.conf_threshold = 0.25f;
config.iou_threshold = 0.45f;
```

### 3. 输入分辨率
```c
// 高精度 (慢)
config.input_width = 1280;
config.input_height = 1280;

// 标准 (推荐)
config.input_width = 640;
config.input_height = 640;

// 快速 (低精度)
config.input_width = 416;
config.input_height = 416;
```

## 应用场景

### 1. 健身应用
- 动作识别和计数 (深蹲、俯卧撑等)
- 姿势纠正
- 运动轨迹分析

### 2. 安防监控
- 异常行为检测 (跌倒、打架等)
- 人流统计
- 危险姿势识别

### 3. 人机交互
- 手势识别
- 虚拟试衣
- 体感游戏

### 4. 医疗健康
- 步态分析
- 康复训练监测
- 姿势评估

## 常见问题

### Q: 为什么检测不到人？
A: 检查以下几点：
1. 确认 `conf_threshold` 不要太高 (建议 0.25)
2. 确保输入图像是 RGB 格式 (不是 BGR)
3. 图像中的人物是否足够清晰
4. 检查模型是否正确加载

### Q: 关键点位置不准确？
A: 可能原因：
1. 输入分辨率太低，尝试使用 640x640 或更高
2. 人物被遮挡或姿势不常见
3. 光照条件差

### Q: 如何处理多人场景？
A: Pipeline 自动支持多人检测：
```c
for (size_t i = 0; i < result.num_poses; i++) {
    // 处理每个检测到的人
}
```

### Q: 支持自定义关键点数量吗？
A: 可以，修改配置：
```c
config.num_keypoints = 133;  // 如使用 WholeBody 模型
```

## 编译示例

```bash
# 方式1: 使用 CMake
cd /path/to/TrtEngineToolkits
cmake -B build -DBUILD_VERSION=v2
cmake --build build

# 方式2: 直接编译单个示例
gcc -o yolopose_example \
    examples/yolopose_pipeline_example.c \
    -I include \
    -L build \
    -ltrtengine_v2 \
    -lcudart \
    -lnvinfer \
    -lm

# 运行示例
./yolopose_example /path/to/yolov8_pose.engine /path/to/image.jpg
```

## 与其他版本的对比

| 特性 | V1 (C++) | V2 (C) | 说明 |
|------|---------|--------|------|
| API 语言 | C++ | Pure C | V2 更易于集成 |
| OpenCV 依赖 | 需要 | 不需要 | V2 无外部依赖 |
| 预处理 | CPU (OpenCV) | CPU (手工实现) | V2 性能相当 |
| NMS 实现 | C++ STL | Pure C | V2 更轻量 |
| 内存管理 | RAII | 手动管理 | V2 需要手动释放 |
| 跨语言调用 | 困难 | 简单 | V2 易于 FFI |

## 目录结构

```
pipelines/yolopose/
├── c_yolopose_structures.h      # 数据结构定义
├── c_yolopose_operations.c      # NMS 和 IoU 实现
├── c_yolopose_pipeline.h        # Pipeline API
├── c_yolopose_pipeline.c        # Pipeline 实现
└── README.md                    # 本文档
```

## 依赖项

- **TensorRT**: 推理引擎
- **CUDA**: GPU 加速
- **trtengine_v2/core**: TRT 引擎封装
- **trtengine_v2/common**: 通用数据结构和算法
- **trtengine_v2/tensor**: Tensor 操作
- **trtengine_v2/utils**: 日志工具

## 许可证

与主项目保持一致

## 作者
TrtEngineToolkits

## 更新日期
2025-11-10

## 参考资源

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [COCO 关键点定义](https://cocodataset.org/#keypoints-2020)
- [TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/)
