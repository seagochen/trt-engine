# TrtEngineToolkits

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Jetson-green.svg)](https://developer.nvidia.com/embedded/jetson-developer-kits)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.0%2B-76B900.svg)](https://developer.nvidia.com/tensorrt)

## 📋 目录

- [概述](#概述)
- [核心特性](#核心特性)
- [架构说明](#架构说明)
  - [V2 架构](#v2-架构-推荐)
  - [目录结构](#目录结构)
- [快速开始](#快速开始)
  - [系统要求](#系统要求)
  - [安装依赖](#安装依赖)
  - [编译项目](#编译项目)
- [使用指南](#使用指南)
  - [C API 使用](#c-api-使用)
  - [Python API 使用](#python-api-使用)
- [Python V2 架构](#python-v2-架构)
  - [数据结构](#数据结构)
  - [转换器](#转换器)
  - [从 V1 迁移](#从-v1-迁移到-v2)
- [示例程序](#示例程序)
- [性能基准](#性能基准)
- [常见问题](#常见问题)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 概述

TrtEngineToolkits 是一个基于 NVIDIA TensorRT 的高性能推理工具包，提供易用的 C/C++ 和 Python API 用于部署深度学习模型。项目支持多种视觉模型（YOLOv8-Pose、EfficientNet 等），并针对 x86 和 Jetson 平台进行了优化。

### 核心特性

- 🚀 **高性能推理**:
  - 基于 TensorRT 的 GPU 加速
  - CUDA 核函数优化后处理（5-10x 加速）
  - 支持批量推理和多流并发

- 🎯 **多模型支持**:
  - YOLOv8-Pose（姿态检测、关键点提取）
  - EfficientNet（分类、特征提取）
  - 易于扩展新模型

- 🔧 **V2 架构**（推荐）:
  - 纯 C API，无外部依赖
  - 模型解耦，可独立使用
  - 更好的跨语言兼容性

- 🌐 **跨平台支持**:
  - x86_64 (Ubuntu, CentOS)
  - ARM64 (Jetson Nano, Xavier, Orin)

- 🐍 **Python 支持**:
  - 完整的 Python 封装（pyengine）
  - 数据转换器和可视化工具
  - 算法模块（Tracker, Filter, Estimation）

---

## 架构说明

### V2 架构 (推荐)

V2 架构是完全重写的版本，具有以下优势：

#### 核心设计原则

1. **解耦设计**: YoloPose 和 EfficientNet 完全独立，可按需使用
2. **纯 C API**: 无 C++ 依赖，易于跨语言调用
3. **CUDA 加速**: 后处理使用 SimpleCudaToolkits 核函数
4. **清晰的数据所有权**: features 只属于 ClassificationResult

#### 架构对比

| 特性 | V1 (已废弃) | V2 (推荐) |
|------|-------------|-----------|
| API 语言 | C++ | Pure C |
| 模型耦合 | 强制捆绑 | 完全解耦 |
| 后处理 | CPU | CUDA 加速 |
| OpenCV 依赖 | 需要 | 不需要 |
| Python FFI | 复杂 | 简单 |
| 内存占用 | 高 | 低 |
| 维护成本 | 高 | 低 |

#### 数据流程

```
V1 架构（已废弃）:
┌─────────────────────────────────────┐
│     PosePipelineV2 (耦合)           │
│  ┌───────────┬─────────────────┐    │
│  │ YoloPose  │  EfficientNet   │    │
│  └───────────┴─────────────────┘    │
│            ↓                         │
│    Skeleton.features = [...]        │
└─────────────────────────────────────┘

V2 架构（推荐）:
┌──────────────────┐  ┌─────────────────────┐
│ YoloPosePipelineV2│  │EfficientNetPipelineV2│
│    (独立)         │  │      (独立)          │
└────────┬─────────┘  └──────────┬──────────┘
         ↓                        ↓
    Skeleton              ClassificationResult
  (bbox + keypoints)    (class + features)
```

### 目录结构

```
TrtEngineToolkits/
├── include/                          # 头文件
│   └── trtengine_v2/                # V2 纯 C API
│       ├── core/                    # TensorRT 引擎核心
│       │   └── trt_engine_multi.h   # 多输入输出引擎
│       ├── common/                  # 通用数据结构
│       │   └── c_structures.h       # 基础结构定义
│       ├── pipelines/               # 模型推理管线
│       │   ├── yolopose/           # YOLOv8-Pose
│       │   │   ├── c_yolopose_pipeline.h
│       │   │   └── c_yolopose_structures.h
│       │   └── efficientnet/       # EfficientNet
│       │       ├── c_efficientnet_pipeline.h
│       │       └── c_efficientnet_structures.h
│       └── utils/                   # 工具函数
│
├── src/                             # 源代码
│   └── trtengine_v2/               # V2 实现
│       ├── core/                   # 引擎实现
│       ├── common/                 # 通用实现
│       └── pipelines/              # 模型实现
│
├── pyengine/                        # Python 封装
│   ├── inference/                  # 推理模块
│   │   ├── c_pipeline/            # C API 封装
│   │   │   ├── yolopose_pipeline_v2.py
│   │   │   ├── efficientnet_pipeline_v2.py
│   │   │   ├── converter_v2.py    # 数据转换器
│   │   │   └── c_structures_v2.py # C 结构体定义
│   │   └── unified_structs/       # 统一数据结构
│   │       ├── inference_results.py
│   │       └── auxiliary_structs.py
│   ├── algorithms/                 # 算法模块
│   │   ├── tracker/               # 目标跟踪
│   │   ├── filters/               # 信号滤波
│   │   └── estimation/            # 姿态估计
│   ├── visualization/              # 可视化工具
│   └── utils/                      # 工具函数
│
├── examples_v2_python/              # Python 示例
│   ├── yolopose_standalone_example.py
│   ├── efficientnet_standalone_example.py
│   ├── cascade_example.py
│   └── README.md
│
├── config/                          # 配置文件
├── scripts/                         # 构建脚本
└── CMakeLists.txt                  # CMake 配置
```

---

## 快速开始

### 系统要求

#### 硬件要求
- NVIDIA GPU (计算能力 >= 6.0)
- 或 NVIDIA Jetson 开发板 (Nano, Xavier, Orin 等)

#### 软件依赖

**必须**:
- CMake >= 3.16
- GCC >= 9.0 或 Clang >= 10.0
- CUDA >= 11.0
- TensorRT >= 8.0
- [SimpleCudaToolkits](https://github.com/seagochen/SimpleCudaToolkits) (需安装到 `/opt/SimpleCudaToolkits`)

**可选**:
- Python >= 3.8 (使用 Python API)
- NumPy, OpenCV-Python (Python 可视化)

### 安装依赖

#### Ubuntu / Jetson

```bash
# 安装基础工具
sudo apt update
sudo apt install -y build-essential cmake git

# 安装 CUDA (如未安装)
# 参考: https://developer.nvidia.com/cuda-downloads

# 安装 TensorRT (如未安装)
# 参考: https://developer.nvidia.com/tensorrt

# 安装 SimpleCudaToolkits
git clone https://github.com/seagochen/SimpleCudaToolkits
cd SimpleCudaToolkits
sudo cp -r include/simple_cuda_toolkits /opt/SimpleCudaToolkits/include/
sudo cp -r lib/* /opt/SimpleCudaToolkits/lib/
```

### 编译项目

```bash
# 克隆仓库
git clone https://github.com/your-org/TrtEngineToolkits.git
cd TrtEngineToolkits

# 配置并编译 (V2 架构)
cmake -B build -DBUILD_V2=ON
cmake --build build -j$(nproc)

# 编译结果
# build/libtrtengine_v2.so          - 动态库
# build/functional_test_v2_cascade  - 级联测试程序
```

### 运行示例

```bash
# 设置库路径
export LD_LIBRARY_PATH=$(pwd)/build:$LD_LIBRARY_PATH

# 运行级联测试（YoloPose + EfficientNet）
./build/functional_test_v2_cascade \
    /path/to/yolov8n-pose.engine \
    /path/to/efficientnet_b0.engine \
    /path/to/test_image.jpg

# 性能测试模式
./build/functional_test_v2_cascade \
    /path/to/yolov8n-pose.engine \
    /path/to/efficientnet_b0.engine \
    /path/to/test_image.jpg \
    --benchmark
```

---

## 使用指南

### C API 使用

#### YOLOv8-Pose 姿态检测

```c
#include "trtengine_v2/pipelines/yolopose/c_yolopose_pipeline.h"

// 1. 创建配置
C_YoloPosePipelineConfig config = c_yolopose_pipeline_get_default_config();
config.engine_path = "/path/to/yolov8_pose.engine";
config.conf_threshold = 0.25f;
config.iou_threshold = 0.45f;

// 2. 创建 pipeline
C_YoloPosePipelineContext* pipeline = c_yolopose_pipeline_create(&config);

// 3. 准备输入图像 (RGB 格式)
C_ImageInput image = {
    .data = your_rgb_data,
    .width = 1920,
    .height = 1080,
    .channels = 3
};

// 4. 执行推理
C_YoloPoseImageResult result = {0};
c_yolopose_infer_single(pipeline, &image, &result);

// 5. 处理结果
printf("检测到 %zu 个人\n", result.num_poses);
for (size_t i = 0; i < result.num_poses; i++) {
    C_YoloPose* pose = &result.poses[i];
    // 访问 bbox
    printf("BBox: [%d,%d,%d,%d], conf=%.2f\n",
           pose->detection.lx, pose->detection.ly,
           pose->detection.rx, pose->detection.ry,
           pose->detection.conf);

    // 访问 17 个关键点
    for (int j = 0; j < 17; j++) {
        if (pose->pts[j].conf > 0.5f) {
            printf("  Keypoint %d: (%.1f, %.1f)\n",
                   j, pose->pts[j].x, pose->pts[j].y);
        }
    }
}

// 6. 清理资源
c_yolopose_image_result_free(&result);
c_yolopose_pipeline_destroy(pipeline);
```

#### EfficientNet 分类和特征提取

```c
#include "trtengine_v2/pipelines/efficientnet/c_efficientnet_pipeline.h"

// 1. 创建配置
C_EfficientNetPipelineConfig config = c_efficientnet_pipeline_get_default_config();
config.engine_path = "/path/to/efficientnet.engine";
config.num_classes = 2;
config.feature_size = 512;

// 2. 创建 pipeline
C_EfficientNetPipelineContext* pipeline = c_efficientnet_pipeline_create(&config);

// 3. 准备输入
C_ImageInput image = {
    .data = your_rgb_data,
    .width = 224,
    .height = 224,
    .channels = 3
};

// 4. 执行推理
C_EfficientNetResult result = {0};
c_efficientnet_infer_single(pipeline, &image, &result);

// 5. 获取结果
printf("预测类别: %d\n", result.class_id);
printf("置信度: %.4f\n", result.confidence);

// 6. 获取特征向量 (512 维)
for (size_t i = 0; i < 10 && i < result.feature_size; i++) {
    printf("Feature[%zu]: %.4f\n", i, result.features[i]);
}

// 7. 清理
c_efficientnet_result_free(&result);
c_efficientnet_pipeline_destroy(pipeline);
```

### Python API 使用

#### 独立使用 YoloPose

```python
from pyengine.inference.c_pipeline import (
    YoloPosePipelineV2,
    yolopose_to_skeletons
)
import cv2

# 1. 创建 pipeline
pipeline = YoloPosePipelineV2(
    library_path="build/libtrtengine_v2.so",
    engine_path="yolov8n-pose.engine",
    conf_threshold=0.25,
    iou_threshold=0.45
)
pipeline.create()

# 2. 加载图像
image_bgr = cv2.imread("test.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 3. 推理
results = pipeline.infer([image_rgb])

# 4. 转换为 Skeleton 对象
from pyengine.inference.c_pipeline import yolopose_to_skeletons
skeletons_per_image = yolopose_to_skeletons(results)

# 5. 处理结果
for skeletons in skeletons_per_image:
    for skeleton in skeletons:
        print(f"BBox: {skeleton.rect}")
        print(f"Confidence: {skeleton.confidence}")
        print(f"Keypoints: {len(skeleton.points)}")

# 6. 清理
pipeline.close()
```

#### 独立使用 EfficientNet

```python
from pyengine.inference.c_pipeline import (
    EfficientNetPipelineV2,
    efficientnet_to_classifications
)

# 1. 创建 pipeline
pipeline = EfficientNetPipelineV2(
    library_path="build/libtrtengine_v2.so",
    engine_path="efficientnet_b0.engine",
    num_classes=2,
    feature_size=512
)
pipeline.create()

# 2. 推理
results = pipeline.infer([image_rgb])

# 3. 转换为 ClassificationResult
classifications = efficientnet_to_classifications(results)

# 4. 使用结果
for cls_result in classifications:
    print(f"Class: {cls_result.class_id}")
    print(f"Confidence: {cls_result.confidence}")
    print(f"Features shape: {len(cls_result.features)}")

# 5. 清理
pipeline.close()
```

#### 级联使用（检测 + 分类）

```python
from pyengine.inference.c_pipeline import (
    YoloPosePipelineV2,
    EfficientNetPipelineV2,
    yolopose_to_skeletons,
    efficientnet_to_classifications
)

# 1. 创建两个独立的 pipeline
yolo = YoloPosePipelineV2(library_path="...", engine_path="yolo.engine")
eff = EfficientNetPipelineV2(library_path="...", engine_path="eff.engine")
yolo.create()
eff.create()

# 2. YoloPose 检测
yolo_results = yolo.infer([image])
skeletons = yolopose_to_skeletons(yolo_results)[0]

# 3. 对每个检测进行分类
for skeleton in skeletons:
    bbox = skeleton.rect
    crop = image[int(bbox.y1):int(bbox.y2), int(bbox.x1):int(bbox.x2)]

    eff_results = eff.infer([crop])
    classifications = efficientnet_to_classifications(eff_results)

    print(f"Person: class={classifications[0].class_id}")
    print(f"  Features: {classifications[0].features[:10]}")

# 4. 清理
yolo.close()
eff.close()
```

---

## Python V2 架构

### 数据结构

#### 核心结构

```python
from pyengine.inference.unified_structs import (
    Skeleton,              # 姿态检测结果
    ClassificationResult,  # 分类结果（新增）
    ExpandedSkeleton,      # 扩展姿态（含姿态分析）
)

# Skeleton (姿态检测)
@dataclass
class Skeleton:
    rect: Rect                  # 边界框
    classification: int         # 类别
    confidence: float           # 置信度
    track_id: int              # 跟踪 ID
    points: List[Point]        # 17 个关键点
    # 注意：V2 中不再包含 features

# ClassificationResult (分类结果)
@dataclass
class ClassificationResult:
    class_id: int              # 预测类别
    confidence: float          # 置信度
    logits: List[float]        # 所有类别的 logits
    features: List[float]      # 特征向量（512-dim）
```

### 转换器

```python
from pyengine.inference.c_pipeline import (
    yolopose_to_skeletons,           # YoloPose → Skeleton
    efficientnet_to_classifications, # EfficientNet → ClassificationResult
    cascade_results_to_unified,      # 合并级联结果
)

# 使用示例
yolo_results = yolo_pipeline.infer([image])
skeletons = yolopose_to_skeletons(yolo_results)

eff_results = eff_pipeline.infer([crop])
classifications = efficientnet_to_classifications(eff_results)
```

### 从 V1 迁移到 V2

#### 主要变化

| 方面 | V1 | V2 |
|------|----|----|
| API | `PosePipelineV2` (耦合) | `YoloPosePipelineV2` + `EfficientNetPipelineV2` (解耦) |
| features 位置 | `Skeleton.features` | `ClassificationResult.features` |
| 转换器 | `pipeline_v1_to_skeletons()` | `yolopose_to_skeletons()` + `efficientnet_to_classifications()` |

#### 迁移步骤

**场景 1: 只使用姿态检测**

```python
# V1 (已废弃)
pipeline = PosePipelineV2(yolo_engine, eff_engine, ...)
results = pipeline.infer([image])

# V2 (推荐)
pipeline = YoloPosePipelineV2(library_path, yolo_engine, ...)
pipeline.create()
results = pipeline.infer([image])
skeletons = yolopose_to_skeletons(results)
```

**场景 2: 级联使用（检测 + 分类）**

```python
# V1 (已废弃)
pipeline = PosePipelineV2(yolo_engine, eff_engine, ...)
results = pipeline.infer([image])
# skeleton.features 可用

# V2 (推荐)
yolo = YoloPosePipelineV2(...)
eff = EfficientNetPipelineV2(...)
# 手动管理级联
yolo_results = yolo.infer([image])
skeletons = yolopose_to_skeletons(yolo_results)
for skeleton in skeletons:
    crop = extract_crop(image, skeleton.rect)
    eff_results = eff.infer([crop])
    classifications = efficientnet_to_classifications(eff_results)
    # classifications[0].features 可用
```

**场景 3: 使用 Tracker**

```python
# V1 (已废弃)
track = UnifiedTrack(detection, use_reid=True)
track.update(new_detection)  # features 自动提取

# V2 (推荐)
track = UnifiedTrack(skeleton, use_reid=True)
track.update(new_skeleton, features=cls_result.features if cls_result else None)
```

#### 向后兼容性

✅ **完全兼容**:
- Visualization 模块 (`InferenceDrawer`)
- Filters 模块 (`apply_savgol_filter_1d` 等)
- Estimation 模块 (`calculate_direction_and_posture`)

⚠️ **需要修改**:
- Tracker 的 `update()` 方法（新增可选 `features` 参数）
- 访问 `skeleton.features` 的代码（改为从 `ClassificationResult.features` 获取）

---

## 示例程序

### C 示例

所有 C 示例位于 `functional_test_v2_cascade.cpp`：

```bash
# 正常模式（运行一次）
./build/functional_test_v2_cascade \
    yolov8n-pose.engine \
    efficientnet_b0.engine \
    test_image.jpg

# 性能测试模式（100次迭代）
./build/functional_test_v2_cascade \
    yolov8n-pose.engine \
    efficientnet_b0.engine \
    test_image.jpg \
    --benchmark
```

### Python 示例

所有 Python 示例位于 `examples_v2_python/`：

```bash
# YoloPose 独立使用
python examples_v2_python/yolopose_standalone_example.py \
    build/libtrtengine_v2.so \
    yolov8n-pose.engine \
    test_image.jpg

# EfficientNet 独立使用
python examples_v2_python/efficientnet_standalone_example.py \
    build/libtrtengine_v2.so \
    efficientnet_b0.engine \
    test_image.jpg

# 级联使用
python examples_v2_python/cascade_example.py \
    build/libtrtengine_v2.so \
    yolov8n-pose.engine \
    efficientnet_b0.engine \
    test_image.jpg
```

---

## 性能基准

### NVIDIA Jetson Orin Nano

| 模型 | 输入尺寸 | Batch | FP16 | 后处理 | 延迟 (ms) | FPS |
|------|---------|-------|------|--------|-----------|-----|
| YOLOv8n-Pose | 640x640 | 1 | ✓ | CUDA | 15 | 66 |
| YOLOv8n-Pose | 640x640 | 4 | ✓ | CUDA | 45 | 89 |
| EfficientNet-B0 | 224x224 | 1 | ✓ | - | 3 | 333 |
| EfficientNet-B0 | 224x224 | 8 | ✓ | - | 18 | 444 |

### NVIDIA RTX 3090

| 模型 | 输入尺寸 | Batch | FP16 | 后处理 | 延迟 (ms) | FPS |
|------|---------|-------|------|--------|-----------|-----|
| YOLOv8n-Pose | 640x640 | 1 | ✓ | CUDA | 2.5 | 400 |
| YOLOv8n-Pose | 640x640 | 16 | ✓ | CUDA | 25 | 640 |
| EfficientNet-B0 | 224x224 | 1 | ✓ | - | 0.8 | 1250 |
| EfficientNet-B0 | 224x224 | 32 | ✓ | - | 15 | 2133 |

### 性能优化建议

1. **批量推理**: 增加 `max_batch_size` 提升吞吐量
2. **CUDA 加速**: V2 的 YoloPose 后处理使用 CUDA（5-10x 加速）
3. **输入尺寸**: 根据精度需求调整（416/640/1280）
4. **多流推理**: 创建多个 pipeline 实例并行推理

---

## 常见问题

### Q: 编译时找不到 SimpleCudaToolkits？
**A**: 确保已安装到 `/opt/SimpleCudaToolkits`，或修改 CMakeLists.txt 中的路径：
```cmake
set(SimpleCudaToolkits_DIR "/your/custom/path")
```

### Q: 运行时提示找不到 libtrtengine_v2.so？
**A**: 设置库路径：
```bash
export LD_LIBRARY_PATH=/path/to/build:$LD_LIBRARY_PATH
```

### Q: Jetson 上性能不佳？
**A**: 确保使用了最大性能模式：
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Q: 如何转换自己的模型？
**A**: 使用 `trtexec` 转换 ONNX 模型：
```bash
trtexec --onnx=model.onnx \
        --saveEngine=model.engine \
        --fp16 \
        --workspace=4096
```

### Q: Python 找不到 pyengine 模块？
**A**: 将项目根目录添加到 PYTHONPATH：
```bash
export PYTHONPATH=/path/to/TrtEngineToolkits:$PYTHONPATH
```

### Q: V1 代码如何迁移到 V2？
**A**: 参考本文档的 [从 V1 迁移到 V2](#从-v1-迁移到-v2) 章节，主要变化：
- 替换 API（`PosePipelineV2` → `YoloPosePipelineV2` + `EfficientNetPipelineV2`）
- 更新 `skeleton.features` 访问（改用 `ClassificationResult.features`）
- Tracker 的 `update()` 方法需要传入 `features` 参数

---

## 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范

- C 代码遵循 Linux Kernel 风格
- Python 代码遵循 PEP 8
- 所有公开 API 必须有详细注释
- 添加新功能需要包含示例和测试

---

## 许可证

GNU GENERAL PUBLIC LICENSE  - 详见 [LICENSE](LICENSE) 文件

---

## 致谢

- NVIDIA TensorRT 团队
- Ultralytics (YOLOv8)
- SimpleCudaToolkits 项目

---

## 更新日志

### Version 2.0.0 (2025-11-11)
- ✨ 完整的 V2 架构实现
- ✨ 模型解耦：YoloPose 和 EfficientNet 独立
- ✨ CUDA 加速后处理（5-10x 提升）
- ✨ 纯 C API，无外部依赖
- ✨ Python V2 封装和转换器
- ✨ 完整的迁移指南和示例
- 🗑️ 移除 V1 代码
- 📝 统一文档结构

### Version 1.0.0
- 🎉 初始版本发布
- ✅ 支持 YOLO 系列模型
- ✅ 支持 Jetson 和 x86 平台

