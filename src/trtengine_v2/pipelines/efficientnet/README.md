# EfficientNet Pipeline

## 概述

这是一个基于 TrtEngine V2 架构的 EfficientNet 推理管线，提供纯 C 接口用于图像分类和特征提取。

## 功能特性

- **纯 C API**: 无 C++/OpenCV 依赖的干净接口
- **批量推理**: 支持单张或批量图像处理
- **双重输出**: 同时输出分类结果和特征向量
- **ImageNet 标准化**: 内置 ImageNet 预处理参数
- **内存安全**: 完整的资源管理和错误处理

## 模型规格

### 输入
- **格式**: RGB图像
- **尺寸**: 224x224 (可配置)
- **预处理**:
  - Bilinear resize
  - ImageNet normalization
    - Mean: [0.485, 0.456, 0.406]
    - Std: [0.229, 0.224, 0.225]

### 输出
1. **Logits** (分类输出)
   - 维度: `[batch_size, num_classes]`
   - 默认: 2 classes

2. **Features** (特征向量)
   - 维度: `[batch_size, feature_size]`
   - 默认: 256 维

## 使用方法

### 1. 基础使用

```c
#include "trtengine_v2/pipelines/efficientnet/c_efficientnet_pipeline.h"

// 1. 获取默认配置
C_EfficientNetPipelineConfig config = c_efficientnet_pipeline_get_default_config();
config.engine_path = "/path/to/efficientnet.engine";

// 2. 创建管线
C_EfficientNetPipelineContext* pipeline = c_efficientnet_pipeline_create(&config);

// 3. 准备输入图像
C_ImageInput image;
image.data = your_rgb_image_data;  // RGB格式
image.width = 640;
image.height = 480;
image.channels = 3;

// 4. 运行推理
C_EfficientNetResult result = {0};
c_efficientnet_infer_single(pipeline, &image, &result);

// 5. 使用结果
printf("Class: %d, Confidence: %.4f\n", result.class_id, result.confidence);
printf("Feature vector size: %zu\n", result.feature_size);

// 访问特征向量
for (size_t i = 0; i < result.feature_size; i++) {
    printf("Feature[%zu] = %.4f\n", i, result.features[i]);
}

// 6. 清理资源
c_efficientnet_result_free(&result);
c_efficientnet_pipeline_destroy(pipeline);
```

### 2. 批量推理

```c
// 1. 配置更大的 batch size
C_EfficientNetPipelineConfig config = c_efficientnet_pipeline_get_default_config();
config.engine_path = "/path/to/efficientnet.engine";
config.max_batch_size = 8;  // 一次处理 8 张图片

C_EfficientNetPipelineContext* pipeline = c_efficientnet_pipeline_create(&config);

// 2. 准备图像批次
C_ImageBatch batch;
batch.count = 4;  // 实际图片数量
batch.images = (C_ImageInput*)malloc(batch.count * sizeof(C_ImageInput));

// 填充图像数据...
for (int i = 0; i < batch.count; i++) {
    batch.images[i].data = ...;
    batch.images[i].width = 640;
    batch.images[i].height = 480;
    batch.images[i].channels = 3;
}

// 3. 批量推理
C_EfficientNetBatchResult results = {0};
c_efficientnet_infer_batch(pipeline, &batch, &results);

// 4. 处理每张图片的结果
for (size_t i = 0; i < results.count; i++) {
    printf("Image %zu: Class %d, Confidence %.4f\n",
           i, results.results[i].class_id, results.results[i].confidence);
}

// 5. 清理
c_efficientnet_batch_result_free(&results);
free(batch.images);
c_efficientnet_pipeline_destroy(pipeline);
```

### 3. 特征相似度计算

```c
// 提取两张图片的特征并计算余弦相似度
C_EfficientNetResult result1 = {0}, result2 = {0};

c_efficientnet_infer_single(pipeline, &image1, &result1);
c_efficientnet_infer_single(pipeline, &image2, &result2);

// 计算余弦相似度
float dot_product = 0.0f;
float norm1 = 0.0f, norm2 = 0.0f;

for (size_t i = 0; i < result1.feature_size; i++) {
    dot_product += result1.features[i] * result2.features[i];
    norm1 += result1.features[i] * result1.features[i];
    norm2 += result2.features[i] * result2.features[i];
}

float cosine_similarity = dot_product / (sqrt(norm1) * sqrt(norm2));
printf("Similarity: %.4f (1.0=identical, 0.0=orthogonal)\n", cosine_similarity);

c_efficientnet_result_free(&result1);
c_efficientnet_result_free(&result2);
```

## 配置选项

```c
typedef struct {
    const char* engine_path;        // TensorRT 引擎路径
    int input_width;                // 输入宽度 (默认: 224)
    int input_height;               // 输入高度 (默认: 224)
    int max_batch_size;             // 最大批次大小 (默认: 1)
    int num_classes;                // 分类数量 (默认: 2)
    int feature_size;               // 特征向量大小 (默认: 256)
    float mean[3];                  // RGB均值 (默认: ImageNet)
    float stddev[3];                // RGB标准差 (默认: ImageNet)
} C_EfficientNetPipelineConfig;
```

## 输出结构

```c
typedef struct {
    int class_id;                   // 预测类别 ID
    float confidence;               // 置信度 (最大 logit 值)
    float* logits;                  // 所有类别的 logits
    size_t num_classes;             // 类别数量
    float* features;                // 特征向量
    size_t feature_size;            // 特征向量大小
} C_EfficientNetResult;
```

## 编译示例

```bash
# 编译示例程序
gcc -o efficientnet_example \
    examples/efficientnet_pipeline_example.c \
    -I include \
    -L build \
    -ltrtengine_v2 \
    -lcudart \
    -lnvinfer \
    -lm

# 运行示例
./efficientnet_example /path/to/efficientnet.engine
```

## 应用场景

1. **图像分类**: 识别图片类别
2. **特征提取**: 生成图像的特征向量用于:
   - 图像检索
   - 相似度匹配
   - 聚类分析
3. **迁移学习**: 使用预训练特征进行下游任务

## 性能优化建议

1. **批量处理**: 尽可能使用批量推理以提高吞吐量
2. **预分配内存**: 对于重复推理，可以复用缓冲区
3. **异步处理**: 可以在多个线程中创建多个 pipeline 实例

## 常见问题

### Q: 如何处理非 RGB 图像？
A: Pipeline 要求 RGB 输入。如果是 BGR (如 OpenCV)，需要先转换。

### Q: 支持的最大 batch size 是多少？
A: 取决于模型构建时的配置和GPU内存，建议不超过 32。

### Q: 特征向量是否已归一化？
A: 特征向量是模型原始输出，如需归一化（如 L2 norm），需要自行处理。

## 作者
TrtEngineToolkits

## 更新日期
2025-11-10
