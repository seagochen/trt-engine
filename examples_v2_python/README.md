# Python Examples for V2 Architecture

This directory contains Python examples demonstrating how to use the V2 architecture's standalone pipelines.

## Overview

The V2 architecture provides **decoupled** pipelines that can be used independently:

- **YoloPosePipelineV2**: Standalone YOLOv8-Pose inference
- **EfficientNetPipelineV2**: Standalone EfficientNet classification and feature extraction

Unlike V1 (which coupled YOLOv8-Pose and EfficientNet together), V2 allows you to:
- Use only YOLOv8-Pose for pose detection
- Use only EfficientNet for classification
- Combine them in your own custom workflows

## Prerequisites

### 1. Build the Library

```bash
cd /home/user/projects/TrtEngineToolkits
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

This generates `libtrtengine_v2.so` in the `build/` directory.

### 2. Prepare TensorRT Engines

You need to convert ONNX models to TensorRT engines:

```bash
# YOLOv8-Pose (640x640)
trtexec --onnx=yolov8n-pose.onnx \
        --saveEngine=yolov8n-pose.engine \
        --fp16

# EfficientNet-B0 (224x224)
trtexec --onnx=efficientnet_b0.onnx \
        --saveEngine=efficientnet_b0.engine \
        --fp16
```

### 3. Python Dependencies

```bash
pip install numpy opencv-python
```

## Examples

### 1. YOLOv8-Pose Standalone

Run pose detection only:

```bash
python yolopose_standalone_example.py \
    ../build/libtrtengine_v2.so \
    yolov8n-pose.engine \
    test_image.jpg
```

**Output:**
- Detection results printed to console
- `output_yolopose_standalone.jpg`: Visualization with bounding boxes and keypoints

### 2. EfficientNet Standalone

Run classification and feature extraction:

```bash
python efficientnet_standalone_example.py \
    ../build/libtrtengine_v2.so \
    efficientnet_b0.engine \
    test_image.jpg
```

**Output:**
- Classification results (class ID, confidence)
- Feature embeddings (512-dim vector)
- Logits for all classes

### 3. Cascade Pipeline

Combine both models in a cascade workflow:

```bash
python cascade_example.py \
    ../build/libtrtengine_v2.so \
    yolov8n-pose.engine \
    efficientnet_b0.engine \
    test_image.jpg
```

**Workflow:**
1. YOLOv8-Pose detects people
2. Crop detected regions
3. EfficientNet classifies each person

**Output:**
- Detection and classification results
- `output_cascade.jpg`: Visualization with both pose and classification info

## API Usage

### YoloPosePipelineV2

```python
from pyengine.inference.c_pipeline import YoloPosePipelineV2

# Create pipeline
pipeline = YoloPosePipelineV2(
    library_path="libtrtengine_v2.so",
    engine_path="yolov8n-pose.engine",
    input_width=640,
    input_height=640,
    max_batch_size=1,
    conf_threshold=0.25,
    iou_threshold=0.45
)

# Initialize
pipeline.create()

# Inference (accepts list of RGB numpy arrays)
results = pipeline.infer([image_rgb])

# Results format:
# [
#     {
#         "image_idx": 0,
#         "detections": [
#             {
#                 "bbox": [lx, ly, rx, ry],
#                 "cls": 0,  # person class
#                 "conf": 0.95,
#                 "keypoints": [
#                     {"x": 100.0, "y": 50.0, "conf": 0.9},
#                     ...  # 17 keypoints total
#                 ]
#             }
#         ]
#     }
# ]

# Cleanup
pipeline.close()
```

### EfficientNetPipelineV2

```python
from pyengine.inference.c_pipeline import EfficientNetPipelineV2

# Create pipeline
pipeline = EfficientNetPipelineV2(
    library_path="libtrtengine_v2.so",
    engine_path="efficientnet_b0.engine",
    input_width=224,
    input_height=224,
    max_batch_size=1,
    num_classes=2,
    feature_size=512
)

# Initialize
pipeline.create()

# Inference
results = pipeline.infer([image_rgb])

# Results format:
# [
#     {
#         "image_idx": 0,
#         "class_id": 1,
#         "confidence": 0.87,
#         "logits": array([0.2, 0.8]),  # numpy array
#         "features": array([...])       # 512-dim numpy array
#     }
# ]

# Cleanup
pipeline.close()
```

### Context Manager Support

Both pipelines support Python context managers:

```python
with YoloPosePipelineV2(...) as pipeline:
    results = pipeline.infer([image])
    # Automatic cleanup on exit
```

## Key Differences from V1

| Feature | V1 (Legacy) | V2 (New) |
|---------|-------------|----------|
| Architecture | Coupled YOLO + EfficientNet | Standalone pipelines |
| Flexibility | Must use both models | Use independently |
| API | `PosePipelineV2` | `YoloPosePipelineV2` + `EfficientNetPipelineV2` |
| Post-processing | CPU-based | CUDA-accelerated |
| Memory | Higher (coupled) | Lower (independent) |

## Performance Notes

1. **Model Warmup**: The first inference is slower due to CUDA initialization. Run warmup iterations for benchmarking.

2. **Batch Size**: The current examples use batch size 1. For better throughput, increase `max_batch_size`.

3. **CUDA Acceleration**: V2 uses CUDA kernels for YOLOv8-Pose post-processing (transpose, filter, sort), providing significant speedup over CPU.

4. **Image Preprocessing**: Images are automatically resized to model input size. For best performance, resize images beforehand.

## Troubleshooting

### Library Loading Errors

If you get "library not found" errors:

```bash
export LD_LIBRARY_PATH=/path/to/build:$LD_LIBRARY_PATH
```

### CUDA Errors

Ensure you have:
- CUDA toolkit installed
- TensorRT installed
- Compatible GPU driver

### Engine Loading Errors

Verify:
- Engine file exists and is valid
- Engine matches the model architecture
- Engine was built for your GPU architecture

## Directory Structure

```
examples_v2_python/
├── README.md                           # This file
├── yolopose_standalone_example.py      # YOLOv8-Pose only
├── efficientnet_standalone_example.py  # EfficientNet only
└── cascade_example.py                  # Combined workflow
```

## Further Reading

- [V2 Architecture Documentation](../docs/v2_architecture.md)
- [C API Reference](../include/trtengine_v2/)
- [Python API Reference](../pyengine/inference/c_pipeline/)
