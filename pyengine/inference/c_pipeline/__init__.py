"""
C Pipeline Wrappers (V2 Architecture)

This package provides Python wrappers for the TrtEngineToolkits V2 C API.

Architecture:
- Decoupled pipelines: YoloPose and EfficientNet work independently
- Pure C API: Better cross-language compatibility
- CUDA-accelerated: High-performance post-processing

Main Classes:
- YoloPosePipelineV2: Standalone YOLOv8-Pose pipeline
- EfficientNetPipelineV2: Standalone EfficientNet pipeline

Data Converters:
- yolopose_to_skeletons: Convert YoloPose output to Skeleton objects
- efficientnet_to_classifications: Convert EfficientNet output to ClassificationResult objects
- cascade_results_to_unified: Merge cascade inference results

C Structures:
- c_structures_v2: Common C structure definitions
"""

# V2 Architecture pipelines
from .yolopose_pipeline_v2 import YoloPosePipelineV2
from .efficientnet_pipeline_v2 import EfficientNetPipelineV2

# Data converters
from .converter_v2 import (
    yolopose_to_skeletons,
    efficientnet_to_classifications,
    cascade_results_to_unified,
    pipeline_v2_to_skeletons,  # Backward compatibility alias
)

# C structure definitions
from .c_structures_v2 import (
    C_KeyPoint,
    C_YoloDetect,
    C_ImageInput,
    C_YoloPose,
    C_EfficientNetResult,
    YOLO_POSE_NUM_KEYPOINTS,
    EFFICIENTNET_DEFAULT_FEAT_SIZE,
    EFFICIENTNET_DEFAULT_NUM_CLASSES,
    EFFICIENTNET_DEFAULT_IMAGE_SIZE
)

__all__ = [
    # V2 Pipelines
    "YoloPosePipelineV2",
    "EfficientNetPipelineV2",

    # Converters
    "yolopose_to_skeletons",
    "efficientnet_to_classifications",
    "cascade_results_to_unified",
    "pipeline_v2_to_skeletons",

    # C structures
    "C_KeyPoint",
    "C_YoloDetect",
    "C_ImageInput",
    "C_YoloPose",
    "C_EfficientNetResult",
    "YOLO_POSE_NUM_KEYPOINTS",
    "EFFICIENTNET_DEFAULT_FEAT_SIZE",
    "EFFICIENTNET_DEFAULT_NUM_CLASSES",
    "EFFICIENTNET_DEFAULT_IMAGE_SIZE",
]
