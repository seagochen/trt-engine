"""
Unified Data Structures for Inference Results

This package provides standardized data structures for representing
inference results from various models (YoloPose, EfficientNet, etc.)

Main classes:
- Rect: Bounding box representation
- Point: Keypoint with coordinates and confidence
- ObjectDetection: Base class for detections
- Skeleton: Pose detection with keypoints
- ClassificationResult: Classification with features (V2)
- ExpandedSkeleton: Skeleton with posture and direction analysis
"""

# Core inference result structures
from .inference_results import (
    InferenceResults,
    Rect,
    Point,
    ObjectDetection,
    Skeleton,
    ClassificationResult,  # New in V2
)

# Auxiliary structures for expanded analysis
from .auxiliary_structs import (
    FaceDirection,
    Pose,
    BodyOrientation,
    ExpandedSkeleton,
)

__all__ = [
    # Base classes
    "InferenceResults",
    "Rect",
    "Point",

    # Detection structures
    "ObjectDetection",
    "Skeleton",
    "ClassificationResult",

    # Expanded analysis
    "FaceDirection",
    "Pose",
    "BodyOrientation",
    "ExpandedSkeleton",
]
