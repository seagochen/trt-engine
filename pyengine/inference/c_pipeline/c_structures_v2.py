"""
Common C structures for V2 architecture

This module defines ctypes structures that match the C API defined in:
- trtengine_v2/common/c_structures.h
- trtengine_v2/pipelines/yolopose/c_yolopose_structures.h
- trtengine_v2/pipelines/efficientnet/c_efficientnet_structures.h
"""
import ctypes


# ============================================================================
#                    Common Structures (c_structures.h)
# ============================================================================

class C_KeyPoint(ctypes.Structure):
    """Single keypoint with coordinates and confidence"""
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("conf", ctypes.c_float),
    ]


class C_YoloDetect(ctypes.Structure):
    """Basic YOLO detection bounding box"""
    _fields_ = [
        ("lx", ctypes.c_int),
        ("ly", ctypes.c_int),
        ("rx", ctypes.c_int),
        ("ry", ctypes.c_int),
        ("cls", ctypes.c_int),
        ("conf", ctypes.c_float),
    ]


class C_ImageInput(ctypes.Structure):
    """Input image structure"""
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("channels", ctypes.c_int),
    ]


class C_ImageBatch(ctypes.Structure):
    """Batch of input images"""
    _fields_ = [
        ("images", ctypes.POINTER(C_ImageInput)),
        ("count", ctypes.c_size_t),
    ]


# ============================================================================
#              YOLOv8-Pose Structures (c_yolopose_structures.h)
# ============================================================================

YOLO_POSE_NUM_KEYPOINTS = 17


class C_YoloPose(ctypes.Structure):
    """YOLO Pose detection result"""
    _fields_ = [
        ("detection", C_YoloDetect),
        ("pts", C_KeyPoint * YOLO_POSE_NUM_KEYPOINTS),
    ]


class C_YoloPoseImageResult(ctypes.Structure):
    """Single image pose detection result"""
    _fields_ = [
        ("image_index", ctypes.c_int),
        ("poses", ctypes.POINTER(C_YoloPose)),
        ("num_poses", ctypes.c_size_t),
    ]


class C_YoloPoseBatchResult(ctypes.Structure):
    """Batch pose detection results"""
    _fields_ = [
        ("results", ctypes.POINTER(C_YoloPoseImageResult)),
        ("num_images", ctypes.c_size_t),
    ]


class C_YoloPosePipelineConfig(ctypes.Structure):
    """YOLOv8-Pose pipeline configuration"""
    _fields_ = [
        ("engine_path", ctypes.c_char_p),
        ("input_width", ctypes.c_int),
        ("input_height", ctypes.c_int),
        ("max_batch_size", ctypes.c_int),
        ("conf_threshold", ctypes.c_float),
        ("iou_threshold", ctypes.c_float),
        ("num_keypoints", ctypes.c_int),
    ]


# ============================================================================
#            EfficientNet Structures (c_efficientnet_structures.h)
# ============================================================================

EFFICIENTNET_DEFAULT_FEAT_SIZE = 512
EFFICIENTNET_DEFAULT_NUM_CLASSES = 2
EFFICIENTNET_DEFAULT_IMAGE_SIZE = 224


class C_EfficientNetResult(ctypes.Structure):
    """EfficientNet classification result with feature vector"""
    _fields_ = [
        ("class_id", ctypes.c_int),
        ("confidence", ctypes.c_float),
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("num_classes", ctypes.c_size_t),
        ("features", ctypes.POINTER(ctypes.c_float)),
        ("feature_size", ctypes.c_size_t),
    ]


class C_EfficientNetBatchResult(ctypes.Structure):
    """Batch of EfficientNet results"""
    _fields_ = [
        ("results", ctypes.POINTER(C_EfficientNetResult)),
        ("count", ctypes.c_size_t),
    ]


class C_EfficientNetPipelineConfig(ctypes.Structure):
    """EfficientNet pipeline configuration"""
    _fields_ = [
        ("engine_path", ctypes.c_char_p),
        ("input_width", ctypes.c_int),
        ("input_height", ctypes.c_int),
        ("max_batch_size", ctypes.c_int),
        ("num_classes", ctypes.c_int),
        ("feature_size", ctypes.c_int),
        ("mean", ctypes.c_float * 3),
        ("stddev", ctypes.c_float * 3),
    ]
