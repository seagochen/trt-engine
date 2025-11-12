"""
YOLOv8-Pose Pipeline V2 - Pure C API Wrapper

This module provides a Python wrapper for the V2 YOLOv8-Pose C API.
It supports standalone pose detection without coupling to other models.

Key features:
- Pure C API (no C++ dependencies)
- CUDA-accelerated post-processing
- Batch inference support
- Clean memory management
"""
import ctypes
from typing import List, Dict, Optional
import numpy as np

from pyengine.utils.logger import logger
from pyengine.inference.c_pipeline.c_structures_v2 import (
    C_ImageInput,
    C_ImageBatch,
    C_YoloPosePipelineConfig,
    C_YoloPoseImageResult,
    C_YoloPoseBatchResult,
    YOLO_POSE_NUM_KEYPOINTS
)


class YoloPosePipelineV2:
    """
    YOLOv8-Pose inference pipeline wrapper

    This class provides a Python interface to the V2 YOLOv8-Pose C API,
    supporting both single and batch inference.
    """

    def __init__(
        self,
        library_path: str,
        engine_path: str,
        input_width: int = 640,
        input_height: int = 640,
        max_batch_size: int = 1,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        num_keypoints: int = YOLO_POSE_NUM_KEYPOINTS
    ):
        """
        Initialize YOLOv8-Pose pipeline

        Args:
            library_path: Path to the shared library (.so file)
            engine_path: Path to the TensorRT engine file
            input_width: Model input width (default: 640)
            input_height: Model input height (default: 640)
            max_batch_size: Maximum batch size (default: 1)
            conf_threshold: Confidence threshold for detections (default: 0.25)
            iou_threshold: IoU threshold for NMS (default: 0.45)
            num_keypoints: Number of keypoints (default: 17 for COCO)
        """
        try:
            self.lib = ctypes.CDLL(library_path)
            logger.info("YoloPosePipelineV2", f"Loaded library: {library_path}")
        except OSError as e:
            raise RuntimeError(f"Failed to load library {library_path}: {e}")

        self.engine_path = engine_path
        self.input_width = input_width
        self.input_height = input_height
        self.max_batch_size = max_batch_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.num_keypoints = num_keypoints

        self._context = None
        self._register_functions()

    def _register_functions(self):
        """Register C function signatures"""

        # C_YoloPosePipelineConfig c_yolopose_pipeline_get_default_config(void);
        self.lib.c_yolopose_pipeline_get_default_config.argtypes = []
        self.lib.c_yolopose_pipeline_get_default_config.restype = C_YoloPosePipelineConfig

        # bool c_yolopose_pipeline_validate_config(const C_YoloPosePipelineConfig* config);
        self.lib.c_yolopose_pipeline_validate_config.argtypes = [
            ctypes.POINTER(C_YoloPosePipelineConfig)
        ]
        self.lib.c_yolopose_pipeline_validate_config.restype = ctypes.c_bool

        # C_YoloPosePipelineContext* c_yolopose_pipeline_create(const C_YoloPosePipelineConfig* config);
        self.lib.c_yolopose_pipeline_create.argtypes = [
            ctypes.POINTER(C_YoloPosePipelineConfig)
        ]
        self.lib.c_yolopose_pipeline_create.restype = ctypes.c_void_p

        # bool c_yolopose_infer_single(context, image, result);
        self.lib.c_yolopose_infer_single.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(C_ImageInput),
            ctypes.POINTER(C_YoloPoseImageResult)
        ]
        self.lib.c_yolopose_infer_single.restype = ctypes.c_bool

        # void c_yolopose_image_result_free(C_YoloPoseImageResult* result);
        self.lib.c_yolopose_image_result_free.argtypes = [
            ctypes.POINTER(C_YoloPoseImageResult)
        ]
        self.lib.c_yolopose_image_result_free.restype = None

        # void c_yolopose_pipeline_destroy(C_YoloPosePipelineContext* context);
        self.lib.c_yolopose_pipeline_destroy.argtypes = [ctypes.c_void_p]
        self.lib.c_yolopose_pipeline_destroy.restype = None

        # const char* c_yolopose_pipeline_get_last_error(context);
        self.lib.c_yolopose_pipeline_get_last_error.argtypes = [ctypes.c_void_p]
        self.lib.c_yolopose_pipeline_get_last_error.restype = ctypes.c_char_p

        # bool c_yolopose_infer_batch(context, batch, result);
        self.lib.c_yolopose_infer_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(C_ImageBatch),
            ctypes.POINTER(C_YoloPoseBatchResult)
        ]
        self.lib.c_yolopose_infer_batch.restype = ctypes.c_bool

        # void c_yolopose_batch_result_free(C_YoloPoseBatchResult* result);
        self.lib.c_yolopose_batch_result_free.argtypes = [
            ctypes.POINTER(C_YoloPoseBatchResult)
        ]
        self.lib.c_yolopose_batch_result_free.restype = None

        logger.info("YoloPosePipelineV2", "C functions registered")

    def create(self):
        """Create and initialize the pipeline"""
        logger.info("YoloPosePipelineV2", "Creating pipeline...")

        # Get default configuration (重要：使用默认配置而不是空配置)
        config = self.lib.c_yolopose_pipeline_get_default_config()

        # Override with user-specified values
        config.engine_path = self.engine_path.encode('utf-8')
        config.input_width = self.input_width
        config.input_height = self.input_height
        config.max_batch_size = self.max_batch_size
        config.conf_threshold = self.conf_threshold
        config.iou_threshold = self.iou_threshold
        config.num_keypoints = self.num_keypoints

        # Validate configuration
        if not self.lib.c_yolopose_pipeline_validate_config(ctypes.byref(config)):
            raise RuntimeError("Invalid pipeline configuration")

        # Create pipeline
        self._context = self.lib.c_yolopose_pipeline_create(ctypes.byref(config))
        if not self._context:
            raise RuntimeError("Failed to create YoloPose pipeline")

        logger.info("YoloPosePipelineV2", "Pipeline created successfully")

    def infer(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Run inference on a list of images

        Args:
            images: List of input images (numpy arrays in RGB format)

        Returns:
            List of detection results, each containing:
            - image_idx: Image index
            - detections: List of pose detections, each with:
                - bbox: [lx, ly, rx, ry]
                - cls: Class ID
                - conf: Confidence score
                - keypoints: List of 17 keypoints, each with {x, y, conf}
        """
        print(f"[DEBUG YoloPose.infer] 开始，图像数量: {len(images)}")

        if self._context is None:
            raise RuntimeError("Pipeline not initialized. Call create() first.")

        if not images:
            return []

        results = []

        for img_idx, img in enumerate(images):
            print(f"[DEBUG YoloPose.infer] 处理图像 {img_idx}/{len(images)}")

            # Validate image
            if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
                logger.error("YoloPosePipelineV2",
                           f"Image {img_idx} must be uint8 numpy array")
                continue

            # Ensure RGB format and contiguous memory
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] != 3:
                logger.error("YoloPosePipelineV2",
                           f"Image {img_idx} must have 3 channels (RGB)")
                continue

            if not img.flags['C_CONTIGUOUS']:
                img = np.ascontiguousarray(img)

            print(f"[DEBUG YoloPose.infer]   图像验证通过: shape={img.shape}, dtype={img.dtype}")

            # Prepare C structure
            print(f"[DEBUG YoloPose.infer]   准备 C_ImageInput...")
            c_image = C_ImageInput()
            c_image.data = img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            c_image.width = img.shape[1]
            c_image.height = img.shape[0]
            c_image.channels = img.shape[2]
            print(f"[DEBUG YoloPose.infer]   C_ImageInput: w={c_image.width}, h={c_image.height}, c={c_image.channels}")

            # Run inference
            print(f"[DEBUG YoloPose.infer]   准备 C_YoloPoseImageResult...")
            c_result = C_YoloPoseImageResult()
            # 显式初始化指针字段为 None (NULL)
            c_result.image_index = 0
            c_result.poses = None
            c_result.num_poses = 0
            print(f"[DEBUG YoloPose.infer]   C_YoloPoseImageResult 初始化完成")

            print(f"[DEBUG YoloPose.infer]   调用 C API: c_yolopose_infer_single...")
            success = self.lib.c_yolopose_infer_single(
                self._context,
                ctypes.byref(c_image),
                ctypes.byref(c_result)
            )
            print(f"[DEBUG YoloPose.infer]   C API 返回: success={success}")

            if not success:
                error_msg = self._get_last_error()
                logger.error("YoloPosePipelineV2",
                           f"Inference failed for image {img_idx}: {error_msg}")
                continue

            # Parse results
            print(f"[DEBUG YoloPose.infer]   解析结果: num_poses={c_result.num_poses}")
            detections = []
            for i in range(c_result.num_poses):
                c_pose = c_result.poses[i]

                # Extract bbox
                bbox = [
                    c_pose.detection.lx,
                    c_pose.detection.ly,
                    c_pose.detection.rx,
                    c_pose.detection.ry
                ]

                # Extract keypoints
                keypoints = []
                for j in range(self.num_keypoints):
                    kpt = c_pose.pts[j]
                    keypoints.append({
                        "x": kpt.x,
                        "y": kpt.y,
                        "conf": kpt.conf
                    })

                detections.append({
                    "bbox": bbox,
                    "cls": c_pose.detection.cls,
                    "conf": c_pose.detection.conf,
                    "keypoints": keypoints
                })

            results.append({
                "image_idx": img_idx,
                "detections": detections
            })

            # Free C memory
            self.lib.c_yolopose_image_result_free(ctypes.byref(c_result))

        return results

    def infer_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Run batch inference on a list of images (true batching)

        Args:
            images: List of input images (numpy arrays in RGB format)

        Returns:
            List of detection results, same format as infer()
        """
        print(f"[DEBUG YoloPose.infer_batch] 开始批处理，图像数量: {len(images)}")

        if self._context is None:
            raise RuntimeError("Pipeline not initialized. Call create() first.")

        if not images:
            return []

        # Validate and prepare all images
        c_images = []
        valid_images = []

        for img_idx, img in enumerate(images):
            # Validate image
            if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
                logger.error("YoloPosePipelineV2",
                           f"Image {img_idx} must be uint8 numpy array")
                continue

            # Ensure RGB format and contiguous memory
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] != 3:
                logger.error("YoloPosePipelineV2",
                           f"Image {img_idx} must have 3 channels (RGB)")
                continue

            if not img.flags['C_CONTIGUOUS']:
                img = np.ascontiguousarray(img)

            valid_images.append(img)

            # Create C structure
            c_image = C_ImageInput()
            c_image.data = img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            c_image.width = img.shape[1]
            c_image.height = img.shape[0]
            c_image.channels = img.shape[2]
            c_images.append(c_image)

        if not c_images:
            return []

        print(f"[DEBUG YoloPose.infer_batch] 有效图像数: {len(c_images)}")

        # Create C_ImageBatch
        c_images_array = (C_ImageInput * len(c_images))(*c_images)
        c_batch = C_ImageBatch()
        c_batch.images = c_images_array
        c_batch.count = len(c_images)

        # Prepare result structure
        c_batch_result = C_YoloPoseBatchResult()
        c_batch_result.results = None
        c_batch_result.num_images = 0

        print(f"[DEBUG YoloPose.infer_batch] 调用 C API: c_yolopose_infer_batch...")
        success = self.lib.c_yolopose_infer_batch(
            self._context,
            ctypes.byref(c_batch),
            ctypes.byref(c_batch_result)
        )
        print(f"[DEBUG YoloPose.infer_batch] C API 返回: success={success}")

        if not success:
            error_msg = self._get_last_error()
            logger.error("YoloPosePipelineV2",
                       f"Batch inference failed: {error_msg}")
            return []

        # Parse results
        results = []
        print(f"[DEBUG YoloPose.infer_batch] 解析结果: num_images={c_batch_result.num_images}")

        for img_idx in range(c_batch_result.num_images):
            c_img_result = c_batch_result.results[img_idx]

            detections = []
            for i in range(c_img_result.num_poses):
                c_pose = c_img_result.poses[i]

                # Extract bbox
                bbox = [
                    c_pose.detection.lx,
                    c_pose.detection.ly,
                    c_pose.detection.rx,
                    c_pose.detection.ry
                ]

                # Extract keypoints
                keypoints = []
                for j in range(self.num_keypoints):
                    kpt = c_pose.pts[j]
                    keypoints.append({
                        "x": kpt.x,
                        "y": kpt.y,
                        "conf": kpt.conf
                    })

                detections.append({
                    "bbox": bbox,
                    "cls": c_pose.detection.cls,
                    "conf": c_pose.detection.conf,
                    "keypoints": keypoints
                })

            results.append({
                "image_idx": img_idx,
                "detections": detections
            })

        # Free C memory
        self.lib.c_yolopose_batch_result_free(ctypes.byref(c_batch_result))

        print(f"[DEBUG YoloPose.infer_batch] 批处理完成，返回 {len(results)} 个结果")
        return results

    def _get_last_error(self) -> Optional[str]:
        """Get last error message from C API"""
        if self._context:
            err_ptr = self.lib.c_yolopose_pipeline_get_last_error(self._context)
            if err_ptr:
                return err_ptr.decode('utf-8')
        return None

    def close(self):
        """Destroy pipeline and free resources"""
        if self._context:
            logger.info("YoloPosePipelineV2", "Destroying pipeline...")
            self.lib.c_yolopose_pipeline_destroy(self._context)
            self._context = None
            logger.info("YoloPosePipelineV2", "Pipeline destroyed")

    def __enter__(self):
        """Context manager entry"""
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def __del__(self):
        """Destructor"""
        self.close()
