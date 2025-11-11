"""
EfficientNet Pipeline V2 - Pure C API Wrapper

This module provides a Python wrapper for the V2 EfficientNet C API.
It supports standalone classification and feature extraction.

Key features:
- Pure C API (no C++ dependencies)
- Batch inference support
- Both classification and feature extraction
- Clean memory management
"""
import ctypes
from typing import List, Dict, Optional
import numpy as np

from pyengine.utils.logger import logger
from pyengine.inference.c_pipeline.c_structures_v2 import (
    C_ImageInput,
    C_EfficientNetPipelineConfig,
    C_EfficientNetResult,
    C_EfficientNetBatchResult,
    EFFICIENTNET_DEFAULT_FEAT_SIZE,
    EFFICIENTNET_DEFAULT_NUM_CLASSES,
    EFFICIENTNET_DEFAULT_IMAGE_SIZE
)


class EfficientNetPipelineV2:
    """
    EfficientNet inference pipeline wrapper

    This class provides a Python interface to the V2 EfficientNet C API,
    supporting both classification and feature extraction.
    """

    def __init__(
        self,
        library_path: str,
        engine_path: str,
        input_width: int = EFFICIENTNET_DEFAULT_IMAGE_SIZE,
        input_height: int = EFFICIENTNET_DEFAULT_IMAGE_SIZE,
        max_batch_size: int = 1,
        num_classes: int = EFFICIENTNET_DEFAULT_NUM_CLASSES,
        feature_size: int = EFFICIENTNET_DEFAULT_FEAT_SIZE,
        mean: List[float] = None,
        stddev: List[float] = None
    ):
        """
        Initialize EfficientNet pipeline

        Args:
            library_path: Path to the shared library (.so file)
            engine_path: Path to the TensorRT engine file
            input_width: Model input width (default: 224)
            input_height: Model input height (default: 224)
            max_batch_size: Maximum batch size (default: 1)
            num_classes: Number of classification classes (default: 2)
            feature_size: Size of feature embedding vector (default: 512)
            mean: Normalization mean [R, G, B] (default: ImageNet values)
            stddev: Normalization std [R, G, B] (default: ImageNet values)
        """
        try:
            self.lib = ctypes.CDLL(library_path)
            logger.info("EfficientNetPipelineV2", f"Loaded library: {library_path}")
        except OSError as e:
            raise RuntimeError(f"Failed to load library {library_path}: {e}")

        self.engine_path = engine_path
        self.input_width = input_width
        self.input_height = input_height
        self.max_batch_size = max_batch_size
        self.num_classes = num_classes
        self.feature_size = feature_size

        # ImageNet default normalization
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.stddev = stddev if stddev is not None else [0.229, 0.224, 0.225]

        self._context = None
        self._register_functions()

    def _register_functions(self):
        """Register C function signatures"""

        # C_EfficientNetPipelineConfig c_efficientnet_pipeline_get_default_config(void);
        self.lib.c_efficientnet_pipeline_get_default_config.argtypes = []
        self.lib.c_efficientnet_pipeline_get_default_config.restype = C_EfficientNetPipelineConfig

        # bool c_efficientnet_pipeline_validate_config(const C_EfficientNetPipelineConfig* config);
        self.lib.c_efficientnet_pipeline_validate_config.argtypes = [
            ctypes.POINTER(C_EfficientNetPipelineConfig)
        ]
        self.lib.c_efficientnet_pipeline_validate_config.restype = ctypes.c_bool

        # C_EfficientNetPipelineContext* c_efficientnet_pipeline_create(const C_EfficientNetPipelineConfig* config);
        self.lib.c_efficientnet_pipeline_create.argtypes = [
            ctypes.POINTER(C_EfficientNetPipelineConfig)
        ]
        self.lib.c_efficientnet_pipeline_create.restype = ctypes.c_void_p

        # bool c_efficientnet_infer_single(context, image, result);
        self.lib.c_efficientnet_infer_single.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(C_ImageInput),
            ctypes.POINTER(C_EfficientNetResult)
        ]
        self.lib.c_efficientnet_infer_single.restype = ctypes.c_bool

        # void c_efficientnet_result_free(C_EfficientNetResult* result);
        self.lib.c_efficientnet_result_free.argtypes = [
            ctypes.POINTER(C_EfficientNetResult)
        ]
        self.lib.c_efficientnet_result_free.restype = None

        # void c_efficientnet_pipeline_destroy(C_EfficientNetPipelineContext* context);
        self.lib.c_efficientnet_pipeline_destroy.argtypes = [ctypes.c_void_p]
        self.lib.c_efficientnet_pipeline_destroy.restype = None

        # const char* c_efficientnet_pipeline_get_last_error(context);
        self.lib.c_efficientnet_pipeline_get_last_error.argtypes = [ctypes.c_void_p]
        self.lib.c_efficientnet_pipeline_get_last_error.restype = ctypes.c_char_p

        logger.info("EfficientNetPipelineV2", "C functions registered")

    def create(self):
        """Create and initialize the pipeline"""
        logger.info("EfficientNetPipelineV2", "Creating pipeline...")

        # Prepare configuration
        config = C_EfficientNetPipelineConfig()
        config.engine_path = self.engine_path.encode('utf-8')
        config.input_width = self.input_width
        config.input_height = self.input_height
        config.max_batch_size = self.max_batch_size
        config.num_classes = self.num_classes
        config.feature_size = self.feature_size

        # Set normalization parameters
        for i in range(3):
            config.mean[i] = self.mean[i]
            config.stddev[i] = self.stddev[i]

        # Validate configuration
        if not self.lib.c_efficientnet_pipeline_validate_config(ctypes.byref(config)):
            raise RuntimeError("Invalid pipeline configuration")

        # Create pipeline
        self._context = self.lib.c_efficientnet_pipeline_create(ctypes.byref(config))
        if not self._context:
            raise RuntimeError("Failed to create EfficientNet pipeline")

        logger.info("EfficientNetPipelineV2", "Pipeline created successfully")

    def infer(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Run inference on a list of images

        Args:
            images: List of input images (numpy arrays in RGB format)

        Returns:
            List of classification results, each containing:
            - image_idx: Image index
            - class_id: Predicted class ID
            - confidence: Confidence score
            - logits: Raw logits for all classes (numpy array)
            - features: Feature embedding vector (numpy array)
        """
        if self._context is None:
            raise RuntimeError("Pipeline not initialized. Call create() first.")

        if not images:
            return []

        results = []

        for img_idx, img in enumerate(images):
            # Validate image
            if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
                logger.error("EfficientNetPipelineV2",
                           f"Image {img_idx} must be uint8 numpy array")
                continue

            # Ensure RGB format and contiguous memory
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] != 3:
                logger.error("EfficientNetPipelineV2",
                           f"Image {img_idx} must have 3 channels (RGB)")
                continue

            if not img.flags['C_CONTIGUOUS']:
                img = np.ascontiguousarray(img)

            # Prepare C structure
            c_image = C_ImageInput()
            c_image.data = img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            c_image.width = img.shape[1]
            c_image.height = img.shape[0]
            c_image.channels = img.shape[2]

            # Run inference
            c_result = C_EfficientNetResult()
            success = self.lib.c_efficientnet_infer_single(
                self._context,
                ctypes.byref(c_image),
                ctypes.byref(c_result)
            )

            if not success:
                error_msg = self._get_last_error()
                logger.error("EfficientNetPipelineV2",
                           f"Inference failed for image {img_idx}: {error_msg}")
                continue

            # Parse results
            # Extract logits
            logits = np.array(
                [c_result.logits[i] for i in range(c_result.num_classes)],
                dtype=np.float32
            )

            # Extract features
            features = np.array(
                [c_result.features[i] for i in range(c_result.feature_size)],
                dtype=np.float32
            )

            results.append({
                "image_idx": img_idx,
                "class_id": c_result.class_id,
                "confidence": c_result.confidence,
                "logits": logits,
                "features": features
            })

            # Free C memory
            self.lib.c_efficientnet_result_free(ctypes.byref(c_result))

        return results

    def _get_last_error(self) -> Optional[str]:
        """Get last error message from C API"""
        if self._context:
            err_ptr = self.lib.c_efficientnet_pipeline_get_last_error(self._context)
            if err_ptr:
                return err_ptr.decode('utf-8')
        return None

    def close(self):
        """Destroy pipeline and free resources"""
        if self._context:
            logger.info("EfficientNetPipelineV2", "Destroying pipeline...")
            self.lib.c_efficientnet_pipeline_destroy(self._context)
            self._context = None
            logger.info("EfficientNetPipelineV2", "Pipeline destroyed")

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
