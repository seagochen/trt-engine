import ctypes
from typing import Dict, List

import numpy as np
import cv2

from pyengine.utils.logger import logger
from pyengine.inference.c_pipeline.detect_base import DetectorBase, C_BatchedPoseResults


class YoloPoseV2(DetectorBase):

    def __init__(
            self,
            library_path: str,             # trtengine的库文件地址
            yolo_engine_path: str,         # YOLO模型的TensorRT引擎文件路径
            yolo_max_batch: int,           # YOLO模型的最大批处理大小
            yolo_cls_thresh: float,        # YOLO模型的分类阈值
            yolo_iou_thresh: float,        # YOLO模型的IOU阈值
    ):
        """
        初始化PosePipeline，加载C语言的共享库。
        """

        try:
            self.lib = ctypes.CDLL(library_path)
            logger.info("YoloposePipelineV2", f"Shared library loaded successfully from {library_path}.")
        except OSError as e:
            raise RuntimeError(f"Failed to load shared library from {library_path}: {e}")

        # Record other parameters
        self.yolo_engine_path = yolo_engine_path
        self.yolo_max_batch = yolo_max_batch
        self.yolo_cls_thresh = yolo_cls_thresh
        self.yolo_iou_thresh = yolo_iou_thresh

        # C端的上下文句柄
        self._context = None

        # 防止图像被GC
        self._processed_images_buffer = None

    def register(self):
        """
        注册YoloPose模型(只需调用一次)
        """
        # void c_register_yolopose_model();
        self.lib.c_register_yolopose_model.argtypes = []
        self.lib.c_register_yolopose_model.restype = None

        # void* c_create_yolopose_context(const char* model_path, int, float, float)
        self.lib.c_create_yolopose_context.argtypes = [
            ctypes.c_char_p,   # 注意：这里是 c_char_p，不是 _c_char_p
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self.lib.c_create_yolopose_context.restype = ctypes.c_void_p

        # C_BatchedPoseResults c_process_batched_images_with_yolopose(void* ctx, ...);
        self.lib.c_process_batched_images_with_yolopose.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self.lib.c_process_batched_images_with_yolopose.restype = C_BatchedPoseResults

        # void c_free_batched_pose_results(C_BatchedPoseResults* results)
        self.lib.c_free_batched_pose_results.argtypes = [ctypes.POINTER(C_BatchedPoseResults)]
        self.lib.c_free_batched_pose_results.restype = None

        # void c_destroy_yolopose_context(void* context)
        self.lib.c_destroy_yolopose_context.argtypes = [ctypes.c_void_p]
        self.lib.c_destroy_yolopose_context.restype = None

        self.lib.c_register_yolopose_model()
        logger.info("YoloposePipelineV2", "YoloPose model registered successfully.")

    def create(self):
        """
        初始化YoloPose上下文。
        """
        logger.info("YoloPoseWrapper", "Creating YoloPose context...")
        yolo_engine_path_c = self.yolo_engine_path.encode("utf-8")

        ctx = self.lib.c_create_yolopose_context(
            yolo_engine_path_c,
            self.yolo_max_batch,
            self.yolo_cls_thresh,
            self.yolo_iou_thresh,
        )
        if not ctx:
            raise RuntimeError("创建 YoloPose 上下文失败。")

        self._context = ctypes.c_void_p(ctx)
        logger.info("YoloPoseWrapper", "YoloPose context created.")

    def infer(self, images: List[np.ndarray]) -> List[Dict]:
        if self._context is None:
            raise RuntimeError("YoloPoseWrapper context 未初始化，请先调用 create_context。")

        num_images = len(images)
        if num_images == 0:
            return []

        all_results = []

        for i in range(0, num_images, self.yolo_max_batch):
            chunk_images = images[i : i + self.yolo_max_batch]
            current_chunk_size = len(chunk_images)

            images_ptr = (ctypes.POINTER(ctypes.c_ubyte) * current_chunk_size)()
            widths = (ctypes.c_int * current_chunk_size)()
            heights = (ctypes.c_int * current_chunk_size)()
            channels = (ctypes.c_int * current_chunk_size)()

            self._processed_images_buffer = []

            for j, img in enumerate(chunk_images):
                if img.shape[0] != 640 or img.shape[1] != 640:
                    img = cv2.resize(img, (640, 640))

                if not img.flags["C_CONTIGUOUS"]:
                    img = np.ascontiguousarray(img)

                self._processed_images_buffer.append(img)
                images_ptr[j] = img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
                widths[j] = img.shape[1]
                heights[j] = img.shape[0]
                channels[j] = img.shape[2]

            c_batched_results = self.lib.c_process_batched_images_with_yolopose(
                self._context,
                images_ptr,
                widths,
                heights,
                channels,
                current_chunk_size,
            )

            for k in range(c_batched_results.num_images):
                c_img_results = c_batched_results.results[k]
                img_idx = i + c_img_results.image_idx

                detections = []
                for l in range(c_img_results.num_detections):
                    c_yolopose = c_img_results.detections[l]

                    keypoints = []
                    for m in range(c_yolopose.num_pts):
                        c_keypoint = c_yolopose.pts[m]
                        keypoints.append(
                            {"x": c_keypoint.x, "y": c_keypoint.y, "conf": c_keypoint.conf}
                        )

                    detections.append(
                        {
                            "bbox": [c_yolopose.lx, c_yolopose.ly, c_yolopose.rx, c_yolopose.ry],
                            "classification": c_yolopose.cls,
                            "confidence": c_yolopose.conf,
                            "keypoints": keypoints,
                            "features": None,
                        }
                    )

                all_results.append({"image_idx": img_idx, "detections": detections})

            self.lib.c_free_batched_pose_results(ctypes.byref(c_batched_results))
            self._processed_images_buffer = None

        return all_results

    def close(self):
        """
        销毁上下文，释放资源。
        """
        if self._context:
            logger.info("YoloPoseWrapper", "Destroying YoloPose context...")
            self.lib.c_destroy_yolopose_context(self._context)
            self._context = None
            logger.info("YoloPoseWrapper", "YoloPose context destroyed.")

    def __del__(self):
        """
        确保对象被回收时销毁pipeline。
        """
        self.close()