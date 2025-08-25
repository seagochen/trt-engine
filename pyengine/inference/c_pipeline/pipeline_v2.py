import ctypes
from typing import List, Dict

import cv2
import numpy as np

from pyengine.utils.logger import logger
from pyengine.inference.c_pipeline.detect_base import DetectorBase, C_BatchedPoseResults


class PosePipelineV2(DetectorBase):

    def __init__(
            self,
            library_path,
            yolo_engine_path: str,          # YOLO模型的TensorRT引擎文件路径
            efficient_engine_path: str,     # EfficientNet模型的TensorRT引擎文件路径
            yolo_max_batch: int,            # YOLO模型的最大批处理大小
            efficient_max_batch: int,       # EfficientNet模型的最大批处理大小
            yolo_cls_thresh: float,         # YOLO模型的分类阈值
            yolo_iou_thresh: float,         # YOLO模型的IOU阈值
            scale_factor: float = 1.0
    ):
        """
        初始化PosePipeline，加载C语言的共享库。
        """

        try:
            # 加载共享库
            self.lib = ctypes.CDLL(library_path)
            logger.info("PosePipeline", f"Shared library loaded successfully from {library_path}.")
        except OSError as e:
            raise RuntimeError(f"无法加载共享库 {library_path}。错误信息: {e}")

        # C端pipeline的上下文句柄
        self._context = None
        self._context_handle_ptr = ctypes.POINTER(ctypes.c_void_p)()  # 用于 create_pipeline 和 destroy_pipeline 时的双重指针

        # 其他参数
        self.max_yolo_batch_size = 0  # 最大批处理大小
        self.max_efficient_batch_size = 0  # 最大批处理大小

        # 用于存储处理后的图像数据，防止被Python垃圾回收
        self._processed_images_buffer = None

        # Register the parameters
        self.yolo_engine_path = yolo_engine_path
        self.yolo_max_batch = yolo_max_batch
        self.yolo_cls_thresh = yolo_cls_thresh
        self.yolo_iou_thresh = yolo_iou_thresh
        self.efficient_engine_path = efficient_engine_path
        self.efficient_max_batch = efficient_max_batch
        self.scale_factor = scale_factor


    def register(self):
        """
        注册姿态检测pipeline所需的模型。必须在创建上下文前调用一次。
        """
        # void c_register_models();
        self.lib.c_register_models.argtypes = []
        self.lib.c_register_models.restype = None

        # YoloEfficientContext* c_create_pose_pipeline(...)
        self.lib.c_create_pose_pipeline.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float
        ]
        self.lib.c_create_pose_pipeline.restype = ctypes.c_void_p  # 不透明指针类型

        # C_BatchedPoseResults c_process_batched_images(...)
        self.lib.c_process_batched_images.argtypes = [
            ctypes.c_void_p,  # YoloEfficientContext* context, 直接传递 void*
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),  # const unsigned char* const* input_images_data
            ctypes.POINTER(ctypes.c_int),   # const int* widths
            ctypes.POINTER(ctypes.c_int),   # const int* heights
            ctypes.POINTER(ctypes.c_int),   # const int* channels
            ctypes.c_int,                   # int num_images
            ctypes.c_float                  # float scale_factor
        ]
        self.lib.c_process_batched_images.restype = C_BatchedPoseResults

        # void c_free_batched_pose_results(C_BatchedPoseResults* results)
        self.lib.c_free_batched_pose_results.argtypes = [ctypes.POINTER(C_BatchedPoseResults)]
        self.lib.c_free_batched_pose_results.restype = None

        # void c_destroy_pose_pipeline(YoloEfficientContext* context)
        self.lib.c_destroy_pose_pipeline.argtypes = [ctypes.c_void_p]  # 传递 void*
        self.lib.c_destroy_pose_pipeline.restype = None

        self.lib.c_register_models()
        logger.info("PosePipeline", "Models registered successfully.")


    def create(self):
        """
        初始化YoloPose和EfficientNet模型。
        """
        logger.info("PosePipeline", "Creating pipeline...")
        yolo_engine_path_c = self.yolo_engine_path.encode('utf-8')
        efficient_engine_path_c = self.efficient_engine_path.encode('utf-8')

        # C函数返回void* (YoloEfficientContext*)，ctypes会映射为Python的int
        # 这里需要将 _context_handle_ptr.contents 置为空，确保C函数返回新的地址
        self._context_handle_ptr.contents = ctypes.c_void_p(0)  # Initialize with null pointer

        context_address = self.lib.c_create_pose_pipeline(
            yolo_engine_path_c,
            efficient_engine_path_c,
            self.yolo_max_batch,
            self.efficient_max_batch,
            self.yolo_cls_thresh,
            self.yolo_iou_thresh
        )
        if context_address is None or context_address == 0:
            raise RuntimeError("创建姿态pipeline上下文失败。")

        # 记录最大批处理大小
        self.max_yolo_batch_size = self.yolo_max_batch
        self.max_efficient_batch_size = self.efficient_max_batch

        # 将context地址存储到 _context 中
        self._context = ctypes.cast(context_address, ctypes.c_void_p)

        # 同时，更新 _context_handle_ptr 的内容，以便 destroy_pipeline 可以使用它
        self._context_handle_ptr.contents = self._context

        logger.info("PosePipeline", "Pipeline created successfully.")


    def infer(self, images: List[np.ndarray]) -> List[Dict]:
            """
            批量处理图像，经过YoloPose和EfficientNet pipeline。
            支持自动分块处理，当输入批次大于模型最大批次时，会自动切分为多个小批次。
            """

            if self._context is None:
                raise RuntimeError("姿态pipeline上下文未初始化。请先调用create_pipeline。")

            num_images = len(images)
            if num_images == 0:
                return []

            all_results = []
            max_batch_size = self.max_yolo_batch_size

            for i in range(0, num_images, max_batch_size):
                chunk_images = images[i: i + max_batch_size]
                current_chunk_size = len(chunk_images)

                image_data_pointers = (ctypes.POINTER(ctypes.c_ubyte) * current_chunk_size)()
                widths = (ctypes.c_int * current_chunk_size)()
                heights = (ctypes.c_int * current_chunk_size)()
                channels = (ctypes.c_int * current_chunk_size)()

                self._processed_images_buffer = []

                for j, img in enumerate(chunk_images):
                    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3:
                        logger.error("PosePipeline",
                                     f"第{i + j}张图片(块内索引{j})必须是3维uint8类型的NumPy数组 (H, W, C)。已跳过。")
                        continue

                    height, width = img.shape[:2]
                    if height != 480 or width != 640:
                        resized_img = cv2.resize(img, (640, 480))
                    else:
                        resized_img = img

                    if not resized_img.flags['C_CONTIGUOUS']:
                        resized_img = np.ascontiguousarray(resized_img)

                    self._processed_images_buffer.append(resized_img)

                    image_data_pointers[j] = resized_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
                    heights[j], widths[j], channels[j] = resized_img.shape

                c_batched_results = self.lib.c_process_batched_images(
                    self._context,
                    image_data_pointers,
                    widths,
                    heights,
                    channels,
                    current_chunk_size,
                    self.scale_factor
                )

                for k in range(c_batched_results.num_images):
                    c_image_results = c_batched_results.results[k]
                    original_image_idx = i + c_image_results.image_idx
                    image_detections = []
                    for l in range(c_image_results.num_detections):
                        c_yolo_pose = c_image_results.detections[l]

                        keypoints = []
                        for m in range(c_yolo_pose.num_pts):
                            c_keypoint = c_yolo_pose.pts[m]
                            keypoints.append({
                                "x": c_keypoint.x,
                                "y": c_keypoint.y,
                                "conf": c_keypoint.conf
                            })

                        # --- 修改点在这里 ---
                        # 检查C++返回的特征指针
                        if c_yolo_pose.feats:
                            # 如果指针有效，读取256维数据
                            feature_vector = np.array(c_yolo_pose.feats[0:256], dtype=np.float32)
                        else:
                            # 如果指针为NULL，创建一个256维的全零向量作为占位符
                            logger.warning("PosePipeline",
                                           f"Detection {l} in original image {original_image_idx} has NULL feature pointer. Returning a zero vector.")

                            # --- 新增调试信息 START ---
                            # 从 c_yolo_pose 结构体中提取详细信息
                            bbox_w = c_yolo_pose.rx - c_yolo_pose.lx  #
                            bbox_h = c_yolo_pose.ry - c_yolo_pose.ly  #
                            conf = c_yolo_pose.conf  #
                            num_pts = c_yolo_pose.num_pts  #

                            # 打印这些详细信息，以帮助分析失败原因
                            logger.info("PosePipeline",
                                        f"  [Debug Info] Failed Detection Details -> BBox Size: {bbox_w}x{bbox_h}, Confidence: {conf:.4f}, Keypoints: {num_pts}")
                            # --- 新增调试信息 END ---

                            feature_vector = np.zeros(256, dtype=np.float32)
                        # --- 修改结束 ---

                        image_detections.append({
                            "bbox": [c_yolo_pose.lx, c_yolo_pose.ly, c_yolo_pose.rx, c_yolo_pose.ry],
                            "classification": c_yolo_pose.cls,
                            "confidence": c_yolo_pose.conf,
                            "keypoints": keypoints,
                            "features": feature_vector  # 现在这里永远是(256,)的数组
                        })

                    all_results.append({
                        "image_idx": original_image_idx,
                        "detections": image_detections
                    })

                self.lib.c_free_batched_pose_results(ctypes.byref(c_batched_results))
                self._processed_images_buffer = []

            return all_results

    def close(self):
        """
        销毁YoloEfficientContext，释放所有相关模型资源。
        """
        if self._context is not None:
            logger.info("PosePipeline", "Destroying pose pipeline...")
            self.lib.c_destroy_pose_pipeline(self._context)
            self._context = None  # Python端也置为None
            self._context_handle_ptr.contents = ctypes.c_void_p(0)  # 也清空这个指针
            logger.info("PosePipeline", "Pose pipeline destroyed.")


    def __del__(self):
        """
        确保对象被回收时销毁pipeline。
        """
        self.close()

