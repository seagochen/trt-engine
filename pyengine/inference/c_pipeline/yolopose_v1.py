# yolopose_v1.py
import ctypes
from typing import Dict, List

import numpy as np
import cv2

from pyengine.inference.c_pipeline.pose_pipeline_v1 import C_BatchedPoseResults
from pyengine.utils.logger import logger


class YoloPoseWrapper:
    def __init__(self, library_path):
        """
        初始化YoloPoseWrapper，加载C语言的共享库。
        """
        try:
            self.lib = ctypes.CDLL(library_path)
            logger.info("YoloPoseWrapper", f"Shared library loaded successfully from {library_path}.")
        except OSError as e:
            raise RuntimeError(f"Failed to load shared library from {library_path}: {e}")

        # 定义函数签名
        self._define_function_signatures()

        # C端的上下文句柄
        self._context = None

        # 防止图像被GC
        self._processed_images_buffer = None

    def _define_function_signatures(self):
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

    def register_model(self):
        """
        注册YoloPose模型（只需调用一次）
        """
        self.lib.c_register_yolopose_model()
        logger.info("YoloPoseWrapper", "YoloPose model registered successfully.")

    def create_context(
        self,
        yolo_engine_path: str,
        yolo_max_batch: int,
        yolo_cls_thresh: float,
        yolo_iou_thresh: float,
    ):
        """
        初始化YoloPose上下文。
        """
        logger.info("YoloPoseWrapper", "Creating YoloPose context...")
        yolo_engine_path_c = yolo_engine_path.encode("utf-8")

        ctx = self.lib.c_create_yolopose_context(
            yolo_engine_path_c,
            yolo_max_batch,
            yolo_cls_thresh,
            yolo_iou_thresh,
        )
        if not ctx:
            raise RuntimeError("创建 YoloPose 上下文失败。")

        self.max_yolo_batch_size = yolo_max_batch
        self._context = ctypes.c_void_p(ctx)
        logger.info("YoloPoseWrapper", "YoloPose context created.")

    def process_batched_images(self, images: List[np.ndarray]) -> List[Dict]:
        if self._context is None:
            raise RuntimeError("YoloPoseWrapper context 未初始化，请先调用 create_context。")

        num_images = len(images)
        if num_images == 0:
            return []

        all_results = []

        for i in range(0, num_images, self.max_yolo_batch_size):
            chunk_images = images[i : i + self.max_yolo_batch_size]
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

    def destroy_context(self):
        """
        销毁上下文，释放资源。
        """
        if self._context:
            logger.info("YoloPoseWrapper", "Destroying YoloPose context...")
            self.lib.c_destroy_yolopose_context(self._context)
            self._context = None
            logger.info("YoloPoseWrapper", "YoloPose context destroyed.")

    def __del__(self):
        # 防止 __init__ 早期异常导致属性不存在
        if hasattr(self, "_context") and self._context:
            self.destroy_context()


"""

import cv2
import numpy as np
import os
import sys

from pyengine.inference.c_pipeline.yolopose_v1 import YoloPoseWrapper

# --- 定义姿态骨架连接（COCO格式常见）---
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

KEYPOINT_COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0),  # Red, Green, Blue, Cyan, Yellow
    (255, 0, 255), (128, 0, 128), (0, 128, 0), (128, 128, 0), (0, 0, 128),
    # Magenta, Purple, Dark Green, Olive, Dark Blue
    (128, 0, 0), (0, 128, 128), (192, 192, 192), (64, 64, 64), (128, 64, 0),  # Dark Red, Teal, Silver, Dark Gray, Brown
    (0, 128, 64), (64, 0, 128)  # Teal-Green, Purple-Blue
]

# 为不同分类结果定义边界框颜色
BBOX_COLOR_CLASS_0 = (0, 0, 255)  # 蓝色代表分类 0 （例如，男性）
BBOX_COLOR_CLASS_1 = (0, 255, 0)  # 绿色代表分类 1 （例如，女性）
# 默认颜色或未指定分类时的颜色 (不再使用BBOX_COLOR)
# BBOX_COLOR = (0, 255, 255) # Cyan for bounding boxes

TEXT_COLOR = (255, 255, 255)  # White for text
FONT_SCALE = 0.7
THICKNESS = 2
KEYPOINT_RADIUS = 5


def draw_detections(image: np.ndarray, detections: list, image_idx: int) -> np.ndarray:
    # Make a copy to avoid modifying the original image
    display_image = image.copy()

    for det in detections:
        # Draw bounding box
        bbox = det['bbox']
        lx, ly, rx, ry = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        classification = det['classification']
        confidence = det['confidence']
        current_bbox_color = (0, 255, 255)  # Default color if classification is not recognized
        label = f"Class {classification}: {confidence:.2f}"

        # 根据分类得分选择边界框颜色
        if classification == 0:
            current_bbox_color = BBOX_COLOR_CLASS_0
            label = f"Class 0: {confidence:.2f}"
        elif classification == 1:
            current_bbox_color = BBOX_COLOR_CLASS_1
            label = f"Class 1: {confidence:.2f}"

        cv2.rectangle(display_image, (lx, ly), (rx, ry), current_bbox_color, THICKNESS)

        # Put classification text
        cv2.putText(display_image, label, (lx, ly - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, THICKNESS)

        # Draw keypoints (no skeleton lines)
        keypoints_list = det['keypoints']
        for i, kp in enumerate(keypoints_list):
            if kp['conf'] > 0.1:  # Only draw keypoints with sufficient confidence
                center = (int(kp['x']), int(kp['y']))
                color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]  # Cycle through colors
                cv2.circle(display_image, center, KEYPOINT_RADIUS, color, -1)  # -1 means filled circle

    return display_image


if __name__ == "__main__":
    # 你的共享库路径
    LIBRARY_PATH = "/home/user/projects/TrtEngineToolkits/build/lib/libjetson.so"

    # 你的 TensorRT 引擎文件路径
    YOLO_ENGINE = "/opt/models/yolov8n-pose.engine"

    pipeline = None
    try:
        pipeline = YoloPoseWrapper(LIBRARY_PATH)

        # 注册模型 (只需调用一次)
        pipeline.register_model()

        # 创建管道实例
        pipeline.create_context(
            yolo_engine_path=YOLO_ENGINE,
            yolo_max_batch=4,
            yolo_cls_thresh=0.5,
            yolo_iou_thresh=0.4
        )

        # 加载图像
        image_paths = [
            "/opt/images/supermarket/customer1.png",
            "/opt/images/supermarket/customer2.png",
            "/opt/images/supermarket/customer3.png",
            "/opt/images/supermarket/customer4.png",
        ]

        # 存储加载的图像，以及它们的索引，以便在结果处理后能找到对应的原始图像
        original_images = []
        loaded_images_for_pipeline = []  # This list will be passed to the pipeline

        print("Loading images...")
        for i, path in enumerate(image_paths):
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not load image from {path}. Skipping.")
                continue

            # Resize image to 640x640 for the pipeline
            img_resized = cv2.resize(img, (640, 640))

            original_images.append({'idx': i, 'path': path, 'image_data': img_resized})  # Store resized image
            loaded_images_for_pipeline.append(img_resized)
            print(
                f"Loaded and resized {path} to {img_resized.shape[1]}x{img_resized.shape[0]} (Shape: {img_resized.shape})")

        if not loaded_images_for_pipeline:
            print("No images loaded. Exiting.")
            sys.exit(0)

        # 处理图像批次
        print("\nProcessing images through pipeline...")
        results = pipeline.process_batched_images(loaded_images_for_pipeline)

        print("\nDisplaying results:")
        for img_res in results:
            image_idx_in_original_list = img_res['image_idx']

            # Find the corresponding original image data
            original_image_info = next((item for item in original_images if item['idx'] == image_idx_in_original_list),
                                       None)

            if original_image_info is None:
                print(f"Error: Original image for index {image_idx_in_original_list} not found.")
                continue

            original_image_data = original_image_info['image_data']
            image_path = original_image_info['path']

            # Draw detections on the image
            drawn_image = draw_detections(original_image_data, img_res['detections'], image_idx_in_original_list)

            # Display the image
            window_name = f"Image {image_idx_in_original_list}: {os.path.basename(image_path)}"
            cv2.imshow(window_name, drawn_image)
            print(f"Displayed image: {window_name} with {len(img_res['detections'])} detections.")

        # Wait for a key press and then close all windows
        print("\nPress any key to close all image windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except RuntimeError as e:
        print(f"Runtime Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # 确保在程序退出时销毁pipeline
        if pipeline:
            pipeline.destroy_context()
            print("Pipeline destroyed.")
"""