import ctypes
from typing import Dict, List

import numpy as np
import cv2  

from pyengine.utils.logger import logger

# --- 1. 定义C结构体映射 ---

class C_KeyPoint(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("conf", ctypes.c_float),
    ]


class C_YoloPose(ctypes.Structure):
    _fields_ = [
        ("lx", ctypes.c_int),
        ("ly", ctypes.c_int),
        ("rx", ctypes.c_int),
        ("ry", ctypes.c_int),
        ("cls", ctypes.c_int),
        ("num_pts", ctypes.c_int),
        ("conf", ctypes.c_float),
        ("pts", ctypes.POINTER(C_KeyPoint)),  # 指向C_KeyPoint数组的指针
        ("feats", ctypes.POINTER(ctypes.c_float)),  # 指向float数组（长度256）的指针
    ]


class C_ImagePoseResults(ctypes.Structure):
    _fields_ = [
        ("image_idx", ctypes.c_int),
        ("num_detections", ctypes.c_int),
        ("detections", ctypes.POINTER(C_YoloPose)),  # 指向C_YoloPose数组的指针
    ]


class C_BatchedPoseResults(ctypes.Structure):
    _fields_ = [
        ("num_images", ctypes.c_int),
        ("results", ctypes.POINTER(C_ImagePoseResults)),  # 指向C_ImagePoseResults数组的指针
    ]


# --- 2. 加载共享库和定义函数签名 ---

class PosePipeline:

    def __init__(self, library_path):
        """
        初始化PosePipeline，加载C语言的共享库。
        """

        try:
            # 加载共享库
            self.lib = ctypes.CDLL(library_path)
            logger.info("PosePipeline", f"Shared library loaded successfully from {library_path}.")
        except OSError as e:
            raise RuntimeError(f"无法加载共享库 {library_path}。错误信息: {e}")

        # 定义函数签名
        self._define_function_signatures()

        # C端pipeline的上下文句柄
        self._context = None
        self._context_handle_ptr = ctypes.POINTER(ctypes.c_void_p)()  # 用于 create_pipeline 和 destroy_pipeline 时的双重指针

        # 其他参数
        self.max_yolo_batch_size = 0  # 最大批处理大小
        self.max_efficient_batch_size = 0  # 最大批处理大小

        # 用于存储处理后的图像数据，防止被Python垃圾回收
        self._processed_images_buffer = None


    def _define_function_signatures(self):
        """
        设置C函数的参数和返回值类型。
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


    def register_models(self):
        """
        注册姿态检测pipeline所需的模型。必须在创建上下文前调用一次。
        """
        logger.info("PosePipeline", "Registering models...")
        self.lib.c_register_models()
        logger.info("PosePipeline", "Models registered successfully.")


    def create_pipeline(self,
                        yolo_engine_path: str,          # YOLO模型的TensorRT引擎文件路径
                        efficient_engine_path: str,     # EfficientNet模型的TensorRT引擎文件路径
                        yolo_max_batch: int,            # YOLO模型的最大批处理大小
                        efficient_max_batch: int,       # EfficientNet模型的最大批处理大小
                        yolo_cls_thresh: float,         # YOLO模型的分类阈值
                        yolo_iou_thresh: float          # YOLO模型的IOU阈值
                        ):
        """
        初始化YoloPose和EfficientNet模型。

        Args:
            yolo_engine_path: YOLO模型的TensorRT引擎文件路径。
            efficient_engine_path: EfficientNet模型的TensorRT引擎文件路径。
            yolo_max_batch: YOLO模型的最大批处理大小。
            efficient_max_batch: EfficientNet模型的最大批处理大小。
            yolo_cls_thresh: YOLO模型的分类阈值。
            yolo_iou_thresh: YOLO模型的IOU阈值。
        """
        logger.info("PosePipeline", "Creating pipeline...")
        yolo_engine_path_c = yolo_engine_path.encode('utf-8')
        efficient_engine_path_c = efficient_engine_path.encode('utf-8')

        # C函数返回void* (YoloEfficientContext*)，ctypes会映射为Python的int
        # 这里需要将 _context_handle_ptr.contents 置为空，确保C函数返回新的地址
        self._context_handle_ptr.contents = ctypes.c_void_p(0)  # Initialize with null pointer

        context_address = self.lib.c_create_pose_pipeline(
            yolo_engine_path_c,
            efficient_engine_path_c,
            yolo_max_batch,
            efficient_max_batch,
            yolo_cls_thresh,
            yolo_iou_thresh
        )
        if context_address is None or context_address == 0:
            raise RuntimeError("创建姿态pipeline上下文失败。")

        # 记录最大批处理大小
        self.max_yolo_batch_size = yolo_max_batch
        self.max_efficient_batch_size = efficient_max_batch

        # 将context地址存储到 _context 中
        self._context = ctypes.cast(context_address, ctypes.c_void_p)

        # 同时，更新 _context_handle_ptr 的内容，以便 destroy_pipeline 可以使用它
        self._context_handle_ptr.contents = self._context

        logger.info("PosePipeline", "Pipeline created successfully.")


    def process_batched_images(self, images: List[np.ndarray], scale_factor: float) -> List[Dict]:
            """
            批量处理图像，经过YoloPose和EfficientNet pipeline。
            支持自动分块处理，当输入批次大于模型最大批次时，会自动切分为多个小批次。

            Args:
                images: NumPy数组列表，每个数组为一张图片，OpenCV格式：(H, W, C)，dtype=np.uint8 (BGR)。
                        注意：此函数将负责内部将图片统一resize到640x480。
            Returns:
                list[dict]: 返回一个包含每张图片检测结果的字典列表。
                                每个字典包含以下键：
                                    - "image_idx": 图片在原始输入列表中的索引
                                    - "detections": 检测到的人体信息列表...
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
                    scale_factor
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

    def destroy_pipeline(self):
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
        self.destroy_pipeline()



"""
import cv2
import numpy as np
import os
import sys

# 假设你的 PosePipeline 类定义在以下路径
# 请确保 pyengine 目录在你的 Python 模块搜索路径中
# 例如，可以通过将项目根目录添加到 PYTHONPATH
# 或者确保脚本从正确的目录运行
try:
    from pyengine.inference.c_pipeline.pose_pipeline_v1 import PosePipeline
except ImportError:
    print("Error: Could not import PosePipeline.")
    print("Please ensure 'pyengine' is in your Python path or working directory.")
    print("Example: sys.path.append('/path/to/your/project/root')")
    sys.exit(1)

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
    EFFICIENT_ENGINE = "/opt/models/efficientnet_b0_feat_logits.engine"

    pipeline = None
    try:
        pipeline = PosePipeline(LIBRARY_PATH)

        # 注册模型 (只需调用一次)
        pipeline.register_models()

        # 创建管道实例
        pipeline.create_pipeline(
            yolo_engine_path=YOLO_ENGINE,
            efficient_engine_path=EFFICIENT_ENGINE,
            yolo_max_batch=4,
            efficient_max_batch=32,
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
        results = pipeline.process_batched_images(loaded_images_for_pipeline, scale_factor=1.0)

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
            pipeline.destroy_pipeline()
            print("Pipeline destroyed.")
"""