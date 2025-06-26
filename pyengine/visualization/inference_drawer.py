import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

from pyengine.utils.logger import logger
from pyengine.inference.unified_structs.inference_results import Rect, ObjectDetection, Point, Skeleton
from pyengine.visualization.schema_loader import SchemaLoader


class GenericInferenceDrawer:
    """
    用于绘制 ObjectDetection 和 Skeleton 对象的绘图类。
    利用 SchemaLoader 加载颜色和骨骼连接配置。
    """

    def __init__(self, schema_file: Optional[str] = None):
        self.schema_loader = SchemaLoader(schema_file)
        self.kpt_color_map = self.schema_loader.kpt_color_map
        self.skeleton_map = self.schema_loader.skeleton_map
        self.bbox_colors = self.schema_loader.bbox_colors  # 这是一个 List[Tuple[int, int, int]]

        # 默认的文本和线条颜色/粗细
        self.text_color = (255, 255, 255)  # White
        self.font_scale = 0.7
        self.thickness = 2
        self.keypoint_radius = 5

        # 缓存 bbox 颜色，确保至少有一个颜色
        if not self.bbox_colors:  # If schema_loader returned an empty list
            logger.warning("GenericInferenceDrawer", "No bbox colors loaded from schema. Using default red.")
            self.bbox_colors = [(0, 0, 255)]  # Default to red if none are loaded

    def _get_classification_color(self, classification_id: int) -> Tuple[int, int, int]:
        """根据分类ID选择边界框颜色，从 schema.json 的 bbox_color 列表中获取。"""
        color_at_index = classification_id % len(self.bbox_colors)  # 确保索引在范围内
        return self.bbox_colors[color_at_index]

    @staticmethod
    def _scale_coordinates(coords: Union[Rect, Point, Tuple[float, float]],
                           original_shape: Tuple[int, int],
                           pipeline_input_shape: Tuple[int, int] = (640, 640)) -> Rect | Point | tuple[int, int] | \
                                                                                  tuple[float, float]:
        """
        根据图像从 pipeline 输入尺寸缩放到原始尺寸的比例调整坐标。
        Args:
            coords: Rect, Point 或 (x, y) 元组。
            original_shape: 原始图像的 (height, width)。
            pipeline_input_shape: pipeline 内部处理的图像尺寸 (height, width)。
        Returns:
            缩放后的 Rect, Point 或 (int, int) 元组。
        """
        original_h, original_w = original_shape
        pipeline_h, pipeline_w = pipeline_input_shape
        scale_x = original_w / pipeline_w
        scale_y = original_h / pipeline_h

        if isinstance(coords, Rect):
            return Rect(
                x1=coords.x1 * scale_x, y1=coords.y1 * scale_y,
                x2=coords.x2 * scale_x, y2=coords.y2 * scale_y
            )
        elif isinstance(coords, Point):
            return Point(
                x=int(coords.x * scale_x), y=int(coords.y * scale_y),
                confidence=coords.confidence
            )
        elif isinstance(coords, tuple) and len(coords) == 2:
            return int(coords[0] * scale_x), int(coords[1] * scale_y)
        else:
            return coords  # Return as is if not a recognized type

    def draw_object_detection(self,
                              image: np.ndarray,
                              detection: ObjectDetection,
                              original_image_shape: Tuple[int, int],
                              enable_track_id: bool = True,
                              label_names: Optional[List[str]] = None) -> np.ndarray:
        """
        在图像上绘制一个 ObjectDetection 对象。
        Args:
            image: 要绘制的 NumPy 图像数组 (BGR)。
            detection: ObjectDetection 实例。
            original_image_shape: 原始图像的 (height, width)，用于坐标缩放。
            enable_track_id: 是否在标签中显示 track_id。
            label_names: 分类标签名称列表 (例如 ['person', 'cat'])。
                         如果提供，将显示分类名称，否则显示分类 ID。
        Returns:
            绘制后的图像副本。
        """
        display_image = image.copy()

        # 缩放边界框坐标
        scaled_rect = self._scale_coordinates(detection.rect, original_image_shape)
        x1, y1, x2, y2 = map(int, [scaled_rect.x1, scaled_rect.y1, scaled_rect.x2, scaled_rect.y2])

        # 获取边界框颜色 - 现在根据 classification 来获取
        bbox_color = self._get_classification_color(detection.classification)

        # 构建标签文本
        class_label_display = str(detection.classification)  # 默认显示分类ID
        if label_names and 0 <= detection.classification < len(label_names):
            class_label_display = label_names[detection.classification]  # 如果有标签名，显示名称

        # 标签格式调整
        label_text = ""
        if enable_track_id and detection.track_id != 0:  # 只有当 track_id 非0时才显示 ID
            label_text = f"ID: {detection.track_id} - "
        label_text += f"Class: {class_label_display} Score: {detection.confidence:.2f}"

        # 绘制边界框
        cv2.rectangle(display_image, (x1, y1), (x2, y2), bbox_color, self.thickness)

        # 绘制标签文本 (在边界框上方)
        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness)
        text_x = x1
        text_y = y1 - 10
        if text_y < 0:  # Ensure text is visible
            text_y = y1 + text_size[1] + 10

        cv2.rectangle(display_image, (text_x, text_y - text_size[1] - 5),
                      (text_x + text_size[0], text_y + 5), bbox_color, -1)  # Background for text
        cv2.putText(display_image, label_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.text_color, self.thickness)

        return display_image

    def draw_skeleton(self,
                      image: np.ndarray,
                      skeleton: Skeleton,
                      original_image_shape: Tuple[int, int],
                      enable_track_id: bool = True,
                      label_names: Optional[List[str]] = None,  # 这里的label_names用于分类名称，而非关键点名称
                      enable_pts_names: bool = False,
                      enable_skeleton: bool = True) -> np.ndarray:
        """
        在图像上绘制一个 Skeleton 对象。
        Args:
            image: 要绘制的 NumPy 图像数组 (BGR)。
            skeleton: Skeleton 实例。
            original_image_shape: 原始图像的 (height, width)，用于坐标缩放。
            enable_track_id: 是否在标签中显示 track_id。
            label_names: 分类标签名称列表 (例如 ['girl', 'boy'])。
                         如果提供，将显示分类名称，否则统一显示 'person'。
            enable_pts_names: 如果为 True，将在关键点附近添加名字。
            enable_skeleton: 如果为 True，绘制关键点和骨骼连线。
        Returns:
            绘制后的图像副本。
        """
        display_image = image.copy()

        # 绘制边界框 (继承自 ObjectDetection)
        # 对于 Skeleton，分类标签优先使用提供的 label_names，否则统一为 'person'
        # 这里的 `label_names` 是用于 `ObjectDetection` 的分类名称，不是关键点名称
        effective_label_names = ['person'] if not label_names else label_names
        display_image = self.draw_object_detection(
            display_image,
            skeleton,  # Skeleton 也是 ObjectDetection
            original_image_shape,
            enable_track_id=enable_track_id,
            label_names=effective_label_names
        )

        if not enable_skeleton:
            return display_image

        # 缩放所有关键点
        scaled_points = [
            self._scale_coordinates(p, original_image_shape, pipeline_input_shape=(640, 640))
            for p in skeleton.points
        ]

        # 将关键点转换为字典，以便通过 ID 快速查找
        # 注意: 关键点的 ID 是从 0 到 16，对应 JSON schema 中的 key
        points_by_id: Dict[int, Point] = {i: p for i, p in enumerate(scaled_points)}

        # 绘制骨骼连线
        for link in self.skeleton_map:
            srt_kpt = points_by_id.get(link.srt_kpt_id)
            dst_kpt = points_by_id.get(link.dst_kpt_id)

            if srt_kpt and dst_kpt and srt_kpt.confidence > 0.1 and dst_kpt.confidence > 0.1:  # 仅绘制置信度高的连线
                cv2.line(display_image,
                         (int(srt_kpt.x), int(srt_kpt.y)),
                         (int(dst_kpt.x), int(dst_kpt.y)),
                         link.color, self.thickness)

        # 绘制关键点
        for kpt_id, kp in points_by_id.items():
            if kp.confidence > 0.1:  # 仅绘制置信度高的关键点
                center = (int(kp.x), int(kp.y))
                color_info = self.kpt_color_map.get(kpt_id)
                kpt_color = color_info.color if color_info else (255, 255, 255)  # Default to white if not found

                cv2.circle(display_image, center, self.keypoint_radius, kpt_color, -1)  # -1 means filled circle

                # 绘制关键点名称
                if enable_pts_names and color_info and color_info.name:
                    kpt_name_text = color_info.name
                    cv2.putText(display_image, kpt_name_text,
                                (center[0] + self.keypoint_radius + 2, center[1] + self.keypoint_radius + 2),  # Offset
                                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.6, self.text_color, 1)  # Smaller font

        return display_image

    def draw_object_detections_batch(self,
                                     image: np.ndarray,
                                     detections: List[ObjectDetection],
                                     original_image_shape: Tuple[int, int],
                                     enable_track_id: bool = True,
                                     label_names: Optional[List[str]] = None) -> np.ndarray:
        """
        在图像上批量绘制 ObjectDetection 对象。
        Args:
            image: 要绘制的 NumPy 图像数组 (BGR)。
            detections: ObjectDetection 实例的列表。
            original_image_shape: 原始图像的 (height, width)，用于坐标缩放。
            enable_track_id: 是否在标签中显示 track_id。
            label_names: 分类标签名称列表。
        Returns:
            绘制后的图像副本。
        """
        display_image = image.copy()
        for det in detections:
            display_image = self.draw_object_detection(
                display_image, det, original_image_shape, enable_track_id, label_names
            )
        return display_image

    def draw_skeletons_batch(self,
                             image: np.ndarray,
                             skeletons: List[Skeleton],
                             original_image_shape: Tuple[int, int],
                             enable_track_id: bool = True,
                             label_names: Optional[List[str]] = None,
                             enable_pts_names: bool = False,
                             enable_skeleton: bool = True) -> np.ndarray:
        """
        在图像上批量绘制 Skeleton 对象。
        Args:
            image: 要绘制的 NumPy 图像数组 (BGR)。
            skeletons: Skeleton 实例的列表。
            original_image_shape: 原始图像的 (height, width)，用于坐标缩放。
            enable_track_id: 是否在标签中显示 track_id。
            label_names: 分类标签名称列表。
            enable_pts_names: 如果为 True，将在关键点附近添加名字。
            enable_skeleton: 如果为 True，绘制关键点和骨骼连线。
        Returns:
            绘制后的图像副本。
        """
        display_image = image.copy()
        for skel in skeletons:
            display_image = self.draw_skeleton(
                display_image, skel, original_image_shape, enable_track_id,
                label_names, enable_pts_names, enable_skeleton
            )
        return display_image


"""
import sys
import cv2
import os  # Import os for path manipulation

from pyengine.visualization.inference_drawer import GenericInferenceDrawer
from pyengine.inference.unified_structs.pipeline_converter import convert_pipeline_v1_to_skeletons
from pyengine.inference.c_pipeline.pose_pipeline_v1 import PosePipeline


if __name__ == "__main__":
    # Your shared library path
    LIBRARY_PATH = "/home/user/projects/TrtEngineToolkits/build/lib/libjetson.so"

    # Your TensorRT engine file path
    YOLO_ENGINE = "/opt/models/yolov8n-pose.engine"
    EFFICIENT_ENGINE = "/opt/models/efficientnet_b0_feat_logits.engine"

    # Create GenericInferenceDrawer instance
    drawer = GenericInferenceDrawer("./configs/schema.json")

    pipeline = None
    try:
        pipeline = PosePipeline(LIBRARY_PATH)

        # Register models (call once)
        pipeline.register_models()

        # Create pipeline instance
        pipeline.create_pipeline(
            yolo_engine_path=YOLO_ENGINE,
            efficient_engine_path=EFFICIENT_ENGINE,
            yolo_max_batch=4,
            efficient_max_batch=32,
            yolo_cls_thresh=0.5,
            yolo_iou_thresh=0.4
        )

        # Load images (call once)
        image_paths = [
            "/opt/images/supermarket/customer1.png",
            "/opt/images/supermarket/customer2.png",
            "/opt/images/supermarket/customer3.png",
            "/opt/images/supermarket/customer4.png",
        ]

        # Store original image info, including original dimensions, for drawing
        original_images_info = []
        loaded_images_for_pipeline = []

        print("Loading images for test (once)...")
        for i, path in enumerate(image_paths):
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not load image from {path}. Skipping.")
                continue

            original_h, original_w = img.shape[:2]  # Record original dimensions

            # Resize image to 640x640 for the pipeline
            img_resized_for_pipeline = cv2.resize(img, (640, 640))
            loaded_images_for_pipeline.append(img_resized_for_pipeline)

            original_images_info.append({
                'idx': i,
                'path': path,
                'original_data': img.copy(),  # Store a copy of the original image for drawing
                'original_shape': (original_h, original_w)
            })
            print(
                f"Loaded and resized {path} to {img_resized_for_pipeline.shape[1]}x{img_resized_for_pipeline.shape[0]} (Shape: {img_resized_for_pipeline.shape})"
            )

        if not loaded_images_for_pipeline:
            print("No images loaded. Exiting.")
            sys.exit(0)

        # Command pipeline to perform inference
        raw_pipeline_results = pipeline.process_batched_images(loaded_images_for_pipeline)

        # Convert results to skeleton data
        # raw_pipeline_results is List[Dict], where each Dict corresponds to one image
        # all_skeletons_per_image is List[List[Skeleton]], outer List for images, inner List for Skeleton objects in that image
        all_skeletons_per_image = convert_pipeline_v1_to_skeletons(raw_pipeline_results)

        # --- Drawing Part ---
        print("\nDrawing results...")
        for i, skeletons_for_image in enumerate(all_skeletons_per_image):
            original_image_info = original_images_info[i]
            original_image = original_image_info['original_data']
            original_shape = original_image_info['original_shape']
            image_path = original_image_info['path']

            display_image = original_image.copy()

            # Example: Assuming we only want to draw Skeleton objects
            # If you also want to draw ObjectDetection (non-Skeleton) objects, you'll need additional logic to distinguish
            if skeletons_for_image:
                # Call the new batch drawing function
                display_image = drawer.draw_skeletons_batch(
                    display_image,
                    skeletons_for_image,
                    original_shape,
                    enable_track_id=False,  # Example: No tracking performed here, so no track_id displayed
                    label_names=['Person', 'Salesperson'],  # Example: classification labels
                    enable_pts_names=False,  # Display keypoint names
                    enable_skeleton=True  # Draw skeleton connections and keypoints
                )
            else:
                print(f"No skeletons to draw for image {i + 1} ({os.path.basename(image_path)})")

            # Show each image in its own window
            window_name = f"Result for {os.path.basename(image_path)}"
            cv2.imshow(window_name, display_image)

        # After showing all images, wait for a single key press to close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    except RuntimeError as e:
        print(f"Runtime Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Ensure the pipeline is destroyed when the program exits
        if pipeline:
            pipeline.destroy_pipeline()
            print("Pipeline destroyed.")
"""