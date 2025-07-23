# 文件名: inference_drawer.py

from typing import List, Dict

import cv2
import numpy as np

# 假设这些导入路径在您的项目中是正确的
from pyengine.inference.unified_structs.auxiliary_structs import ExpandedSkeleton, FaceDirection
from pyengine.inference.unified_structs.inference_results import Skeleton, Rect
from pyengine.visualization.schema_loader import SchemaLoader


class InferenceDrawer:
    """
    一个统一、模块化、可配置的推理结果绘制类。
    此版本假定处理的图像尺寸是固定的，并在初始化时指定，
    用于将归一化的坐标转换为像素坐标。
    """

    def __init__(self,
                 schema_loader: SchemaLoader,
                 image_width: int,
                 image_height: int,
                 bbox_confidence_threshold: float = 0.5,
                 kpt_confidence_threshold: float = 0.5,
                 link_confidence_threshold: float = 0.5):
        """
        初始化绘制器。

        Args:
            schema_loader (SchemaLoader): 用于加载可视化配置的实例。
            image_width (int): 目标图像的固定宽度。
            image_height (int): 目标图像的固定高度。
            ...
        """
        self.bbox_conf_thresh = bbox_confidence_threshold
        self.kpt_conf_thresh = kpt_confidence_threshold
        self.link_conf_thresh = link_confidence_threshold
        self.schema = schema_loader

        # --- NEW: 存储固定的图像尺寸 ---
        self.image_width = image_width
        self.image_height = image_height

        self.blink_counter = 0

    def _scale_point(self, point: tuple, image_size=None) -> tuple:
        """将单个归一化的点 (x, y) 缩放到像素坐标"""

        # 如果提供了输入的 image_size, 则将 image_size / (image_width, image_height) 作为缩放因子
        if image_size:
            width, height = image_size
            scale_x = width / self.image_width
            scale_y = height / self.image_height
            return int(point[0] * scale_x), int(point[1] * scale_y)
        else:
            # 不做缩放，直接返回
            return int(point[0]), int(point[1])


    def _scale_rect(self, rect: Rect, image_size) -> tuple:
        """将归一化的矩形 Rect 缩放到像素坐标"""
        # x1 = int(rect.x1 * self.image_width)
        # y1 = int(rect.y1 * self.image_height)
        # x2 = int(rect.x2 * self.image_width)
        # y2 = int(rect.y2 * self.image_height)
        # return x1, y1, x2, y2

        # 如果提供了输入的 image_size, 则将 image_size / (image_width, image_height) 作为缩放因子
        if image_size:
            width, height = image_size
            scale_x = width / self.image_width
            scale_y = height / self.image_height
            return (
                int(rect.x1 * scale_x),
                int(rect.y1 * scale_y),
                int(rect.x2 * scale_x),
                int(rect.y2 * scale_y)
            )
        else:
            # 不做缩放，直接返回
            return (
                int(rect.x1),
                int(rect.y1),
                int(rect.x2),
                int(rect.y2)
            )

    def _draw_bbox(self, image: np.ndarray, skeleton: Skeleton, label_map: Dict = None, highlight_classes: Dict = None):
        """绘制单个边界框和标签，支持闪烁风格"""
        # --- FIX: 使用存储的尺寸进行缩放 ---
        x1, y1, x2, y2 = self._scale_rect(skeleton.rect, image_size=(image.shape[1], image.shape[0]))

        is_highlighted = highlight_classes and skeleton.classification in highlight_classes
        if is_highlighted:
            color1, color2 = highlight_classes[skeleton.classification]
            bbox_color = color1 if self.blink_counter % 2 == 0 else color2
            label = f"ID: {skeleton.classification}"
        else:
            bbox_colors = self.schema.bbox_colors
            bbox_color = bbox_colors[skeleton.classification % len(bbox_colors)]
            label = f"ID: {skeleton.classification}"
            if label_map and skeleton.classification in label_map:
                label = label_map[skeleton.classification]
            label += f" {skeleton.confidence:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), bbox_color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 2)

    def _draw_keypoints(self, image: np.ndarray, skeleton: Skeleton):
        """绘制单个骨架的所有关节点"""
        for i, point in enumerate(skeleton.points):
            if point.confidence > self.kpt_conf_thresh:
                # --- FIX: 使用存储的尺寸进行缩放 ---
                kpt_pos = self._scale_point((point.x, point.y), image_size=(image.shape[1], image.shape[0]))
                kpt_schema = self.schema.kpt_color_map.get(i)
                if kpt_schema:
                    color = kpt_schema.color
                    cv2.circle(image, kpt_pos, 5, color, -1)

    def _draw_skeleton_links(self, image: np.ndarray, skeleton: Skeleton):
        """绘制单个骨架的骨骼连接线"""
        for link_schema in self.schema.skeleton_map:
            p1_idx, p2_idx = link_schema.srt_kpt_id, link_schema.dst_kpt_id

            if p1_idx < len(skeleton.points) and p2_idx < len(skeleton.points):
                p1 = skeleton.points[p1_idx]
                p2 = skeleton.points[p2_idx]

                if p1.confidence > self.link_conf_thresh and p2.confidence > self.link_conf_thresh:
                    # --- FIX: 使用存储的尺寸进行缩放 ---
                    p1_pos = self._scale_point((p1.x, p1.y), image_size=(image.shape[1], image.shape[0]))
                    p2_pos = self._scale_point((p2.x, p2.y), image_size=(image.shape[1], image.shape[0]))
                    color = link_schema.color
                    cv2.line(image, p1_pos, p2_pos, color, 2)

    def _draw_face_direction(self, image: np.ndarray, skeleton: ExpandedSkeleton):
        """绘制单个面部朝向向量"""
        if hasattr(skeleton, 'direction_type') and skeleton.direction_type != FaceDirection.Unknown:
            # --- FIX: 使用存储的尺寸进行缩放 ---
            origin_x, origin_y = self._scale_point(skeleton.direction_origin, image_size=(image.shape[1], image.shape[0]))

            # 向量本身是方向和长度，不应缩放
            vec_x, vec_y = skeleton.direction_vector
            end_x = int(origin_x + vec_x * 50)
            end_y = int(origin_y + vec_y * 50)

            cv2.arrowedLine(image, (origin_x, origin_y), (end_x, end_y), (0, 255, 255), 2, tipLength=0.3)
            cv2.circle(image, (origin_x, origin_y), 5, (0, 255, 0), -1)

    def draw_inference(self,
                       image: np.ndarray,
                       skeletons: List,
                       label_map: Dict = None,
                       draw_bbox: bool = True,
                       draw_kpts: bool = True,
                       draw_links: bool = True,
                       draw_direction: bool = True,
                       highlight_classes: Dict = None) -> np.ndarray:
        """
        在图像上绘制所有给定的骨架信息，支持高亮闪烁。
        """
        display_image = image.copy()

        for skeleton in skeletons:
            if skeleton.confidence < self.bbox_conf_thresh:
                continue

            if draw_bbox:
                self._draw_bbox(display_image, skeleton, label_map, highlight_classes)

            if draw_links:
                self._draw_skeleton_links(display_image, skeleton)

            if draw_kpts:
                self._draw_keypoints(display_image, skeleton)

            if draw_direction and isinstance(skeleton, ExpandedSkeleton):
                self._draw_face_direction(display_image, skeleton)

        self.blink_counter += 1
        return display_image