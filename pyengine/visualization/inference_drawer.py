# 文件名: inference_drawer.py

from typing import List, Dict, Tuple

import cv2
import numpy as np

from pyengine.font import text_painter
# 假设这些导入路径在您的项目中是正确的
from pyengine.inference.unified_structs.auxiliary_structs import ExpandedSkeleton, FaceDirection
from pyengine.inference.unified_structs.inference_results import Skeleton
from pyengine.visualization.scheme_loader import SchemeLoader
from pyengine.utils.mapper import scale_utils


class InferenceDrawer:
    """
    一个统一、模块化、可配置的推理结果绘制类。
    此版本假定处理的图像尺寸是固定的，并在初始化时指定，
    用于将归一化的坐标转换为像素坐标。
    """

    def __init__(self,
                 scheme_loader: SchemeLoader,
                 image_width: int,
                 image_height: int,
                 bbox_confidence_threshold: float = 0.5,
                 kpt_confidence_threshold: float = 0.5):
        """
        初始化绘制器。

        Args:
            scheme_loader (SchemeLoader): 用于加载可视化配置的实例。
            image_width (int): 目标图像的固定宽度。
            image_height (int): 目标图像的固定高度。
            ...
        """
        self.bbox_conf_thresh = bbox_confidence_threshold
        self.kpt_conf_thresh = kpt_confidence_threshold
        self.schema = scheme_loader

        # --- NEW: 存储固定的图像尺寸 ---
        self.image_width = image_width
        self.image_height = image_height

        self.blink_counter = 0

    def _draw_bbox(self, image: np.ndarray, skeleton: Skeleton, bbox_color: Tuple[int, int, int] = None):
        """
        绘制单个边界框和标签，支持闪烁风格
        """
        # 使用存储的尺寸进行缩放
        # x1, y1, x2, y2 = self._scale_rect(skeleton.rect, image_size=(image.shape[1], image.shape[0]))
        x1, y1, x2, y2 = scale_utils.scale_rect(
            src_width=640, src_height=640,
            dst_width=image.shape[1], dst_height=image.shape[0],
            rect=skeleton.rect
        )

        # 根据目标所属类别，选择bbox的颜色
        if bbox_color is None:
            bbox_colors = self.schema.bbox_colors
            bbox_color = bbox_colors[skeleton.classification % len(bbox_colors)]

        # 绘制bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, 2)

    def _draw_keypoints(self, image: np.ndarray, skeleton: Skeleton):
        """
        绘制单个骨架的所有关节点
        """
        for i, point in enumerate(skeleton.points):
            if point.confidence > self.kpt_conf_thresh:
                # 使用存储的尺寸进行缩放
                # kpt_pos = self._scale_point((point.x, point.y), image_size=(image.shape[1], image.shape[0]))
                kpt_pos = scale_utils.scale_point(
                    src_width=640, src_height=640,
                    dst_width=image.shape[1], dst_height=image.shape[0],
                    point=(point.x, point.y)
                )
                kpt_schema = self.schema.kpt_color_map.get(i)
                if kpt_schema:
                    color = kpt_schema.color
                    cv2.circle(image, kpt_pos, 5, color, -1)

    def _draw_skeleton_links(self, image: np.ndarray, skeleton: Skeleton):
        """
        绘制单个骨架的骨骼连接线
        """
        for link_schema in self.schema.skeleton_map:
            p1_idx, p2_idx = link_schema.srt_kpt_id, link_schema.dst_kpt_id

            if p1_idx < len(skeleton.points) and p2_idx < len(skeleton.points):
                p1 = skeleton.points[p1_idx]
                p2 = skeleton.points[p2_idx]

                if p1.confidence > self.kpt_conf_thresh and p2.confidence > self.kpt_conf_thresh:
                    # 使用存储的尺寸进行缩放
                    # p1_pos = self._scale_point((p1.x, p1.y), image_size=(image.shape[1], image.shape[0]))
                    # p2_pos = self._scale_point((p2.x, p2.y), image_size=(image.shape[1], image.shape[0]))
                    p1_pos = scale_utils.scale_point(
                        src_width=640, src_height=640,
                        dst_width=image.shape[1], dst_height=image.shape[0],
                        point=(p1.x, p1.y)
                    )
                    p2_pos = scale_utils.scale_point(
                        src_width=640, src_height=640,
                        dst_width=image.shape[1], dst_height=image.shape[0],
                        point=(p2.x, p2.y)
                    )
                    color = link_schema.color
                    cv2.line(image, p1_pos, p2_pos, color, 2)

    @staticmethod
    def _draw_face_direction(image: np.ndarray, skeleton: ExpandedSkeleton):
        """
        绘制单个面部朝向向量
        """
        if skeleton.direction_type != FaceDirection.Unknown:
            # 使用存储的尺寸进行缩放
            origin_x, origin_y = scale_utils.scale_point(
                src_width=640, src_height=640,
                dst_width=image.shape[1], dst_height=image.shape[0],
                point=skeleton.direction_origin
            )

            # 向量本身是方向和长度，不应缩放
            vec_x, vec_y = skeleton.direction_vector
            end_x = int(skeleton.direction_modulus * vec_x + origin_x)
            end_y = int(skeleton.direction_modulus * vec_y + origin_y)
            # end_x = int(origin_x + vec_x * 50)
            # end_y = int(origin_y + vec_y * 50)

            # 绘制箭头
            cv2.arrowedLine(image, (origin_x, origin_y), (end_x, end_y),
                            (0, 255, 255), 2, tipLength=0.3)
            # cv2.circle(image, (origin_x, origin_y), 5, (0, 255, 0), -1)

    def _draw_label(self,
                    image: np.ndarray,
                    skeleton: ExpandedSkeleton,
                    label_map: Dict = None,
                    background_color: Tuple[int, int, int] = None,
                    append_track_id: bool = False):

        """
        在图像上绘制标签文本。
        """
        if label_map and skeleton.classification in label_map.keys():
            label = label_map[skeleton.classification]
        else:
            label = f"Class {skeleton.classification}"
        
        if append_track_id and skeleton.track_id is not None:
            label = f"#{skeleton.track_id} {label}"

        # 加上置信度信息
        label = f"{label}-{skeleton.confidence:.2f}"

        # 使用存储的尺寸进行缩放
        # x1, y1 = self._scale_point((skeleton.rect.x1, skeleton.rect.y1), image_size=(image.shape[1], image.shape[0]))
        x1, y1 = scale_utils.scale_point(
            src_width=640, src_height=640,
            dst_width=image.shape[1], dst_height=image.shape[0],
            point=(skeleton.rect.x1, skeleton.rect.y1)
        )

        # 选择标签颜色
        if background_color is None:
            bbox_colors = self.schema.bbox_colors
            bbox_color = bbox_colors[skeleton.classification % len(bbox_colors)]
        else:
            bbox_color = background_color

        # 绘制标签背景和文本
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), bbox_color, -1)
        # cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 2)
        text_painter.draw_text(image, label, (x1, y1 - h - 5), font_scale=0.5, thickness=1)

    def _draw_bbox_with_highlight(self, image: np.ndarray, skeleton: ExpandedSkeleton, highlight_classname:str):
        """
        绘制带有高亮的边界框。
        """
        # 使用存储的尺寸进行缩放
        # x1, y1, x2, y2 = self._scale_rect(skeleton.rect, image_size=(image.shape[1], image.shape[0]))
        x1, y1, x2, y2 = scale_utils.scale_rect(
            src_width=640, src_height=640,
            dst_width=image.shape[1], dst_height=image.shape[0],
            rect=skeleton.rect
        )

        # 选择高亮颜色
        bbox_highlight_colors = self.schema.highlight_colors
        highlight_color = bbox_highlight_colors.get(highlight_classname, [[255, 0, 0], [255, 255, 255]])

        # 绘制高亮边界框
        if self.blink_counter % 2 == 0:
            cv2.rectangle(image, (x1, y1), (x2, y2), highlight_color[1], 2)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), highlight_color[0], 2)
        

    def draw_inference(self,
                       image: np.ndarray,
                       skeletons: List,
                       draw_bbox: bool = True,
                       draw_kpts: bool = True,
                       draw_links: bool = True,
                       draw_direction: bool = True,
                       label_map: Dict = None,
                       enable_track_id: bool = False,
                       bbox_color: Tuple[int, int, int] = None,
                       highlight_classes: List[Tuple[bool, str]] = None) -> np.ndarray:
        """
        在图像上绘制所有给定的骨架信息，支持高亮闪烁。
        """
        display_image = image.copy()

        for idx, skeleton in enumerate(skeletons):
            if skeleton.confidence < self.bbox_conf_thresh:
                continue

            if draw_bbox:
                self._draw_bbox(display_image, skeleton, bbox_color=bbox_color)

            if draw_links:
                self._draw_skeleton_links(display_image, skeleton)

            if draw_kpts:
                self._draw_keypoints(display_image, skeleton)

            if draw_direction and isinstance(skeleton, ExpandedSkeleton):
                self._draw_face_direction(display_image, skeleton)

            if label_map:
                self._draw_label(display_image, skeleton, label_map,
                                 background_color=bbox_color,
                                 append_track_id=enable_track_id)

            if highlight_classes:
                enable_highlight, highlight_classname = highlight_classes[idx]
                if enable_highlight:
                    self._draw_bbox_with_highlight(display_image, skeleton, highlight_classname)

        self.blink_counter += 1
        return display_image
    
    def draw_bbox_with_custom_color(self,
                                  image: np.ndarray,
                                  skeleton: ExpandedSkeleton,
                                  bbox_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        绘制边界框，使用自定义颜色。
        """
        display_image = image.copy()

        if skeleton.confidence < self.bbox_conf_thresh:
            return display_image

        self._draw_bbox(display_image, skeleton, bbox_color=bbox_color)

        return display_image
    