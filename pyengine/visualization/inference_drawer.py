# inference_drawer.py
from typing import List, Dict, Tuple, Literal
import cv2
import numpy as np

from pyengine.font import text_painter
from pyengine.inference.unified_structs.auxiliary_structs import ExpandedSkeleton, FaceDirection
from pyengine.inference.unified_structs.inference_results import Skeleton
from pyengine.visualization.scheme_loader import SchemeLoader
from pyengine.utils import scale_utils


CoordSpace = Literal["pixel", "model"]


class InferenceDrawer:
    """
    统一、模块化、可配置的推理结果绘制类。

    重要说明：
    - coord_space='pixel' ：输入坐标已经是像素坐标(相对于当前 image 的尺寸)，不再缩放，直接画(适合全图 merged)。
    - coord_space='model' ：输入坐标在模型坐标系里(例如 640x640)，会按 (model_input_size -> 当前 image.shape) 缩放后绘制。

    怎么用
    1) 全图(merged，像素坐标)

    app_pipeline_v2.py 里初始化 Drawer 改为(或保持默认，也 OK，因为默认就是 pixel)：

    drawer = InferenceDrawer(
        scheme_loader=SchemeLoader(SCHEME_CONFIG),
        image_width=3040, image_height=1368,
        coord_space="pixel"  # ★ 关键
    )


    然后直接：

    vis = drawer.draw_inference(frame, merged, draw_bbox=True, draw_kpts=True, draw_links=True, draw_direction=False)


    这会不缩放，按原坐标绘制。

    2) 子图(模型坐标 640×640)

    如果你想单独显示每个 640×640 的 crop 推理结果(传入的 skeleton 仍在 640 坐标)，就：

    drawer_640 = InferenceDrawer(
        scheme_loader=SchemeLoader(SCHEME_CONFIG),
        image_width=640, image_height=640,
        coord_space="model",           # ★ 使用模型坐标模式
        model_input_size=(640, 640)
    )

    resized_crop = cv2.resize(crop, (640, 640))
    img = drawer_640.draw_inference(resized_crop, sks_in_this_tile, ...)
    """

    def __init__(self,
                 scheme_loader: SchemeLoader,
                 image_width: int,
                 image_height: int,
                 *,
                 bbox_confidence_threshold: float = 0.5,
                 kpt_confidence_threshold: float = 0.5,
                 coord_space: CoordSpace = "pixel",
                 model_input_size: Tuple[int, int] = (640, 640)):
        self.bbox_conf_thresh = bbox_confidence_threshold
        self.kpt_conf_thresh = kpt_confidence_threshold
        self.schema = scheme_loader

        # 仅作信息目的(不强依赖)
        self.image_width = image_width
        self.image_height = image_height

        # ★ 新增：坐标系模式
        self.coord_space: CoordSpace = coord_space
        self.model_w, self.model_h = model_input_size

        self.blink_counter = 0

    # ---------- 坐标映射工具 ----------
    def _map_point(self, image: np.ndarray, x: float, y: float) -> Tuple[int, int]:
        """
        把输入点 (x, y) 映射到当前 image 的像素坐标。
        - pixel 模式：直接取整；
        - model 模式：用模型尺寸 -> image.shape 做缩放。
        """
        if self.coord_space == "pixel":
            return int(round(x)), int(round(y))
        else:
            # 按模型尺寸 → 当前 image 尺寸缩放
            p = scale_utils.scale_euler_pt(
                src_width=self.model_w, src_height=self.model_h,
                dst_width=image.shape[1], dst_height=image.shape[0],
                point=(int(x), int(y))
            )
            return int(p[0]), int(p[1])

    def _map_rect(self, image: np.ndarray, rect) -> Tuple[int, int, int, int]:
        if self.coord_space == "pixel":
            return int(round(rect.x1)), int(round(rect.y1)), int(round(rect.x2)), int(round(rect.y2))
        else:
            r = scale_utils.scale_sk_rect(
                src_width=self.model_w, src_height=self.model_h,
                dst_width=image.shape[1], dst_height=image.shape[0],
                rect=rect
            )
            return int(r.x1), int(r.y1), int(r.x2), int(r.y2)  # ← 读 dataclass 字段

    # ---------- 具体绘制 ----------
    def _draw_bbox(self, image: np.ndarray, skeleton: Skeleton, bbox_color: Tuple[int, int, int] = None):
        x1, y1, x2, y2 = self._map_rect(image, skeleton.rect)

        if bbox_color is None:
            bbox_colors = self.schema.bbox_colors
            bbox_color = bbox_colors[skeleton.classification % len(bbox_colors)]

        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, 2)

    def _draw_keypoints(self, image: np.ndarray, skeleton: Skeleton):
        for i, point in enumerate(skeleton.points):
            if point.confidence > self.kpt_conf_thresh:
                kx, ky = self._map_point(image, point.x, point.y)
                kpt_schema = self.schema.kpt_color_map.get(i)
                if kpt_schema:
                    color = kpt_schema.color
                    cv2.circle(image, (kx, ky), 5, color, -1)

    def _draw_skeleton_links(self, image: np.ndarray, skeleton: Skeleton):
        for link_schema in self.schema.skeleton_map:
            p1_idx, p2_idx = link_schema.srt_kpt_id, link_schema.dst_kpt_id
            if p1_idx < len(skeleton.points) and p2_idx < len(skeleton.points):
                p1 = skeleton.points[p1_idx]
                p2 = skeleton.points[p2_idx]
                if p1.confidence > self.kpt_conf_thresh and p2.confidence > self.kpt_conf_thresh:
                    x1, y1 = self._map_point(image, p1.x, p1.y)
                    x2, y2 = self._map_point(image, p2.x, p2.y)
                    cv2.line(image, (x1, y1), (x2, y2), link_schema.color, 2)

    @staticmethod
    def _draw_face_direction(image: np.ndarray, skeleton: ExpandedSkeleton):
        if skeleton.direction_type != FaceDirection.Unknown:
            origin_x, origin_y = int(round(skeleton.direction_origin[0])), int(round(skeleton.direction_origin[1]))
            # 方向向量不缩放
            vec_x, vec_y = skeleton.direction_vector
            end_x = int(skeleton.direction_modulus * vec_x + origin_x)
            end_y = int(skeleton.direction_modulus * vec_y + origin_y)
            cv2.arrowedLine(image, (origin_x, origin_y), (end_x, end_y),
                            (0, 255, 255), 2, tipLength=0.3)

    def _draw_label(self,
                    image: np.ndarray,
                    skeleton: ExpandedSkeleton,
                    label_map: Dict = None,
                    background_color: Tuple[int, int, int] = None,
                    append_track_id: bool = False):

        if label_map and skeleton.classification in label_map.keys():
            label = label_map[skeleton.classification]
        else:
            label = f"Class {skeleton.classification}"
        if append_track_id and skeleton.track_id is not None:
            label = f"#{skeleton.track_id} {label}"
        label = f"{label}-{skeleton.confidence:.2f}"

        x1, y1 = self._map_point(image, skeleton.rect.x1, skeleton.rect.y1)

        if background_color is None:
            bbox_colors = self.schema.bbox_colors
            bbox_color = bbox_colors[skeleton.classification % len(bbox_colors)]
        else:
            bbox_color = background_color

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), bbox_color, -1)
        text_painter.draw_text(image, label, (x1, y1 - h - 5), font_scale=0.5, thickness=1)

    def _draw_bbox_with_highlight(self, image: np.ndarray, skeleton: ExpandedSkeleton, highlight_classname: str):
        x1, y1, x2, y2 = self._map_rect(image, skeleton.rect)
        bbox_highlight_colors = self.schema.highlight_colors
        highlight_color = bbox_highlight_colors.get(highlight_classname, [[255, 0, 0], [255, 255, 255]])

        if self.blink_counter % 2 == 0:
            cv2.rectangle(image, (x1, y1), (x2, y2), highlight_color[1], 2)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), highlight_color[0], 2)

    # ---------- 外部接口 ----------
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
        display_image = image.copy()
        if skeleton.confidence < self.bbox_conf_thresh:
            return display_image
        self._draw_bbox(display_image, skeleton, bbox_color=bbox_color)
        return display_image
