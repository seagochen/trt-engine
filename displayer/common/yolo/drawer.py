from typing import List, Union, Tuple

import cv2
import numpy as np

from common.yolo.facial_orientation import FacialOrientation2D
from common.yolo.schema_loader import SchemaLoader
from common.yolo.ultralytics_results_wrapper import YoloPose, Yolo, YoloPoseSorted, YoloSorted


class Drawer:
    """绘图工具类，负责在图像上绘制骨骼、关键点和边界框。"""

    def __init__(self, schema_loader: SchemaLoader):
        self.kpt_color_map = schema_loader.kpt_color_map
        self.skeleton_map = schema_loader.skeleton_map
        self.bbox_colors = schema_loader.bbox_colors


    @staticmethod
    def draw_color_bbox(image: np.ndarray, 
                        text: str, text_color: Tuple[int, int, int],
                        bbox_coords: Tuple[int, int, int, int], bbox_color: Tuple[int, int, int]):

        # 绘制边界框
        cv2.rectangle(image, (bbox_coords[0], bbox_coords[1]),
                      (bbox_coords[2], bbox_coords[3]), bbox_color, 1)

        # 绘制文字
        if text is not None:
            # 获取文字大小
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # 绘制信息背景
            cv2.rectangle(image, (bbox_coords[0], bbox_coords[1] - text_height - 5),
                          (bbox_coords[0] + text_width, bbox_coords[1]), bbox_color, -1)

            # 显示文字
            cv2.putText(image, text, (bbox_coords[0], bbox_coords[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)


    def draw_skeletons(self, frame: np.ndarray, results: List[Union[YoloPose, YoloPoseSorted]],
                       conf_threshold: float = 0.5, bbox_style: str = "normal",
                       show_skeletons: bool = True, show_pts: bool = True,
                       show_pts_name: bool = True) -> np.ndarray:
        """
        在图像上绘制骨骼。

        Args:
            frame: cv2 图像
            results: YoloPose 或 YoloPoseSorted 对象列表
            conf_threshold: 置信度阈值
            bbox_style: 边界框风格，可选值：

            - "normal" 所有被识别到的人物都绘制蓝色边界框，
            - "chromatic" 当results为YoloPoseSorted时，根据id绘制不同颜色的边界框
            - "none" 不绘制边界框

            show_skeletons: 是否绘制骨骼
            show_pts: 是否绘制关键点
            show_pts_name: 是否显示关键点名称

        Returns:
            绘制后的图像
        """

        frame_copy = frame.copy()

        for pose in results:

            # 首先绘制关键点
            if show_pts:
                for idx, kpt in enumerate(pose.pts):
                    if kpt.conf > conf_threshold and idx in self.kpt_color_map:
                        kp = self.kpt_color_map[idx]
                        cv2.circle(frame_copy, (int(kpt.x), int(kpt.y)), 3, kp.color, -1)

                        # 显示关键点名称
                        if show_pts_name:
                            cv2.putText(frame_copy, kp.name, (int(kpt.x), int(kpt.y - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, kp.color, 1)

            # 绘制骨骼
            if show_skeletons:
                for bone in self.skeleton_map:
                    srt_kp = pose.pts[bone.srt_kpt_id]
                    dst_kp = pose.pts[bone.dst_kpt_id]

                    if all([
                        srt_kp.conf > conf_threshold,
                        dst_kp.conf > conf_threshold,
                        srt_kp.x > 0, srt_kp.y > 0,
                        dst_kp.x > 0, dst_kp.y > 0
                    ]):
                        cv2.line(frame_copy, (int(srt_kp.x), int(srt_kp.y)),
                                 (int(dst_kp.x), int(dst_kp.y)), bone.color, 1)


            # 绘制边界框
            if bbox_style == "normal":

                # 创建文字
                text = f"{pose.conf:.2f}"

                # 绘制信息背景
                self.draw_color_bbox(frame_copy, text, (255, 255, 255),
                                     (pose.lx, pose.ly, pose.rx, pose.ry), (255, 0, 0))

            elif bbox_style == "chromatic":

                # 创建文字
                text = f"{pose.oid}: {pose.conf:.2f}"

                # 绘制信息背景
                self.draw_color_bbox(frame_copy, text, (255, 255, 255),
                                     (pose.lx, pose.ly, pose.rx, pose.ry), self.bbox_colors[pose.oid % len(self.bbox_colors)])

        return frame_copy


    # def draw_facial_vectors_2d(self, frame: np.ndarray, orientation_vectors: List[FacialOrientation2D],
    #                            different_vectors: bool = False, show_legend: bool = False) -> np.ndarray:
    #     """
    #     在图像上绘制面部方向向量。

    #     :param frame: cv2 图像
    #     :param orientation_vectors: FacialOrientation2D 对象列表
    #     :param different_vectors: 是否使用不同颜色的向量
    #     :param show_legend: 是否显示图例
    #     :return: 绘制后的图像
    #     """
    #     frame_copy = frame.copy()

    #     for idx, vector in enumerate(orientation_vectors):
    #         # 确定向量颜色
    #         if different_vectors:
    #             vector_color = self.bbox_colors[idx % len(self.bbox_colors)]
    #         else:
    #             vector_color = (255, 0, 0)  # 默认蓝色

    #         # 绘制箭头
    #         cv2.arrowedLine(frame_copy, (vector.origin_x, vector.origin_y), (vector.dest_x, vector.dest_y),
    #                         vector_color, 2)

    #         # 绘制图例
    #         if show_legend:
    #             face_direction = str(vector)
    #             cv2.putText(frame_copy, face_direction, (vector.origin_x + 5, vector.origin_y + 15),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, vector_color, 1)

    #     return frame_copy

    def draw_objects(self, frame: np.ndarray, results: List[Union[Yolo, YoloSorted]], 
                     labels: List[Union[str, int]] = None,
                     conf_threshold: float = 0.5, bbox_style: str = "normal") -> np.ndarray:
        """
        在图像上绘制边界框。
        """
        frame_copy = frame.copy()

        for obj in results:

            # bboxの色を決定
            if bbox_style == "normal":
                bbox_color = (255, 0, 0)
            elif bbox_style == "chromatic":
                bbox_color = self.bbox_colors[obj.oid % len(self.bbox_colors)]
            elif bbox_style == "by_class":
                bbox_color = self.bbox_colors[obj.cls % len(self.bbox_colors)]
            else:
                bbox_color = (255, 0, 0)

            # 绘制边界框
            if obj.conf > conf_threshold:
                if labels is not None and obj.cls < len(labels):
                    text = f"{labels[obj.cls]}: {obj.conf:.2f} - {obj.oid}"
                    self.draw_color_bbox(frame_copy, text, (255, 255, 255),
                                        (obj.lx, obj.ly, obj.rx, obj.ry), bbox_color)
                
                elif labels is not None and obj.cls >= len(labels):
                    text = f"Unknown: {obj.conf:.2f} - {obj.oid}"
                    self.draw_color_bbox(frame_copy, text, (255, 255, 255),
                                        (obj.lx, obj.ly, obj.rx, obj.ry), bbox_color)
                
                elif labels is None and bbox_style == "chromatic":
                    text = f"{obj.oid}: {obj.conf:.2f}"
                    self.draw_color_bbox(frame_copy, text, (255, 255, 255),
                                        (obj.lx, obj.ly, obj.rx, obj.ry), bbox_color)

                else:
                    self.draw_color_bbox(frame_copy, None, (255, 255, 255),
                                        (obj.lx, obj.ly, obj.rx, obj.ry), bbox_color)

        return frame_copy


    def draw_facial_vectors_2d(self, frame: np.ndarray, orientation_vectors: List[FacialOrientation2D],
                            different_vectors: bool = False, show_legend: bool = False) -> np.ndarray:
        """
        在图像上绘制面部方向向量。

        :param frame: cv2 图像
        :param orientation_vectors: FacialOrientation2D 对象列表
        :param different_vectors: 是否使用不同颜色的向量
        :param show_legend: 是否显示图例
        :return: 绘制后的图像
        """
        frame_copy = frame.copy()

        for idx, vector in enumerate(orientation_vectors):
            # 确定向量颜色
            if different_vectors:
                vector_color = self.bbox_colors[idx % len(self.bbox_colors)]
            else:
                vector_color = (255, 0, 0)  # 默认蓝色

            # 判断是否正面
            if vector.orientation == 0:
                # 绘制一个中空圆和圆点
                cv2.circle(frame_copy, (vector.origin_x, vector.origin_y), 10, vector_color, 2)  # 中空圆
                cv2.circle(frame_copy, (vector.origin_x, vector.origin_y), 3, vector_color, -1)  # 实心圆点
            elif vector.origin_x == vector.dest_x and vector.origin_y == vector.dest_y:
                # 绘制一个点（非正面但起点和终点相同）
                cv2.circle(frame_copy, (vector.origin_x, vector.origin_y), 3, vector_color, -1)
            else:
                # 绘制箭头
                cv2.arrowedLine(frame_copy, (vector.origin_x, vector.origin_y), (vector.dest_x, vector.dest_y),
                                vector_color, 2)

            # 绘制图例
            if show_legend:
                face_direction = str(vector)
                cv2.putText(frame_copy, face_direction, (vector.origin_x + 5, vector.origin_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, vector_color, 1)

        return frame_copy
