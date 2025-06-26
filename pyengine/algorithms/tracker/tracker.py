# tracker.py
from collections import deque
from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter

from pyengine.inference.unified_structs.inference_results import ObjectDetection, Rect


class UnifiedTrack:
    """
    一个统一的 Track 类，可以用于 DeepSORT 和 SORT。
    通过 'use_reid' 参数控制是否存储和使用 Re-ID 特征。
    """
    _next_id = 0  # 静态变量，用于生成唯一的轨迹ID

    def __init__(self, detection: ObjectDetection, use_reid: bool = True):
        """
        初始化一个新的轨迹。
        Args:
            detection (ObjectDetection): 用于初始化轨迹的第一个检测结果。
            use_reid (bool): 如果为 True，将存储和更新 Re-ID 特征。
                             用于 DeepSORT 时设为 True，用于 SORT 时设为 False。
        """
        self.track_id = UnifiedTrack._next_id
        UnifiedTrack._next_id += 1

        self.use_reid = use_reid

        # Kalman Filter for state estimation
        # 8D 状态: [x, y, aspect_ratio, height, vx, vy, vaspect_ratio, vheight]
        # x,y: 边界框中心坐标
        # aspect_ratio: 边界框宽高比 (width / height)
        # height: 边界框高度
        # vx,vy,vaspect_ratio,vheight: 对应的速度
        self.kf = KalmanFilter(dim_x=8, dim_z=4)  # 8D 状态，4D 测量 (x, y, a, h)

        # 状态转移矩阵 (F): 假设恒定速度模型
        dt = 1.0  # 假设帧间隔为1
        self.kf.F = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])

        # 测量矩阵 (H): 我们直接测量 [x, y, a, h]
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])

        # 过程噪声协方差 (Q): 模型的噪声，需要根据实际情况调优
        # 通常速度分量的噪声会大一些
        self.kf.Q = np.diag([1., 1., 1., 1., 0.1, 0.1, 0.1, 0.1]) * 0.01

        # 测量噪声协方差 (R): 检测器的噪声
        self.kf.R = np.diag([1., 1., 1., 1.]) * 10

        # 初始状态协方差 (P): 初始不确定性
        self.kf.P *= 1000.

        # 从第一个检测结果初始化 Kalman 滤波器的状态 x
        # 你的 ObjectDetection 现在包含一个 Rect 对象
        box = detection.rect # 获取 Rect 对象
        center_x = (box.x1 + box.x2) / 2
        center_y = (box.y1 + box.y2) / 2
        width = box.x2 - box.x1
        height = box.y2 - box.y1
        aspect_ratio = width / height if height > 0 else 0  # 避免除以零

        # 初始速度设为0
        self.kf.x = np.array([center_x, center_y, aspect_ratio, height, 0, 0, 0, 0]).reshape((8, 1))

        if self.use_reid:
            # 存储最近的 Re-ID 特征，用于 DeepSORT
            # detection.features 现在是一个 List[float]，需要转换为 np.array
            self.features = deque([np.array(detection.features)], maxlen=100) if detection.features else deque(maxlen=100)
        else:
            self.features = None  # SORT模式下不存储特征

        self.time_since_update = 0  # 距离上次成功更新的帧数
        self.hits = 1  # 轨迹被检测命中的总次数
        self.age = 0  # 轨迹存在的总帧数

    def predict(self):
        """预测轨迹在下一帧的状态。"""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: ObjectDetection):
        """
        根据新的检测结果更新轨迹状态。
        Args:
            detection (ObjectDetection): 匹配到的新检测结果。
        """
        # 你的 ObjectDetection 现在包含一个 Rect 对象
        box = detection.rect # 获取 Rect 对象
        center_x = (box.x1 + box.x2) / 2
        center_y = (box.y1 + box.y2) / 2
        width = box.x2 - box.x1
        height = box.y2 - box.y1
        aspect_ratio = width / height if height > 0 else 0

        measurement = np.array([center_x, center_y, aspect_ratio, height]).reshape((4, 1))

        self.kf.update(measurement)  # 更新 Kalman 滤波器状态

        if self.use_reid and detection.features:
            self.features.append(np.array(detection.features))  # 确保转换为 np.array

        self.hits += 1
        self.time_since_update = 0  # 重置未更新计数

    def get_state(self) -> Rect:
        """从 Kalman 滤波器的当前状态中获取预测的边界框。"""
        x, y, a, h = self.kf.x[:4].flatten()
        w = a * h  # 从宽高比和高度计算宽度
        # 返回 Rect 对象
        return Rect(x1=x - w / 2, y1=y - h / 2, x2=x + w / 2, y2=y + h / 2)

    def get_feature(self) -> Optional[np.ndarray]:
        """
        如果 use_reid 为 True，则返回平均 Re-ID 特征，否则返回 None。
        """
        if self.use_reid and self.features and len(self.features) > 0:
            return np.mean(list(self.features), axis=0)
        return None

    def is_confirmed(self, min_hits: int) -> bool:
        """检查轨迹是否被确认（达到最小命中次数）。"""
        return self.hits >= min_hits

    def is_deleted(self, max_age: int) -> bool:
        """检查轨迹是否应该被删除（长时间未更新）。"""
        return self.time_since_update > max_age