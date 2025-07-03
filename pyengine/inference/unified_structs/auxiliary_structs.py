from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Tuple
from pyengine.inference.unified_structs.inference_results import Skeleton, Rect, Point


class FaceDirection(IntEnum):
    Front = 0
    Left = 1
    Right = 2
    Back = 3
    Unknown = -1

    def __str__(self): return self.name.lower()

    @classmethod
    def from_value(cls, value): return cls(value)


class Pose(IntEnum):
    Standing = 0
    Bending = 1
    Sitting = 2
    Unknown = -1

    def __str__(self): return self.name.lower()

    @classmethod
    def from_value(cls, value): return cls(value)


@dataclass
class ExtendedSkeleton(Skeleton):
    # 用于标记姿态
    pose: Pose = Pose.Unknown

    # (修改) 修正拼写错误 directionn_type -> direction
    direction: FaceDirection = FaceDirection.Unknown

    # 关于面部朝向的其他辅助信息，例如角度，向量，模长
    direction_angle: float = 0.0
    direction_modulus: float = 0.0
    direction_vector: Tuple[float, float] = (0.0, 0.0)  # Will be stored as JSON string
    direction_origin: Tuple[float, float] = (0.0, 0.0)  # Will be stored as JSON string

    def __post_init__(self):
        """
        在对象初始化后，确保 direction 和 pose 是正确的枚举类型。
        """
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

        # (修改) 使用修正后的字段名 self.direction
        if not isinstance(self.direction, FaceDirection):
            self.direction = FaceDirection.from_value(self.direction)
        if not isinstance(self.pose, Pose):
            self.pose = Pose.from_value(self.pose)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtendedSkeleton":
        """
        从字典创建实例。
        这个方法是必需的，以确保反序列化逻辑链的完整性。
        它调用父类的方法来处理所有嵌套对象的转换，
        同时确保最终的实例是作为 ExtendedSkeleton 创建的，
        从而能够正确接收 pose 和 direction 等新字段。
        """
        # Handle enum conversion for pose and direction if they come as raw values
        if 'pose' in data and not isinstance(data['pose'], Pose):
            data['pose'] = Pose.from_value(data['pose'])
        if 'direction' in data and not isinstance(data['direction'], FaceDirection):
            data['direction'] = FaceDirection.from_value(data['direction'])

        # Call parent's from_dict, which handles 'points' and 'rect' conversion
        # This will also set the 'features', 'classification', 'confidence', 'track_id'
        instance = super().from_dict(data)

        # Handle tuple fields that might have been stored as JSON strings
        # This assumes data['direction_vector'] and data['direction_origin']
        # would be lists from json.loads, which tuple() can convert.
        if isinstance(instance.direction_vector, list):
            instance.direction_vector = tuple(instance.direction_vector)
        if isinstance(instance.direction_origin, list):
            instance.direction_origin = tuple(instance.direction_origin)

        return instance