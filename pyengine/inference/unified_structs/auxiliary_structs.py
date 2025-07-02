from dataclasses import dataclass
from enum import IntEnum

from pyengine.inference.unified_structs.inference_results import Skeleton


class Face(IntEnum):
    Front = 0
    Left = 1
    Right = 2
    Back = 3
    Unknown = 4

    def __str__(self): return self.name.lower()

    @classmethod
    def from_value(cls, value): return cls(value)


class Pose(IntEnum):
    Standing = 0
    Bending = 1
    Sitting = 2
    Unknown = 3

    def __str__(self): return self.name.lower()

    @classmethod
    def from_value(cls, value): return cls(value)


@dataclass
class ExtendedSkeleton(Skeleton):
    face: Face = Face.Unknown
    pose: Pose = Pose.Unknown

    def __post_init__(self):
        """
        在对象初始化后，确保 face 和 pose 是正确的枚举类型。
        这是一种非常好的实践，它将类型转换逻辑封装在类的内部。
        """
        super().__post_init__() if hasattr(super(), '__post_init__') else None

        if not isinstance(self.face, Face):
            self.face = Face.from_value(self.face)
        if not isinstance(self.pose, Pose):
            self.pose = Pose.from_value(self.pose)
