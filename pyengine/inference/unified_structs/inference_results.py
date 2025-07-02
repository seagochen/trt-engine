import json
from dataclasses import dataclass, field, asdict, astuple
from typing import List, Union, Any, Type, TypeVar, Tuple

# 为泛型类方法定义 TypeVar
T = TypeVar('T', bound='InferenceResult')


@dataclass
class InferenceResult:

    """基类，提供通用的序列化和反序列化方法。"""

    def to_list(self) -> Tuple[Any, ...]:
        """将数据类实例转换为其字段值的元组。"""
        return astuple(self)

    def to_dict(self: T) -> dict[str, Any]: # 添加了 self 的类型提示
        """将数据类实例转换为字段名和值的字典。"""
        return asdict(self)

    def to_json(self, indent: int = 4) -> str:
        """将数据类实例转换为 JSON 字符串。"""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_list(cls: Type[T], data: List[Any]) -> T:
        """从值列表创建数据类实例。
        注意：对于字段顺序与列表顺序一致的简单数据类，此方法效果最佳。
        """
        # 类型检查器可能会对 *data 警告，但在运行时，只要 data 与字段匹配就是正确的。
        return cls(*data)

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        """从字典创建数据类实例。
        注意：对于字典键与字段名一致的简单数据类，此方法效果最佳。
        """
        # 类型检查器可能会对 **data 警告，但在运行时，只要 data 与字段匹配就是正确的。
        return cls(**data)

    @classmethod
    def from_json(cls: Type[T], data: str) -> Union[T, List[T]]:
        """从 JSON 字符串创建一个或多个数据类实例。"""
        try:
            data_parsed = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError("无效的 JSON 数据") from e

        if isinstance(data_parsed, list):
            # 确保列表中的每一项都是字典，以便 from_dict 使用
            if not all(isinstance(item, dict) for item in data_parsed):
                raise TypeError("JSON 列表必须只包含字典")
            return [cls.from_dict(item) for item in data_parsed]
        elif isinstance(data_parsed, dict):
            return cls.from_dict(data_parsed)
        else:
            raise TypeError("JSON 必须表示一个字典或字典列表")


@dataclass
class Rect:
    """Represents a rectangle with its top-left and bottom-right coordinates."""
    x1: float = 0.0 # 左上角 x 坐标
    y1: float = 0.0 # 左上角 y 坐标
    x2: float = 0.0 # 右下角 x 坐标
    y2: float = 0.0 # 右下角 y 坐标


@dataclass
class ObjectDetection(InferenceResult):
    """Represents detection bounding box with class and confidence."""
    rect: Rect = field(default_factory=Rect)  # 包含检测框的矩形
    classification: int = 0
    confidence: float = 0.0

    # ----------------- 用于追踪用的特殊字段，平时不使用 -----------------
    track_id: int = 0
    features: List[float] = field(default_factory=list)  # 特征向量列表

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectDetection":
        """
        从字典创建实例，并手动将嵌套的 'rect' 字典转换为 Rect 对象。
        """
        # 基类的 from_dict 不知道 'rect' 应该是一个 Rect 对象。
        # 我们必须手动将 'rect' 字典转换为一个 Rect 对象。
        if 'rect' in data and isinstance(data.get('rect'), dict):
            data['rect'] = Rect(**data['rect'])

        # 调用父类的 from_dict，让它处理通用逻辑。
        # 这里 super().from_dict 实际上是 InferenceResult.from_dict
        return super().from_dict(data)

@dataclass
class Point(InferenceResult):
    """Represents a single keypoint with its coordinates and confidence."""
    x: float = 0
    y: float = 0
    confidence: float = 0.0


@dataclass
class Skeleton(ObjectDetection):
    """Represents a human skeleton, inheriting bounding box info and adding keypoints."""
    points: List[Point] = field(default_factory=list) # 一个Point对象的列表

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Skeleton":
        """
        从字典创建实例，并手动转换所有嵌套对象（rect 和 points）。
        """
        # 这个类有自己的嵌套对象：一个 Point 列表。
        # 我们必须将列表中的每个字典都转换为 Point 对象。
        if 'points' in data and isinstance(data.get('points'), list):
            data['points'] = [Point(**p) for p in data['points']]

        # 它还从 ObjectDetection 继承了 'rect'。我们也必须在这里处理它。
        if 'rect' in data and isinstance(data.get('rect'), dict):
            data['rect'] = Rect(**data['rect'])

        # 调用父类 ObjectDetection 的 from_dict，让它去处理 'rect' 字段。
        # 这样就无需在此重复 'rect' 的转换逻辑。
        return super().from_dict(data)