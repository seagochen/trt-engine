import json
from dataclasses import dataclass, field, asdict, astuple
from typing import List, Union, Any, Type, TypeVar, Tuple

# Define TypeVar for generic class methods
T = TypeVar('T', bound='YoloBase')


@dataclass
class YoloBase:
    """Base class, providing common serialization and deserialization methods."""

    def to_list(self) -> Tuple[Any, ...]:
        """Converts the dataclass instance to a tuple of its field values."""
        return astuple(self)

    def to_dict(self) -> dict[str, Any]:
        """Converts the dataclass instance to a dictionary of its field names and values."""
        return asdict(self)

    def to_json(self, indent: int = 4) -> str:
        """Converts the dataclass instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_list(cls: Type[T], data: List[Any]) -> T:
        """Creates a dataclass instance from a list of values.
        Note: This works best for simple dataclasses where list order matches field order.
        """
        # Type checkers might warn about *data, but at runtime, it's correct if data matches fields.
        return cls(*data)

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        """Creates a dataclass instance from a dictionary.
        Note: This works best for simple dataclasses where dict keys match field names.
        """
        # Type checkers might warn about **data, but at runtime, it's correct if data matches fields.
        return cls(**data)

    @classmethod
    def from_json(cls: Type[T], data: str) -> Union[T, List[T]]:
        """Creates one or more dataclass instances from a JSON string."""
        try:
            data_parsed = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON data") from e

        if isinstance(data_parsed, list):
            # Ensure each item in the list is a dictionary for from_dict
            if not all(isinstance(item, dict) for item in data_parsed):
                raise TypeError("JSON list must contain only dictionaries")
            return [cls.from_dict(item) for item in data_parsed]
        elif isinstance(data_parsed, dict):
            return cls.from_dict(data_parsed)
        else:
            raise TypeError("JSON must represent a dict or a list of dicts")


@dataclass
class Yolo(YoloBase):
    """Represents a YOLO detection bounding box with class and confidence."""
    lx: int = 0
    ly: int = 0
    rx: int = 0
    ry: int = 0
    cls: int = 0
    conf: float = 0.0


@dataclass
class YoloPoint(YoloBase):
    """Represents a single keypoint with its coordinates and confidence."""
    x: int = 0
    y: int = 0
    conf: float = 0.0


@dataclass
class YoloPose(Yolo):
    """Represents a YOLO pose detection, extending Yolo with keypoints."""
    # Using 'list' as type hint for pts for clarity and allowing mutable default
    # but still using default_factory for safety.
    pts: List[YoloPoint] = field(default_factory=lambda: [YoloPoint() for _ in range(17)])

    def to_list(self) -> Tuple[Any, ...]:
        """Converts YoloPose to a tuple, including nested YoloPoints as lists."""
        # Get base Yolo fields as tuple
        base_tuple = super().to_list()
        # Convert list of YoloPoint objects to list of YoloPoint's list representations
        pts_list_of_lists = [pt.to_list() for pt in self.pts]
        # Combine base fields with the list of keypoint lists
        return base_tuple + (pts_list_of_lists,)

    def to_dict(self) -> dict[str, Any]:
        """Converts YoloPose to a dictionary, including nested YoloPoints as dictionaries."""
        base_dict = super().to_dict()
        base_dict['pts'] = [pt.to_dict() for pt in self.pts]
        return base_dict

    @classmethod
    def from_list(cls, data: List[Any]) -> 'YoloPose':
        """Creates a YoloPose instance from a list of values.
        Expected format: [lx, ly, rx, ry, cls, conf, [[pt1.x, pt1.y, pt1.conf], ...]]
        """
        # YoloPose has 6 fields from Yolo + 1 field for 'pts' = 7 top-level elements
        if len(data) != 7:
            raise ValueError(f"List must contain exactly 7 elements for YoloPose, but got {len(data)}")

        # Unpack base Yolo data and the list of points data
        lx, ly, rx, ry, cls_val, conf, pts_data_list = data

        # Ensure pts_data_list is a list (of lists)
        if not isinstance(pts_data_list, list):
            raise TypeError("The last element in the list for YoloPose must be a list of keypoint lists.")

        # Reconstruct YoloPoint objects from their list representations
        pts = [YoloPoint.from_list(pt_item) for pt_item in pts_data_list]

        # Create YoloPose instance using keyword arguments for clarity and robustness
        return cls(lx=lx, ly=ly, rx=rx, ry=ry, cls=cls_val, conf=conf, pts=pts)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'YoloPose':
        """Creates a YoloPose instance from a dictionary, handling nested YoloPoint objects."""
        # Safely pop 'pts' data. It should be a list of dicts.
        pts_data = data.pop('pts', [])
        if not isinstance(pts_data, list) or not all(isinstance(item, dict) for item in pts_data):
            raise TypeError("The 'pts' field in the dictionary for YoloPose must be a list of keypoint dictionaries.")

        # Reconstruct YoloPoint objects from their dictionary representations
        pts = [YoloPoint.from_dict(pt_dict) for pt_dict in pts_data]

        # Create the YoloPose instance. **data will now only contain Yolo's base fields.
        return cls(**data, pts=pts)


@dataclass
class YoloSortedBase(YoloBase):
    """Abstract base class for sorted YOLO detections with an 'oid' field."""
    oid: int = 0

    def to_list(self) -> Tuple[Any, ...]:
        """Converts the sorted object to a list, including oid as the first element."""
        # This will be overridden by subclasses to include their specific data
        # For YoloSorted, it's (oid, lx, ly, rx, ry, cls, conf)
        # For YoloPoseSorted, it's (oid, lx, ly, rx, ry, cls, conf, [[pt1], ...])
        raise NotImplementedError("Subclasses must implement to_list specific to their structure.")

    @classmethod
    def from_list(cls: Type[T], data: List[Any]) -> T:
        """Creates a sorted object from a list, expecting oid as the first element.
        This method needs to be implemented by concrete subclasses to correctly unpack.
        """
        raise NotImplementedError("Subclasses must implement from_list specific to their structure.")

    def to_dict(self) -> dict[str, Any]:
        """Converts the sorted object to a dictionary, including oid."""
        base_dict = super().to_dict() # This will get direct fields from the child class (Yolo or YoloPose)
        # Remove 'oid' if it was included by asdict (from the direct field)
        # and re-add it explicitly as the first key.
        # This is a bit tricky with asdict/astuple as they get all fields including oid
        # Let's override to_dict directly in the concrete classes for cleaner control.
        raise NotImplementedError("Subclasses must implement to_dict specific to their structure.")

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        """Creates a sorted object from a dictionary, expecting 'oid'.
        This method needs to be implemented by concrete subclasses to correctly unpack.
        """
        raise NotImplementedError("Subclasses must implement from_dict specific to their structure.")


@dataclass
class YoloPoseSorted(YoloPose):
    """Extended YoloPose class, adding an 'oid' field for sorting/tracking."""
    oid: int = 0  # New field

    def to_list(self) -> Tuple[Any, ...]:
        """Converts YoloPoseSorted to a tuple, including oid."""
        # Get the YoloPose (base) tuple representation: (lx, ly, rx, ry, cls, conf, pts_list_of_lists)
        base_pose_tuple = super().to_list()
        # Combine oid with the base pose tuple
        return (self.oid,) + base_pose_tuple

    @classmethod
    def from_list(cls, data: List[Any]) -> 'YoloPoseSorted':
        """Creates a YoloPoseSorted object from a list.
        Expected format: [oid, lx, ly, rx, ry, cls, conf, [[pt1.x, pt1.y, pt1.conf], ...]]
        """
        # YoloPoseSorted has 1 (oid) + 6 (Yolo fields) + 1 (pts list) = 8 top-level elements
        if len(data) != 8:
            raise ValueError(f"List must contain exactly 8 elements for YoloPoseSorted, but got {len(data)}")

        # Unpack oid and the rest of the data for YoloPose
        oid_val, *pose_data = data
        # Use YoloPose's from_list to handle the rest of the data
        pose_instance = YoloPose.from_list(pose_data)
        # Create YoloPoseSorted instance by combining oid and the pose instance's dictionary representation
        return cls(oid=oid_val, **pose_instance.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Converts YoloPoseSorted to a dictionary, including oid."""
        base_dict = super().to_dict() # Gets all YoloPose fields including 'pts'
        base_dict['oid'] = self.oid # Add 'oid'
        return base_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'YoloPoseSorted':
        """Creates a YoloPoseSorted object from a dictionary."""
        oid_val = data.pop('oid', 0) # Safely pop 'oid'
        # Use YoloPose's from_dict to handle the rest of the data
        pose_instance = YoloPose.from_dict(data) # 'data' now only contains YoloPose fields
        # Create YoloPoseSorted instance by combining oid and the pose instance's dictionary representation
        return cls(oid=oid_val, **pose_instance.to_dict())


@dataclass
class YoloSorted(Yolo):
    """Extended Yolo class, adding an 'oid' field for sorting/tracking."""
    oid: int = 0  # New field

    def to_list(self) -> Tuple[Any, ...]:
        """Converts YoloSorted to a tuple, including oid."""
        # Get the Yolo (base) tuple representation: (lx, ly, rx, ry, cls, conf)
        base_yolo_tuple = super().to_list()
        # Combine oid with the base Yolo tuple
        return (self.oid,) + base_yolo_tuple

    @classmethod
    def from_list(cls, data: List[Any]) -> 'YoloSorted':
        """Creates a YoloSorted object from a list.
        Expected format: [oid, lx, ly, rx, ry, cls, conf]
        """
        # YoloSorted has 1 (oid) + 6 (Yolo fields) = 7 top-level elements
        if len(data) != 7:
            raise ValueError(f"List must contain exactly 7 elements for YoloSorted, but got {len(data)}")

        # Unpack oid and the rest of the data for Yolo
        oid_val, *yolo_data = data
        # Use Yolo's from_list to handle the rest of the data
        yolo_instance = Yolo.from_list(yolo_data)
        # Create YoloSorted instance by combining oid and the yolo instance's dictionary representation
        return cls(oid=oid_val, **yolo_instance.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Converts YoloSorted to a dictionary, including oid."""
        base_dict = super().to_dict() # Gets all Yolo fields
        base_dict['oid'] = self.oid # Add 'oid'
        return base_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'YoloSorted':
        """From a dictionary, creates a YoloSorted object."""
        oid_val = data.pop('oid', 0) # Safely pop 'oid'
        # Use Yolo's from_dict to handle the rest of the data
        yolo_instance = Yolo.from_dict(data) # 'data' now only contains Yolo fields
        # Create YoloSorted instance by combining oid and the yolo instance's dictionary representation
        return cls(oid=oid_val, **yolo_instance.to_dict())

@dataclass
class Posture:
    """
    人体姿态信息：
      action: 动作编码
          0 - 未知
          1 - 弯腰
          2 - 坐
          3 - 下蹲
          4 - 站立
    """
    action: int

@dataclass
class FacialDirection:
    """
    面部朝向信息：
      modulus: 模长，根据检测框宽度计算得到
      vector: 单位方向向量 (vec_x, vec_y)
      origin: 原点坐标 (origin_x, origin_y)
      direction_desc: 方向描述 ("Front", "Left", "Right", "Back", "Unknown")
      direction_type: 离散方向编码（0: 正面, 1: 左侧, 2: 右侧, 3: 背面, -1: 未知）
    """
    modulus: int
    vector: Tuple[float, float]
    origin: Tuple[int, int]
    direction_desc: str
    direction_type: int