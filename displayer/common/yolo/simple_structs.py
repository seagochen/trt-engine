import json
from dataclasses import dataclass, field
from typing import List


@dataclass
class Yolo:
    lx: int
    ly: int
    rx: int
    ry: int
    cls: int
    conf: float

    def to_json(self):
        """Converts the Yolo object to a JSON string."""
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)

    @staticmethod
    def from_json(data: str):
        """Creates a Yolo object or a list of Yolo objects from a JSON string."""
        data_list = json.loads(data)
        if isinstance(data_list, list):
            return [Yolo(**item) for item in data_list]
        else:
            return Yolo(**data_list)


@dataclass
class YoloPoint:
    x: int
    y: int
    conf: float

    def to_json(self):
        """Converts the YoloPoint object to a JSON string."""
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)

    @staticmethod
    def from_json(data: str):
        """Creates a YoloPoint object from a JSON string."""
        data_dict = json.loads(data)
        return YoloPoint(**data_dict)


@dataclass
class YoloPose:
    lx: int
    ly: int
    rx: int
    ry: int
    conf: float
    pts: List[YoloPoint] = field(default_factory=list)

    def to_json(self):
        """Converts the YoloPose object to a JSON string."""
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)

    @staticmethod
    def from_json(data: str):
        """Creates one or more YoloPose objects from a JSON string."""
        data_list = json.loads(data)

        # Helper function to process individual YoloPose objects
        def parse_yolo_pose(data_dict):
            pts_data = data_dict.pop("pts", [])
            pts = [YoloPoint(**pt) for pt in pts_data]
            return YoloPose(pts=pts, **data_dict)

        # Check if data_list is a list or a single dict
        if isinstance(data_list, list):
            return [parse_yolo_pose(item) for item in data_list]
        else:
            return parse_yolo_pose(data_list)
