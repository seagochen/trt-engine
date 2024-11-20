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

    def __init__(self, lx: int = 0, ly: int = 0, rx: int = 0, ry: int = 0, cls: int = 0, conf: float = 0.0):
        self.lx = lx
        self.ly = ly
        self.rx = rx
        self.ry = ry
        self.cls = cls
        self.conf = conf

    def to_list(self):
        """Converts the Yolo object to a list."""
        return [self.lx, self.ly, self.rx, self.ry, self.cls, self.conf]

    def from_list(self, data: List):
        """Creates a Yolo object from a list."""
        self.lx = data[0]
        self.ly = data[1]
        self.rx = data[2]
        self.ry = data[3]
        self.cls = data[4]
        self.conf = data[5]

    def to_dict(self):
        """Converts the Yolo object to a dictionary."""
        return {
            "lx": self.lx,
            "ly": self.ly,
            "rx": self.rx,
            "ry": self.ry,
            "cls": self.cls,
            "conf": self.conf
        }

    def from_dict(self, data: dict):
        """Creates a Yolo object from a dictionary."""
        self.lx = data["lx"]
        self.ly = data["ly"]
        self.rx = data["rx"]
        self.ry = data["ry"]
        self.cls = data["cls"]
        self.conf = data["conf"]

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

    def __init__(self, x: int = 0, y: int = 0, conf: float = 0.0):
        self.x = x
        self.y = y
        self.conf = conf

    def to_list(self):
        """Converts the YoloPoint object to a list."""
        return [self.x, self.y, self.conf]

    def from_list(self, data: List):
        """Creates a YoloPoint object from a list."""
        self.x = data[0]
        self.y = data[1]
        self.conf = data[2]

    def to_dict(self):
        """Converts the YoloPoint object to a dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "conf": self.conf
        }

    def from_dict(self, data: dict):
        """Creates a YoloPoint object from a dictionary."""
        self.x = data["x"]
        self.y = data["y"]
        self.conf = data["conf"]

    def to_json(self):
        """Converts the YoloPoint object to a JSON string."""
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)

    @staticmethod
    def from_json(data: str):
        """Creates a YoloPoint object from a JSON string."""
        data_dict = json.loads(data)
        return YoloPoint(**data_dict)


@dataclass
class YoloPose(Yolo):
    pts: List[YoloPoint] = field(default_factory=list)

    def __init__(self, lx: int = 0, ly: int = 0, rx: int = 0, ry: int = 0, cls: int = 0, conf: float = 0.0, pts: List[YoloPoint] = None):
        self.lx = lx
        self.ly = ly
        self.rx = rx
        self.ry = ry
        self.cls = cls
        self.conf = conf

        # Ensure self.pts is always initialized
        self.pts = pts if pts is not None else [YoloPoint() for _ in range(17)]

    def to_list(self):
        """Converts the YoloPose object to a list."""
        return [self.lx, self.ly, self.rx, self.ry, self.cls, self.conf, [pt.to_list() for pt in self.pts]]

    def from_list(self, data: List):
        """Creates a YoloPose object from a list."""
        self.lx = data[0]
        self.ly = data[1]
        self.rx = data[2]
        self.ry = data[3]
        self.cls = data[4]
        self.conf = data[5]

        # Create a list to store the keypoints
        self.pts = [YoloPoint(*pt) for pt in data[6]]

    def to_dict(self):
        """Converts the YoloPose object to a dictionary."""
        return {
            "lx": self.lx,
            "ly": self.ly,
            "rx": self.rx,
            "ry": self.ry,
            "cls": self.cls,
            "conf": self.conf,
            "pts": [pt.to_dict() for pt in self.pts]
        }

    def from_dict(self, data: dict):
        """Creates a YoloPose object from a dictionary."""
        self.lx = data["lx"]
        self.ly = data["ly"]
        self.rx = data["rx"]
        self.ry = data["ry"]
        self.cls = data["cls"]
        self.conf = data["conf"]

        # Create a list to store the keypoints
        self.pts = [YoloPoint(**pt) for pt in data["pts"]]

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
