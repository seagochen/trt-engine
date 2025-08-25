import ctypes
from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np


class C_KeyPoint(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("conf", ctypes.c_float),
    ]


class C_YoloPose(ctypes.Structure):
    _fields_ = [
        ("lx", ctypes.c_int),
        ("ly", ctypes.c_int),
        ("rx", ctypes.c_int),
        ("ry", ctypes.c_int),
        ("cls", ctypes.c_int),
        ("num_pts", ctypes.c_int),
        ("conf", ctypes.c_float),
        ("pts", ctypes.POINTER(C_KeyPoint)),  # 指向C_KeyPoint数组的指针
        ("feats", ctypes.POINTER(ctypes.c_float)),  # 指向float数组(长度256)的指针
    ]


class C_ImagePoseResults(ctypes.Structure):
    _fields_ = [
        ("image_idx", ctypes.c_int),
        ("num_detections", ctypes.c_int),
        ("detections", ctypes.POINTER(C_YoloPose)),  # 指向C_YoloPose数组的指针
    ]


class C_BatchedPoseResults(ctypes.Structure):
    _fields_ = [
        ("num_images", ctypes.c_int),
        ("results", ctypes.POINTER(C_ImagePoseResults)),  # 指向C_ImagePoseResults数组的指针
    ]


class DetectorBase(ABC):
    """Abstract interface so the rest of your code can be backend-agnostic."""

    @abstractmethod
    def register(self) -> None:
        """Register models/resources (once per process)."""
        ...

    @abstractmethod
    def create(self) -> None:
        """Create/init native context(s)."""
        ...

    @abstractmethod
    def infer(self, images: List[np.ndarray]) -> List[Dict]:
        """Run batched inference and return unified results."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Destroy contexts and free resources."""
        ...
