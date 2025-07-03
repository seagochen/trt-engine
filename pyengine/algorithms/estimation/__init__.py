from typing import Tuple

import numpy as np

from pyengine.inference.unified_structs.auxiliary_structs import Pose, FaceDirection
from pyengine.inference.unified_structs.inference_results import Skeleton, Point, Rect


def is_valid_point(pt: Point) -> bool:
    """判断关键点是否有效：置信度 > 0.2 且坐标均大于 0"""
    return pt.confidence > 0.2 and pt.x > 0 and pt.y > 0


def is_valid_bbox(bbox: Rect) -> bool:
    """判断检测框是否有效：坐标均大于 0"""
    return bbox.x1 > 0 and bbox.y1 > 0 and bbox.x2 > 0 and bbox.y2 > 0


def compute_modulus(skeleton: Skeleton, divisor: float = 3.0) -> float:
    """
    根据检测框宽度计算模长，若检测框有效则返回宽度的 1/3，
    """
    bbox_rect = skeleton.rect

    if is_valid_bbox(bbox_rect):
        bbox_width = abs(bbox_rect.x1 - bbox_rect.x2)
        return bbox_width / divisor

    return 0.0


def analyze_front_side_back_face(nose: Point,
                                 left_eye: Point,
                                 right_eye: Point,
                                 left_ear: Point,
                                 right_ear: Point,
                                 valid_left_ear: bool,
                                 valid_right_ear: bool) -> Tuple[FaceDirection, float, Tuple[float, float], Tuple[float, float]]: # (修改) 增加 float 返回类型用于角度
    """Analyzes facial direction when nose and both eyes are valid."""
    mid_eye_x = (left_eye.x + right_eye.x) / 2.0
    mid_eye_y = (left_eye.y + right_eye.y) / 2.0
    face_vec = (nose.x - mid_eye_x, nose.y - mid_eye_y)

    angle_rad = np.arctan2(face_vec[1], face_vec[0])
    angle_deg = np.degrees(angle_rad)
    angle_deg_norm = (angle_deg + 360) % 360

    front_threshold = 5

    if (90 - front_threshold) <= angle_deg_norm <= (90 + front_threshold):
        orientation = FaceDirection.Front
    elif angle_deg_norm < 90 - front_threshold:
        orientation = FaceDirection.Left
    else:
        orientation = FaceDirection.Right

    vec_x, vec_y = 0.0, 0.0
    if valid_left_ear and valid_right_ear:
        ear_mid_x = (left_ear.x + right_ear.x) / 2.0
        ear_mid_y = (left_ear.y + right_ear.y) / 2.0
        adj_vec = (nose.x - ear_mid_x, nose.y - ear_mid_y)
        norm = np.sqrt(adj_vec[0] ** 2 + adj_vec[1] ** 2)
        if norm != 0:
            vec_x, vec_y = adj_vec[0] / norm, adj_vec[1] / norm
    else:
        norm_face_vec = np.sqrt(face_vec[0] ** 2 + face_vec[1] ** 2)
        if norm_face_vec != 0:
            vec_x, vec_y = face_vec[0] / norm_face_vec, face_vec[1] / norm_face_vec
        else:
            vec_x, vec_y = 0.0, 0.0

    origin_x, origin_y = nose.x, nose.y
    # (修改) 返回计算出的角度 angle_deg_norm
    return orientation, angle_deg_norm, (vec_x, vec_y), (origin_x, origin_y)


def analyze_single_eye_face(nose: Point,
                            left_eye: Point,
                            right_eye: Point) -> Tuple[FaceDirection, float, Tuple[float, float], Tuple[float, float]]: # (修改) 增加 float 返回类型用于角度
    """Analyzes facial direction when nose and only one eye are valid."""
    origin_x, origin_y = nose.x, nose.y
    angle = 0.0 # (修改) 为这种情况设置默认角度

    if is_valid_point(left_eye) and not is_valid_point(right_eye):
        orientation = FaceDirection.Left
        vec_x, vec_y = 1.0, 0.0
    elif is_valid_point(right_eye) and not is_valid_point(left_eye):
        orientation = FaceDirection.Right
        vec_x, vec_y = -1.0, 0.0
    else:
        orientation = FaceDirection.Unknown
        vec_x, vec_y = 0.0, 0.0

    # (修改) 返回默认角度
    return orientation, angle, (vec_x, vec_y), (origin_x, origin_y)


def analyze_back_face_ears_only(left_ear: Point,
                                right_ear: Point) -> Tuple[FaceDirection, float, Tuple[float, float], Tuple[float, float]]: # (修改) 增加 float 返回类型用于角度
    """Analyzes facial direction when only both ears are valid (implies back of head)."""
    orientation = FaceDirection.Back
    ear_mid_x = (left_ear.x + right_ear.x) / 2.0
    ear_mid_y = (left_ear.y + right_ear.y) / 2.0
    origin_x, origin_y = ear_mid_x, ear_mid_y
    vec_x, vec_y = 0.0, -1.0
    angle = 0.0 # (修改) 为这种情况设置默认角度

    # (修改) 返回默认角度
    return orientation, angle, (vec_x, vec_y), (origin_x, origin_y)
