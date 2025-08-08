# posture_simple_calculation.py (New Logic Framework)

from enum import Enum
from typing import Tuple

from pyengine.algorithms.estimation import is_valid_point, compute_modulus, analyze_front_side_back_face
from pyengine.inference.unified_structs.auxiliary_structs import BodyOrientation, FaceDirection, ExpandedSkeleton, Pose
from pyengine.inference.unified_structs.inference_results import Skeleton, Point


def get_shoulder_orientation(
    left_shoulder: Point, right_shoulder: Point,
    valid_left_shoulder: bool, valid_right_shoulder: bool
) -> BodyOrientation:
    """
    判断身体姿态是“正面”还是“侧面”。
    这是所有判断的第一个总开关。
    """
    if not (valid_left_shoulder and valid_right_shoulder):
        return BodyOrientation.Unknown

    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    shoulder_height_diff = abs(left_shoulder.y - right_shoulder.y)

    if shoulder_height_diff < 1e-5:
        return BodyOrientation.Frontal

    ratio = shoulder_width / shoulder_height_diff
    if ratio > 2.0:
        return BodyOrientation.Frontal
    else:
        return BodyOrientation.Side


def calculate_direction_and_posture(skeleton: Skeleton, default_confidence:float = 0.3) -> ExpandedSkeleton:
    """
    [全新逻辑]
    严格按照分层规则判断朝向：
    1. 判断身体姿态（正面/侧面）。
    2. 在身体姿态的上下文中，解读面部特征。
    3. 处理未知情况。
    """
    modulus = compute_modulus(skeleton, divisor=3.0)
    
    # --- 获取所有需要的关键点 ---
    nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder = skeleton.points[:7]

    # --- 检查所有关键点的有效性 ---
    valid_nose = is_valid_point(nose, default_confidence)
    valid_left_eye = is_valid_point(left_eye, default_confidence)
    valid_right_eye = is_valid_point(right_eye, default_confidence)
    valid_left_ear = is_valid_point(left_ear, default_confidence)
    valid_right_ear = is_valid_point(right_ear, default_confidence)
    valid_left_shoulder = is_valid_point(left_shoulder, default_confidence)
    valid_right_shoulder = is_valid_point(right_shoulder, default_confidence)

    # --- 初始化返回变量 ---
    orientation = FaceDirection.Unknown
    angle, vec_x, vec_y, origin_x, origin_y = 0.0, 0.0, 0.0, 0.0, 0.0

    # --- 开始执行分层判断逻辑 ---
    
    # 首先，获取身体的整体姿态（正面/侧面）
    body_pose = get_shoulder_orientation(left_shoulder, right_shoulder, valid_left_shoulder, valid_right_shoulder)

    # --- 规则1: 身体姿态为“正面” ---
    if body_pose == BodyOrientation.Frontal:
        # 规则1.1: 身体朝前 (left_shoulder.x > right_shoulder.x)
        if left_shoulder.x > right_shoulder.x:
            # 只有当鼻子和双眼都可见时，才能计算精确的面部法向量和角度
            if valid_nose and valid_left_eye and valid_right_eye:
                orientation, angle, (vec_x, vec_y), (origin_x, origin_y) = \
                    analyze_front_side_back_face(nose, left_eye, right_eye, left_ear, right_ear, valid_left_ear, valid_right_ear)
            else:
                # 如果面部特征不全，则默认身体和面部朝向一致，为正面
                orientation = FaceDirection.Front
        
        # 规则1.2: 身体朝后 (left_shoulder.x < right_shoulder.x)
        elif left_shoulder.x < right_shoulder.x:
            # 检查是否有“回头”迹象
            if valid_left_eye:       # 如果看到左眼，说明人向自己的右肩回头，面朝屏幕右侧
                orientation = FaceDirection.Right
            elif valid_right_eye:    # 如果看到右眼，说明人向自己的左肩回头，面朝屏幕左侧
                orientation = FaceDirection.Left
            else:                    # 否则，认为头和身体方向一致，为背面
                orientation = FaceDirection.Back

    # --- 规则2: 身体姿态为“侧面” ---
    elif body_pose == BodyOrientation.Side:
        # 检查左右两侧是否有可见的面部特征（眼或耳）
        left_features_visible = valid_left_eye or valid_left_ear
        right_features_visible = valid_right_eye or valid_right_ear

        # 规则2.3: 如果两侧特征都可见，说明虽然身体是侧的，但脸正对我们
        if left_features_visible and right_features_visible:
            orientation = FaceDirection.Front
        # 规则2.1: 如果只看到右侧特征，认为面朝屏幕左侧
        elif right_features_visible:
            orientation = FaceDirection.Left
        # 规则2.2: 如果只看到左侧特征，认为面朝屏幕右侧
        elif left_features_visible:
            orientation = FaceDirection.Right
    
    # --- 规则3: 不属于以上所有情况，判断为未知 ---
    # (此处的 orientation 默认为 FaceDirection.Unknown)

    # --- 创建并返回最终结果 ---
    extended_skeleton = ExpandedSkeleton(
        classification=skeleton.classification,
        confidence=skeleton.confidence,
        track_id=skeleton.track_id,
        features=skeleton.features,
        rect=skeleton.rect,
        points=skeleton.points,
        posture_type=calculate_bbox_aspect_ratio(skeleton),
        direction_type=orientation,
        direction_angle=angle,
        direction_modulus=modulus,
        direction_vector=(vec_x, vec_y),
        direction_origin=(origin_x, origin_y)
    )
    return extended_skeleton


def calculate_bbox_aspect_ratio(skeleton: Skeleton, default_confidence=0.5) -> Pose:
    """使用检测框的宽高比来简单判断站立或蹲/坐。"""
    bbox_width = abs(skeleton.rect.x1 - skeleton.rect.x2)
    bbox_height = abs(skeleton.rect.y1 - skeleton.rect.y2)

    if bbox_width == 0 or bbox_height == 0:
        return Pose.Unknown

    if len(skeleton.points) < 17:
        return Pose.Unknown

    left_ankle = skeleton.points[15]
    right_ankle = skeleton.points[16]
    if not is_valid_point(left_ankle, default_confidence) or \
            not is_valid_point(right_ankle, default_confidence):
        return Pose.Unknown

    aspect_ratio = bbox_height / bbox_width
    standing_threshold = 3.0

    if aspect_ratio >= standing_threshold:
        return Pose.Standing
    else:
        return Pose.Squatting