# %%
# posture_simple_calculation.py (New Logic Framework)

from enum import Enum
from typing import Tuple

from pyengine.algorithms.estimation import is_valid_point, compute_modulus, analyze_front_side_back_face
from pyengine.inference.unified_structs.auxiliary_structs import BodyOrientation, FaceDirection, ExpandedSkeleton, Pose
from pyengine.inference.unified_structs.inference_results import Skeleton, Point

# %% [markdown]
# ## 计算肩部朝向

# %%
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

# %% [markdown]
# ## 计算 `bbox` 的长宽比
# 
# 这种方法是一种比较粗浅的方法，可以用来评估目标人物是站立还是下蹲。
# 
# 其中 `default_ratio_threshold` 在 1080P的相机情况下使用3.0比较合适，在4K的画质中，可能要改为2.0。

# %%
def calculate_bbox_aspect_ratio(skeleton: Skeleton,
                                default_confidence: float=0.5,
                                default_ratio_threshold: float = 3.0) -> Pose:
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
    # print(aspect_ratio)

    if aspect_ratio >= default_ratio_threshold:
        return Pose.Standing
    else:
        return Pose.Squatting


# %% [markdown]
# ## 多特征融合的姿态判断
#
# 这个方法结合多个特征来判断站立/蹲下,提高鲁棒性:
# 1. bbox 长宽比
# 2. 肩膀到脚踝的垂直距离
# 3. 膝盖的相对位置
# 4. 髋关节的相对位置

# %%
def calculate_posture_robust(
    skeleton: Skeleton,
    confidence_threshold: float = 0.5,
    bbox_ratio_threshold: float = 1.6,
    bbox_ratio_weight: float = 0.3,
    vertical_span_weight: float = 0.4,
    knee_position_weight: float = 0.2,
    hip_position_weight: float = 0.1
) -> Tuple[Pose, float]:
    """
    使用多特征融合的姿态判断方法

    Args:
        skeleton: 骨架数据
        confidence_threshold: 关键点置信度阈值
        bbox_ratio_threshold: bbox长宽比阈值(站立/蹲下分界线)
        bbox_ratio_weight: bbox长宽比特征权重
        vertical_span_weight: 垂直跨度特征权重
        knee_position_weight: 膝盖位置特征权重
        hip_position_weight: 髋关节位置特征权重

    Returns:
        (姿态, 置信度): Pose枚举值和对应的置信度分数(0-1)
    """
    points = skeleton.points

    # 确保有足够的关键点
    if len(points) < 17:
        return Pose.Unknown, 0.0

    bbox_w = abs(skeleton.rect.x2 - skeleton.rect.x1)
    bbox_h = abs(skeleton.rect.y2 - skeleton.rect.y1)

    if bbox_w == 0 or bbox_h == 0:
        return Pose.Unknown, 0.0

    # 特征1: bbox 长宽比
    aspect_ratio = bbox_h / bbox_w

    # 特征2: 关键点垂直分布 (肩膀到脚踝的距离 vs bbox高度)
    shoulder_y = None
    ankle_y = None

    # 获取肩膀中点
    left_shoulder, right_shoulder = points[5], points[6]
    if is_valid_point(left_shoulder, confidence_threshold) and \
       is_valid_point(right_shoulder, confidence_threshold):
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

    # 获取脚踝位置 (取可见的那个)
    left_ankle, right_ankle = points[15], points[16]
    if is_valid_point(left_ankle, confidence_threshold):
        ankle_y = left_ankle.y
    elif is_valid_point(right_ankle, confidence_threshold):
        ankle_y = right_ankle.y

    # 特征3: 膝盖位置 (如果可见)
    # 蹲下时膝盖会上移,站立时膝盖在中间偏下
    knee_ratio = None
    left_knee, right_knee = points[13], points[14]
    if shoulder_y is not None:
        for knee in [left_knee, right_knee]:
            if is_valid_point(knee, confidence_threshold):
                # 膝盖在 肩膀-bbox底部 的相对位置
                knee_ratio = (knee.y - shoulder_y) / (bbox_h * 0.8) if bbox_h > 0 else None
                break

    # 特征4: 髋关节位置
    hip_ratio = None
    left_hip, right_hip = points[11], points[12]
    if shoulder_y is not None:
        for hip in [left_hip, right_hip]:
            if is_valid_point(hip, confidence_threshold):
                hip_ratio = (hip.y - shoulder_y) / bbox_h if bbox_h > 0 else None
                break

    # ===== 决策逻辑 =====
    standing_score = 0.0
    squatting_score = 0.0
    total_weight = 0.0

    # 权重1: bbox 长宽比
    if aspect_ratio > 0:
        weight = bbox_ratio_weight
        if aspect_ratio >= bbox_ratio_threshold:
            standing_score += weight
        else:
            squatting_score += weight
        total_weight += weight

    # 权重2: 肩膀到脚踝的距离占比
    if shoulder_y is not None and ankle_y is not None:
        weight = vertical_span_weight
        vertical_span = ankle_y - shoulder_y
        vertical_ratio = vertical_span / bbox_h if bbox_h > 0 else 0

        # 站立时这个比例应该较大 (>0.6)
        # 蹲下时较小 (<0.5)
        if vertical_ratio > 0.6:
            standing_score += weight
        elif vertical_ratio < 0.5:
            squatting_score += weight
        else:
            # 中间区域按比例分配
            standing_score += weight * (vertical_ratio - 0.5) / 0.1
            squatting_score += weight * (0.6 - vertical_ratio) / 0.1
        total_weight += weight

    # 权重3: 膝盖位置
    if knee_ratio is not None:
        weight = knee_position_weight
        # 站立时膝盖在中下部 (0.4-0.6)
        # 蹲下时膝盖上移 (<0.3)
        if knee_ratio < 0.3:
            squatting_score += weight
        elif knee_ratio > 0.4:
            standing_score += weight
        total_weight += weight

    # 权重4: 髋关节位置
    if hip_ratio is not None:
        weight = hip_position_weight
        # 站立时髋在中部 (0.3-0.5)
        # 蹲下时髋会下移或位置不明显
        if 0.3 < hip_ratio < 0.5:
            standing_score += weight
        elif hip_ratio > 0.6 or hip_ratio < 0.2:
            squatting_score += weight
        total_weight += weight

    # 归一化并做决策
    if total_weight > 0:
        standing_score /= total_weight
        squatting_score /= total_weight

        if standing_score > squatting_score:
            return Pose.Standing, standing_score
        else:
            return Pose.Squatting, squatting_score

    return Pose.Unknown, 0.0

# %% [markdown]
# ## 计算脸部和姿态
# 
# 外部主要调用 `calculate_direction_and_posture`

# %%
def calculate_direction_and_posture(
    skeleton: Skeleton,
    default_confidence: float = 0.3,
    default_ratio_threshold: float = 3.0,
    use_robust_posture: bool = True,
    posture_config: dict = None
) -> ExpandedSkeleton:
    """
    [全新逻辑]
    严格按照分层规则判断朝向：
    1. 判断身体姿态(正面/侧面)。
    2. 在身体姿态的上下文中，解读面部特征。
    3. 处理未知情况。

    Args:
        skeleton: 骨架数据
        default_confidence: 关键点置信度阈值
        default_ratio_threshold: 传统方法的bbox长宽比阈值
        use_robust_posture: 是否使用多特征融合的姿态识别方法
        posture_config: 姿态识别配置字典,包含:
            - confidence_threshold: 关键点置信度阈值
            - bbox_ratio_threshold: bbox长宽比阈值
            - bbox_ratio_weight: bbox权重
            - vertical_span_weight: 垂直跨度权重
            - knee_position_weight: 膝盖位置权重
            - hip_position_weight: 髋关节位置权重
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

    # 首先，获取身体的整体姿态(正面/侧面)
    body_pose = get_shoulder_orientation(left_shoulder, right_shoulder, valid_left_shoulder, valid_right_shoulder)

    # --- 规则1: 身体姿态为"正面" ---
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
            # 检查是否有"回头"迹象
            if valid_left_eye:       # 如果看到左眼，说明人向自己的右肩回头，面朝屏幕右侧
                orientation = FaceDirection.Right
            elif valid_right_eye:    # 如果看到右眼，说明人向自己的左肩回头，面朝屏幕左侧
                orientation = FaceDirection.Left
            else:                    # 否则，认为头和身体方向一致，为背面
                orientation = FaceDirection.Back

    # --- 规则2: 身体姿态为"侧面" ---
    elif body_pose == BodyOrientation.Side:
        # 检查左右两侧是否有可见的面部特征(眼或耳)
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

    # --- 姿态识别：根据配置选择方法 ---
    if use_robust_posture and posture_config:
        # 使用多特征融合方法
        posture, confidence = calculate_posture_robust(
            skeleton,
            confidence_threshold=posture_config.get('confidence_threshold', 0.5),
            bbox_ratio_threshold=posture_config.get('bbox_ratio_threshold', 1.6),
            bbox_ratio_weight=posture_config.get('bbox_ratio_weight', 0.3),
            vertical_span_weight=posture_config.get('vertical_span_weight', 0.4),
            knee_position_weight=posture_config.get('knee_position_weight', 0.2),
            hip_position_weight=posture_config.get('hip_position_weight', 0.1)
        )
    else:
        # 使用传统的bbox长宽比方法
        posture = calculate_bbox_aspect_ratio(skeleton, default_ratio_threshold=default_ratio_threshold)

    # --- 创建并返回最终结果 ---
    extended_skeleton = ExpandedSkeleton(
        classification=skeleton.classification,
        confidence=skeleton.confidence,
        track_id=skeleton.track_id,
        features=skeleton.features,
        rect=skeleton.rect,
        points=skeleton.points,
        posture_type=posture,
        direction_type=orientation,
        direction_angle=angle,
        direction_modulus=modulus,
        direction_vector=(vec_x, vec_y),
        direction_origin=(origin_x, origin_y)
    )
    return extended_skeleton


