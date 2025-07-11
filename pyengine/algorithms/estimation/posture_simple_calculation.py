from pyengine.algorithms.estimation import is_valid_point, compute_modulus, analyze_front_side_back_face, \
    analyze_single_eye_face, analyze_back_face_ears_only
from pyengine.inference.unified_structs.auxiliary_structs import FaceDirection, ExpandedSkeleton, Pose
from pyengine.inference.unified_structs.inference_results import Skeleton


def calculate_bbox_aspect_ratio(skeleton: Skeleton) -> Pose:
    """
    根据检测框的长宽比判断人体姿态。
    新的编码规则建议:

    # TODO: Replace with more accurate model-based posture detection

        0 - 未知 (Unknown)
        1 - 弯腰 (Bending) - （可选）如果仍需判断弯腰，则需要关键点。
                              如果仅用长宽比，弯腰可能被判断为站立或蹲/坐。
                              这里暂时将其简化，不单独判断弯腰，除非有明确的区分逻辑。
        2 - 坐/下蹲 (Sitting/Squatting)
        3 - 站立 (Standing)
    """
    bbox_width = abs(skeleton.rect.x1 - skeleton.rect.x2)
    bbox_height = abs(skeleton.rect.y1 - skeleton.rect.y2)

    if bbox_width == 0 or bbox_height == 0:
        return Pose.Unknown  # 未知，无效检测框

    # 检查关键点列表长度是否足够
    if len(skeleton.points) < 17:  # COCO model has 17 keypoints (0-16)
        # Not enough keypoints to determine ankles, return unknown or a default
        return Pose.Unknown

    # COCO 17 keypoints:
    # 15: Left Ankle
    # 16: Right Ankle
    left_ankle = skeleton.points[15]
    right_ankle = skeleton.points[16]
    if not is_valid_point(left_ankle) or not is_valid_point(right_ankle):
        return Pose.Unknown

    # 计算bbox的长宽比，这种方式能比单纯的基于关键点的坐标更稳定。
    aspect_ratio = bbox_height / bbox_width

    # 设定站立的阈值，假设站立时长宽比大于等于3.0，这个是成年人站立时的常见比例。
    standing_threshold = 3.0

    # 根据长宽比判断姿态
    if aspect_ratio >= standing_threshold:
        return Pose.Standing
    else:
        return Pose.Squatting  # 坐/下蹲


def calculate_direction_and_posture(skeleton: Skeleton) -> ExpandedSkeleton:
    """
    根据 pose_extend 中的面部关键点信息分析面部朝向，返回 FacialDirection 对象。
    """
    modulus = compute_modulus(skeleton, divisor=3.0)
    nose, right_eye, left_eye, right_ear, left_ear = skeleton.points[:5]

    valid_nose = is_valid_point(nose)
    valid_right_eye = is_valid_point(right_eye)
    valid_left_eye = is_valid_point(left_eye)
    valid_right_ear = is_valid_point(right_ear)
    valid_left_ear = is_valid_point(left_ear)

    orientation = FaceDirection.Unknown
    # (修改) 初始化角度变量
    angle = 0.0
    vec_x, vec_y = 0.0, 0.0
    origin_x, origin_y = 0.0, 0.0

    if valid_nose and valid_left_eye and valid_right_eye:
        # (修改) 接收返回的角度值
        orientation, angle, (vec_x, vec_y), (origin_x, origin_y) = \
            analyze_front_side_back_face(nose,
                                         left_eye,
                                         right_eye,
                                         left_ear,
                                         right_ear,
                                         valid_left_ear,
                                         valid_right_ear)

    elif valid_nose and (valid_left_eye or valid_right_eye):
        # (修改) 接收返回的角度值
        orientation, angle, (vec_x, vec_y), (origin_x, origin_y) = \
            analyze_single_eye_face(nose, left_eye, right_eye)

    elif valid_left_ear and valid_right_ear and not (valid_nose or valid_left_eye or valid_right_eye):
        # (修改) 接收返回的角度值
        orientation, angle, (vec_x, vec_y), (origin_x, origin_y) = \
            analyze_back_face_ears_only(left_ear, right_ear)

    # 开始将Skeleton升级为ExtendedSkeleton
    extended_skeleton = ExpandedSkeleton(
        # 复制 Skeleton 的所有字段
        classification=skeleton.classification,
        confidence=skeleton.confidence,
        track_id=skeleton.track_id,
        features=skeleton.features,
        rect=skeleton.rect,
        points=skeleton.points,

        # 姿态检测, 不更新任何姿态相关字段
        posture_type=calculate_bbox_aspect_ratio(skeleton),

        # 面部朝向
        direction_type=orientation,
        direction_angle=angle,
        direction_modulus=modulus,
        direction_vector=(vec_x, vec_y),
        direction_origin=(origin_x, origin_y)
    )
    return extended_skeleton