import math
from typing import List, Union, Tuple

import numpy as np
from pyengine.inference.extend.yolo.data_struct import YoloPose, YoloPoseSorted, YoloPoint
from pyengine.inference.extend.yolo.basic import Posture, FacialDirection


# -------------- 核心分析类 --------------
class PoseInsight:
    """
    PoseInsight 类整合了人体姿态与面部朝向的分析。
    调用 analyze_poses() 方法，传入包含 YoloPose 或 YoloPoseSorted 对象的列表，
    返回每个 pose 的 Posture 和 FacialDirection 信息。
    """
    ORIENTATION_TEXTS = {
        0: "Front",
        1: "Left",
        2: "Right",
        3: "Back",
        -1: "Unknown"
    }

    @staticmethod
    def _angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        """计算二维向量 v1 和 v2 之间的夹角（单位：度）"""
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        norm1 = math.hypot(v1[0], v1[1])
        norm2 = math.hypot(v2[0], v2[1])
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_angle = max(min(dot / (norm1 * norm2), 1.0), -1.0)
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)

    @staticmethod
    def _analyze_body_pose(pose: Union[YoloPose, YoloPoseSorted]) -> Posture:
        """
        根据检测框的长宽比判断人体姿态。
        新的编码规则建议:
            0 - 未知 (Unknown)
            1 - 弯腰 (Bending) - （可选）如果仍需判断弯腰，则需要关键点。
                                  如果仅用长宽比，弯腰可能被判断为站立或蹲/坐。
                                  这里暂时将其简化，不单独判断弯腰，除非有明确的区分逻辑。
            2 - 坐/下蹲 (Sitting/Squatting)
            3 - 站立 (Standing)
        """
        bbox_width = abs(pose.rx - pose.lx)
        bbox_height = abs(pose.ry - pose.ly)

        if bbox_width == 0 or bbox_height == 0:
            return Posture(action=0)  # 未知，无效检测框

        # 检查关键点列表长度是否足够
        if len(pose.pts) < 17:  # COCO model has 17 keypoints (0-16)
            # Not enough keypoints to determine ankles, return unknown or a default
            return Posture(action=0)

        # COCO 17 keypoints:
        # 15: Left Ankle
        # 16: Right Ankle
        left_ankle = pose.pts[15]
        right_ankle = pose.pts[16]
        if not PoseInsight._is_valid_point(left_ankle) or not PoseInsight._is_valid_point(right_ankle):
            return Posture(action=0)

        aspect_ratio = bbox_height / bbox_width

        standing_threshold = 3.0

        if aspect_ratio >= standing_threshold:
            action_code = 3  # Standing
        else:
            action_code = 2  # Sitting/Squatting

        # TODO: Replace with more accurate model-based posture detection
        return Posture(action=action_code)

    @staticmethod
    def _is_valid_point(pt) -> bool:
        """判断关键点是否有效：置信度 > 0.2 且坐标均大于 0"""
        return pt.conf > 0.2 and pt.x > 0 and pt.y > 0

    @staticmethod
    def _compute_modulus(pose: Union[YoloPose, YoloPoseSorted]) -> int:
        """
        根据检测框宽度计算模长，若检测框有效则返回宽度的 1/3，
        否则返回默认值 10。
        """
        if pose.lx is not None and pose.ly is not None and pose.rx is not None and pose.rx > pose.lx:
            return int(abs(pose.lx - pose.rx) / 3.0)
        return 10

    @staticmethod
    def _get_keypoints(pts: List[YoloPoint]) -> Tuple[YoloPoint, YoloPoint, YoloPoint, YoloPoint, YoloPoint]:
        """Helper to safely get the first 5 keypoints."""
        # This assumes pts has at least 5 elements, which is checked in _analyze_facial_direction
        return pts[0], pts[1], pts[2], pts[3], pts[4]

    @staticmethod
    def _analyze_front_side_back_face(nose: YoloPoint, left_eye: YoloPoint, right_eye: YoloPoint,
                                      left_ear: YoloPoint, right_ear: YoloPoint,
                                      valid_left_ear: bool, valid_right_ear: bool) -> Tuple[
        int, Tuple[float, float], Tuple[int, int]]:
        """Analyzes facial direction when nose and both eyes are valid."""
        mid_eye_x = (left_eye.x + right_eye.x) / 2.0
        mid_eye_y = (left_eye.y + right_eye.y) / 2.0
        face_vec = (nose.x - mid_eye_x, nose.y - mid_eye_y)

        angle_rad = np.arctan2(face_vec[1], face_vec[0])
        angle_deg = np.degrees(angle_rad)
        angle_deg_norm = (angle_deg + 360) % 360

        # Define your thresholds for angles
        front_threshold = 5  # +/- degrees around 90 for front
        # side_threshold = 30  # +/- degrees around 0/180 for side
        # back_threshold = 20  # +/- degrees around 270 for back

        # 身体正面摄像头时，+/-5°的时候，基本上是视线朝向正前方，超过95或者低于85的时候，基本上视线落在在摄像头的左侧或者右侧方位
        if (90 - front_threshold) <= angle_deg_norm <= (90 + front_threshold):
            orientation = 0  # Front

        elif angle_deg_norm < 90 - front_threshold:
            orientation = 1  # Left side (person's left to camera, face points right on image)

        else:
            orientation = 2  # Right side (person's right to camera, face points left on image)

        # elif (angle_deg_norm <= side_threshold) or (angle_deg_norm >= (360 - side_threshold)):
        #     orientation = 1  # Left side (person's left to camera, face points right on image)
        # elif (180 - side_threshold) <= angle_deg_norm <= (180 + side_threshold):
        #     orientation = 2  # Right side (person's right to camera, face points left on image)

        # 面向摄像头的时候，是不可能存在 3 这种状态的
        # elif (270 - back_threshold) <= angle_deg_norm <= (270 + back_threshold):
        #     orientation = 3  # Back
        # else:
        #     orientation = -1  # Unknown

        # print("正面", orientation, angle_deg_norm)

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

        origin_x, origin_y = int(nose.x), int(nose.y)
        return orientation, (vec_x, vec_y), (origin_x, origin_y)

    @staticmethod
    def _analyze_single_eye_face(nose: YoloPoint, left_eye: YoloPoint, right_eye: YoloPoint) -> Tuple[
        int, Tuple[float, float], Tuple[int, int]]:
        """Analyzes facial direction when nose and only one eye are valid."""
        orientation = -1
        vec_x, vec_y = 0.0, 0.0
        origin_x, origin_y = int(nose.x), int(nose.y)

        if PoseInsight._is_valid_point(left_eye) and not PoseInsight._is_valid_point(right_eye):
            orientation = 1  # Left side
            vec_x, vec_y = 1.0, 0.0  # Vector pointing right on image
        elif PoseInsight._is_valid_point(right_eye) and not PoseInsight._is_valid_point(left_eye):
            orientation = 2  # Right side
            vec_x, vec_y = -1.0, 0.0  # Vector pointing left on image
        return orientation, (vec_x, vec_y), (origin_x, origin_y)

    @staticmethod
    def _analyze_back_face_ears_only(left_ear: YoloPoint, right_ear: YoloPoint) -> Tuple[
        int, Tuple[float, float], Tuple[int, int]]:
        """Analyzes facial direction when only both ears are valid (implies back of head)."""
        orientation = 3  # Back
        ear_mid_x = (left_ear.x + right_ear.x) / 2.0
        ear_mid_y = (left_ear.y + right_ear.y) / 2.0
        origin_x, origin_y = int(ear_mid_x), int(ear_mid_y)
        vec_x, vec_y = 0.0, -1.0  # Vector pointing upwards on image
        return orientation, (vec_x, vec_y), (origin_x, origin_y)

    @staticmethod
    def _analyze_facial_direction(pose: Union[YoloPose, YoloPoseSorted]) -> FacialDirection:
        """
        根据 pose 中的面部关键点信息分析面部朝向，返回 FacialDirection 对象。
        """
        modulus = PoseInsight._compute_modulus(pose)
        pts = pose.pts

        # Default values for unknown state
        orientation = -1
        vec_x, vec_y = 0.0, 0.0
        origin_x, origin_y = int(pose.lx), int(pose.ly)

        if len(pts) < 5:  # Not enough keypoints for face analysis
            pass  # Will return unknown based on defaults

        else:
            nose, right_eye, left_eye, right_ear, left_ear = PoseInsight._get_keypoints(pts)

            valid_nose = PoseInsight._is_valid_point(nose)
            valid_right_eye = PoseInsight._is_valid_point(right_eye)
            valid_left_eye = PoseInsight._is_valid_point(left_eye)
            valid_right_ear = PoseInsight._is_valid_point(right_ear)
            valid_left_ear = PoseInsight._is_valid_point(left_ear)

            # Scenario 1: Nose and both eyes are valid (most detailed analysis)
            if valid_nose and valid_left_eye and valid_right_eye:
                orientation, (vec_x, vec_y), (origin_x, origin_y) = \
                    PoseInsight._analyze_front_side_back_face(nose, left_eye, right_eye,
                                                              left_ear, right_ear,
                                                              valid_left_ear, valid_right_ear)

            # Scenario 2: Nose and only one eye is valid (side profile)
            elif valid_nose and (valid_left_eye or valid_right_eye):
                orientation, (vec_x, vec_y), (origin_x, origin_y) = \
                    PoseInsight._analyze_single_eye_face(nose, left_eye, right_eye)

            # Scenario 3: Face keypoints unavailable, but both ears are valid (back profile)
            elif valid_left_ear and valid_right_ear and not (valid_nose or valid_left_eye or valid_right_eye):
                orientation, (vec_x, vec_y), (origin_x, origin_y) = \
                    PoseInsight._analyze_back_face_ears_only(left_ear, right_ear)

            # Scenario 4: Not enough valid keypoints for any specific determination (remains unknown)
            # TODO： 后续我们会将面部、姿态检测的代码都整合成一个模型进行判断

        facial_direction = FacialDirection(
            modulus=modulus,
            vector=(vec_x, vec_y),
            origin=(origin_x, origin_y),
            direction_desc=PoseInsight.ORIENTATION_TEXTS.get(orientation, "Unknown"),
            direction_type=orientation
        )
        return facial_direction

    @staticmethod
    def analyze_poses(poses: List[Union[YoloPose, YoloPoseSorted]]) -> List[Tuple[Posture, FacialDirection]]:
        """
        分析传入的 poses 列表，返回每个 pose 的 Posture 和 FacialDirection 信息，结果格式为：
            [
                (Posture(...), FacialDirection(...)),
                (Posture(...), FacialDirection(...)),
                ...
            ]
        """
        results = []
        for pose in poses:
            posture = PoseInsight._analyze_body_pose(pose)
            facial_direction = PoseInsight._analyze_facial_direction(pose)
            results.append((posture, facial_direction))
        return results