from common.yolo.ultralytics_results_wrapper import YoloPose
from typing import List


# class FacialOrientation2D:
#     ORIENTATION_TEXTS = {0: "Face front", 1: "Face left", 2: "Face right", 3: "Face back", -1: "Unknown"}
    
#     def __init__(self, length=100):
#         """
#         Initialize the FacialOrientation2D class with default values.

#         :param length: Length of the direction vector for visualization purposes.
#         """
#         self.origin_x = self.origin_y = 0.0
#         self.vec_x = self.vec_y = 0.0
#         self.dest_x = self.dest_y = 0.0
#         self.lx = self.ly = 0
#         self.length = length
#         self.orientation = -1

#     def check_orientation(self, pose: YoloPose):
#         """
#         Determine facial orientation based on keypoints in the pose.

#         :param pose: YoloPose object with keypoints for face orientation analysis.
#         """
#         self.lx, self.ly = pose.lx, pose.ly
#         nose_pt, right_ear_pt, left_ear_pt = pose.pts[0], pose.pts[3], pose.pts[4]
#         ear_tmp_x, ear_tmp_y = 0, 0

#         if self._is_valid_pt(nose_pt):
#             if self._is_valid_pt(right_ear_pt) and self._is_valid_pt(left_ear_pt):
#                 self.orientation = 0  # Front
#             elif self._is_valid_pt(right_ear_pt):
#                 self.orientation, ear_tmp_x, ear_tmp_y = 2, right_ear_pt.x, right_ear_pt.y  # Right
#             elif self._is_valid_pt(left_ear_pt):
#                 self.orientation, ear_tmp_x, ear_tmp_y = 1, left_ear_pt.x, left_ear_pt.y  # Left
#             else:
#                 self.orientation = -1  # Unknown
#         elif self._is_valid_pt(right_ear_pt) or self._is_valid_pt(left_ear_pt):
#             self.orientation = 3  # Back
#         else:
#             self.orientation = -1  # Unknown

#         if self.orientation in [1, 2]:
#             self._calculate_vector(nose_pt, ear_tmp_x, ear_tmp_y)
#         elif self.orientation == 3:
#             self._set_front_face(nose_pt)
#         else:
#             self._reset_orientation()

#         self.origin_x, self.origin_y = int(self.origin_x), int(self.origin_y)
#         self.dest_x, self.dest_y = int(self.dest_x), int(self.dest_y)

#     def _calculate_vector(self, nose_pt, ear_x, ear_y):
#         self.origin_x, self.origin_y = nose_pt.x, nose_pt.y
#         self.vec_x, self.vec_y = self._normalize(nose_pt.x - ear_x, nose_pt.y - ear_y)
#         self.dest_x = self.origin_x + self.vec_x * self.length
#         self.dest_y = self.origin_y + self.vec_y * self.length

#     def _set_front_face(self, nose_pt):
#         self.origin_x, self.origin_y = nose_pt.x, nose_pt.y
#         self.vec_x = self.vec_y = 0
#         self.dest_x = self.dest_y = nose_pt.x, nose_pt.y

#     def _reset_orientation(self):
#         self.origin_x = self.origin_y = 0
#         self.vec_x = self.vec_y = 0
#         self.dest_x = self.dest_y = 0

#     @staticmethod
#     def _is_valid_pt(pt):
#         return pt.conf > 0.2 and pt.x > 0 and pt.y > 0

#     @staticmethod
#     def _normalize(x, y):
#         vec_len = (x ** 2 + y ** 2) ** 0.5
#         return (x / vec_len, y / vec_len) if vec_len != 0 else (0, 0)

#     def __str__(self):
#         return self.ORIENTATION_TEXTS.get(self.orientation, "Unknown")


# class FacialOrientation2D:
#     ORIENTATION_TEXTS = {0: "Face front", 1: "Face left", 2: "Face right", 3: "Face back", -1: "Unknown"}
    
#     def __init__(self, length=100):
#         self.origin_x = self.origin_y = 0.0
#         self.vec_x = self.vec_y = 0.0
#         self.dest_x = self.dest_y = 0.0
#         self.lx = self.ly = 0
#         self.length = length
#         self.orientation = -1

#     def check_orientation(self, pose: YoloPose):
#         self.lx, self.ly = pose.lx, pose.ly
#         nose_pt, right_ear_pt, left_ear_pt = pose.pts[0], pose.pts[3], pose.pts[4]
#         ear_tmp_x, ear_tmp_y = 0, 0

#         if self._is_valid_pt(nose_pt):
#             if self._is_valid_pt(right_ear_pt) and self._is_valid_pt(left_ear_pt):
#                 self.orientation = 0
#             elif self._is_valid_pt(right_ear_pt):
#                 self.orientation, ear_tmp_x, ear_tmp_y = 2, right_ear_pt.x, right_ear_pt.y
#             elif self._is_valid_pt(left_ear_pt):
#                 self.orientation, ear_tmp_x, ear_tmp_y = 1, left_ear_pt.x, left_ear_pt.y
#             else:
#                 self.orientation = -1
#         elif self._is_valid_pt(right_ear_pt) or self._is_valid_pt(left_ear_pt):
#             self.orientation = 3
#         else:
#             self.orientation = -1

#         if self.orientation in [1, 2]:
#             self._calculate_vector(nose_pt, ear_tmp_x, ear_tmp_y)
#         elif self.orientation == 3:
#             self._set_front_face(nose_pt)
#         else:
#             self._reset_orientation()

#         self.origin_x, self.origin_y = int(self.origin_x), int(self.origin_y)
#         self.dest_x, self.dest_y = int(self.dest_x), int(self.dest_y)

#     def _calculate_vector(self, nose_pt, ear_x, ear_y):
#         self.origin_x, self.origin_y = nose_pt.x, nose_pt.y
#         self.vec_x, self.vec_y = self._normalize(nose_pt.x - ear_x, nose_pt.y - ear_y)
#         self.dest_x = self.origin_x + self.vec_x * self.length
#         self.dest_y = self.origin_y + self.vec_y * self.length

#     # def _set_front_face(self, nose_pt):
#     #     self.origin_x, self.origin_y = nose_pt.x, nose_pt.y
#     #     self.vec_x = self.vec_y = 0
#     #     self.dest_x, self.dest_y = nose_pt.x, nose_pt.y  # 修复为解包赋值

#     def _set_front_face(self, nose_pt):
#         """Set values for a front-facing orientation."""
#         self.origin_x, self.origin_y = nose_pt.x, nose_pt.y
#         self.dest_x, self.dest_y = nose_pt.x, nose_pt.y  # 确保起点和终点都为鼻子位置
#         self.vec_x = self.vec_y = 0  # 确保向量为零        

#     def _reset_orientation(self):
#         self.origin_x = self.origin_y = 0
#         self.vec_x = self.vec_y = 0
#         self.dest_x = self.dest_y = 0

#     @staticmethod
#     def _is_valid_pt(pt):
#         return pt.conf > 0.2 and pt.x > 0 and pt.y > 0

#     @staticmethod
#     def _normalize(x, y):
#         vec_len = (x ** 2 + y ** 2) ** 0.5
#         return (x / vec_len, y / vec_len) if vec_len != 0 else (0, 0)

#     def __str__(self):
#         return self.ORIENTATION_TEXTS.get(self.orientation, "Unknown")


class FacialOrientation2D:
    ORIENTATION_TEXTS = {0: "Face front", 1: "Face left", 2: "Face right", 3: "Face back", -1: "Unknown"}
    
    def __init__(self, length=100):
        self.origin_x = self.origin_y = 0.0
        self.vec_x = self.vec_y = 0.0
        self.dest_x = self.dest_y = 0.0
        self.lx = self.ly = 0
        self.length = length
        self.orientation = -1

    def check_orientation(self, pose: YoloPose):
        self.lx, self.ly = pose.lx, pose.ly
        nose_pt, right_ear_pt, left_ear_pt = pose.pts[0], pose.pts[3], pose.pts[4]
        ear_tmp_x, ear_tmp_y = 0, 0

        if self._is_valid_pt(nose_pt):
            if self._is_valid_pt(right_ear_pt) and self._is_valid_pt(left_ear_pt):
                self.orientation = 0
            elif self._is_valid_pt(right_ear_pt):
                self.orientation, ear_tmp_x, ear_tmp_y = 2, right_ear_pt.x, right_ear_pt.y
            elif self._is_valid_pt(left_ear_pt):
                self.orientation, ear_tmp_x, ear_tmp_y = 1, left_ear_pt.x, left_ear_pt.y
            else:
                self.orientation = -1
        elif self._is_valid_pt(right_ear_pt) or self._is_valid_pt(left_ear_pt):
            self.orientation = 3
        else:
            self.orientation = -1

        if self.orientation in [1, 2]:
            self._calculate_vector(nose_pt, ear_tmp_x, ear_tmp_y)
        elif self.orientation == 0:  # 修复为正面情况
            self._set_front_face(nose_pt)
        else:
            self._reset_orientation()

        self.origin_x, self.origin_y = int(self.origin_x), int(self.origin_y)
        self.dest_x, self.dest_y = int(self.dest_x), int(self.dest_y)

    def _calculate_vector(self, nose_pt, ear_x, ear_y):
        self.origin_x, self.origin_y = nose_pt.x, nose_pt.y
        self.vec_x, self.vec_y = self._normalize(nose_pt.x - ear_x, nose_pt.y - ear_y)
        self.dest_x = self.origin_x + self.vec_x * self.length
        self.dest_y = self.origin_y + self.vec_y * self.length

    def _set_front_face(self, nose_pt):
        """Set values for a front-facing orientation."""
        self.origin_x, self.origin_y = nose_pt.x, nose_pt.y
        self.dest_x, self.dest_y = nose_pt.x, nose_pt.y  # 确保起点和终点都为鼻子位置
        self.vec_x = self.vec_y = 0  # 确保向量为零

    def _reset_orientation(self):
        self.origin_x = self.origin_y = 0
        self.vec_x = self.vec_y = 0
        self.dest_x = self.dest_y = 0

    @staticmethod
    def _is_valid_pt(pt):
        return pt.conf > 0.2 and pt.x > 0 and pt.y > 0

    @staticmethod
    def _normalize(x, y):
        vec_len = (x ** 2 + y ** 2) ** 0.5
        return (x / vec_len, y / vec_len) if vec_len != 0 else (0, 0)

    def __str__(self):
        return self.ORIENTATION_TEXTS.get(self.orientation, "Unknown")



def detect_facial_orientations(results: List[YoloPose], return_type="str", return_vectors=False) -> List:
    """
    Determine facial orientations and optionally return facial orientation vectors.

    :param results: List of YoloPose objects with keypoints for face orientation.
    :param return_type: Specify "str" for text or any other value for numeric output.
    :param return_vectors: If True, returns the FacialOrientation2D objects in addition to orientation.
    :return: List of facial orientations as strings or integers (and optionally vectors).
    """
    facial_orientations = []
    facial_vectors = []

    for pose in results:
        vector = FacialOrientation2D()
        vector.check_orientation(pose)
        facial_vectors.append(vector)

        if return_type == "str":
            orientation = vector.ORIENTATION_TEXTS[vector.orientation]
        else:
            orientation = vector.orientation

        facial_orientations.append(orientation)

    return (facial_orientations, facial_vectors) if return_vectors else facial_orientations
