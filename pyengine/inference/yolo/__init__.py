import ctypes
from ctypes import c_char_p, c_bool, c_int, c_float, c_ubyte, POINTER
from typing import Union

import numpy as np

from pyengine.inference.yolo.data_struct import YoloPose, YoloPoint, Yolo


# # 定义结果结构体
# # Define the structure for YoloStruct
# class YoloStruct(ctypes.Structure):
#     _fields_ = [
#         ("lx", c_int),
#         ("ly", c_int),
#         ("rx", c_int),
#         ("ry", c_int),
#         ("conf", c_float),
#         ("cls", c_int)
#     ]


# # Define the structure for YoloPointStruct
# class YoloPointStruct(ctypes.Structure):
#     _fields_ = [
#         ("x", c_int),
#         ("y", c_int),
#         ("conf", c_float)
#     ]


# 封装成 Python 类
class YoloWrapper:
    def __init__(self, model_path: str, use_pose: bool = False):
        self.yolo_pose = use_pose

        #### 加载动态库 ####
        self.c_apis = ctypes.cdll.LoadLibrary('/opt/TrtEngineToolkits/cmake-build-release/libjetson.so')  # 或 'cyolo.dll' on Windows

        #### 函数原型定义 ####

        # void c_yolo_init(const char *model_path, bool b_use_pose=false)
        self.c_apis.c_yolo_init.argtypes = [c_char_p, c_bool]
        self.c_apis.c_yolo_init.restype = None

        # bool c_yolo_release();
        self.c_apis.c_yolo_release.argtypes = []
        self.c_apis.c_yolo_release.restype = c_bool
     
        # bool c_yolo_add_image(int n_index, byte* cstr, int n_channels, int n_width, int n_height)
        self.c_apis.c_yolo_add_image.argtypes = [c_int, POINTER(c_ubyte), c_int, c_int, c_int]
        self.c_apis.c_yolo_add_image.restype = c_bool

        # bool c_yolo_inference()
        self.c_apis.c_yolo_inference.argtypes = []
        self.c_apis.c_yolo_inference.restype = c_bool
       
        # int c_yolo_available_results(int n_index, float f_clsThreshold, float f_nmsThreshold)
        self.c_apis.c_yolo_available_results.argtypes = [c_int, c_float, c_float]
        self.c_apis.c_yolo_available_results.restype = c_int

        # float* c_yolo_get_result(int n_itemIndex)
        self.c_apis.c_yolo_get_result.argtypes = [c_int]
        self.c_apis.c_yolo_get_result.restype = POINTER(c_float)

        #### 初始化 trt engine ####
        self.c_apis.c_yolo_init(model_path.encode('utf-8'), use_pose)

    def release(self):
        return self.c_apis.c_yolo_release()

    def add_image(self, index: int, image: np.ndarray) -> bool:
        assert image.flags['C_CONTIGUOUS'], "Image must be C-contiguous"
        h, w, c = image.shape
        ptr = image.ctypes.data_as(POINTER(c_ubyte))
        return self.c_apis.c_yolo_add_image(index, ptr, c, w, h)

    def inference(self) -> bool:
        return self.c_apis.c_yolo_inference()

    def available_results(self, index: int, cls_thresh: float, nms_thresh: float) -> int:
        return self.c_apis.c_yolo_available_results(index, cls_thresh, nms_thresh)

    def get_result(self, item_index: int) -> Union[None, YoloPose, Yolo]:
        result_ptr = self.c_apis.c_yolo_get_result(item_index)
        if not result_ptr:
            return None

        if self.yolo_pose:
            result_array = np.ctypeslib.as_array(result_ptr, shape=(57,))   # lx, ly, rx, ry, conf, cls, pt1_x, pt1_y, pt1_conf, ...

            pts = []
            for i in range(17):
                pts.append(YoloPoint(x=int(result_array[6 + i * 3]), y=int(result_array[7 + i * 3]), conf=float(result_array[8 + i * 3])))

            pose = YoloPose(lx=int(result_array[0]),
                            ly=int(result_array[1]),
                            rx=int(result_array[2]),
                            ry=int(result_array[3]),
                            cls=int(result_array[4]),
                            conf=float(result_array[5]),
                            pts=pts)
            return pose

        else:
            result_array = np.ctypeslib.as_array(result_ptr, shape=(6,))    # lx, ly, rx, ry, conf, cls

            yolo = Yolo(lx=int(result_array[0]),
                        ly=int(result_array[1]),
                        rx=int(result_array[2]),
                        ry=int(result_array[3]),
                        cls=int(result_array[4]),
                        conf=float(result_array[5]))

            return yolo
