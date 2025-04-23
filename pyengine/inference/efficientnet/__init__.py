import ctypes
from ctypes import c_char_p, c_bool, c_int, c_float, c_ubyte, POINTER
from typing import Union

import numpy as np


# 封装成 Python 类
class EfficientNetWrapper:
    def __init__(self, model_path: str, max_batch_size: int = 1):

        #### 加载动态库 ####
        self.c_apis = ctypes.cdll.LoadLibrary('/opt/TrtEngineToolkits/cmake-build-release/libjetson.so')  # 或 'cyolo.dll' on Windows

        #### 函数原型定义 ####

        # void c_efficient_net_init(const char* engine_file_path, int maximum_batch_size);
        self.c_apis.c_efficient_net_init.argtypes = [c_char_p, c_int]
        self.c_apis.c_efficient_net_init.restype = None

        # void c_efficient_net_release();
        self.c_apis.c_efficient_net_release.argtypes = []
        self.c_apis.c_efficient_net_release.restype = None

        # bool c_efficient_net_add_image(int n_index, byte* cstr, int n_channels, int n_width, int n_height)
        self.c_apis.c_efficient_net_add_image.argtypes = [c_int, POINTER(c_ubyte), c_int, c_int, c_int]
        self.c_apis.c_efficient_net_add_image.restype = c_bool

        # bool c_efficient_net_inference()
        self.c_apis.c_efficient_net_inference.argtypes = []
        self.c_apis.c_efficient_net_inference.restype = c_bool

        # float* c_efficient_net_get_result(int n_index, int* n_size)
        self.c_apis.c_efficient_net_get_result.argtypes = [c_int, POINTER(c_int)]
        self.c_apis.c_efficient_net_get_result.restype = POINTER(c_float)

        # 初始化 trt engine
        self.c_apis.c_efficient_net_init(model_path.encode('utf-8'), max_batch_size)

    def release(self):
        self.c_apis.c_efficient_net_release()

    def add_image(self, index: int, image: np.ndarray) -> bool:
        assert image.flags['C_CONTIGUOUS'], "Image must be C-contiguous"
        h, w, c = image.shape
        ptr = image.ctypes.data_as(POINTER(c_ubyte))
        return self.c_apis.c_efficient_net_add_image(index, ptr, c, w, h)

    def inference(self) -> bool:
        return self.c_apis.c_efficient_net_inference()
    

    def get_result(self, item_index: int) -> Union[np.ndarray, None]:
        """
        获取推理结果
        :param item_index: 图像索引
        :return: 推理结果，None表示失败
        """
        size = c_int(0)
        result_ptr = self.c_apis.c_efficient_net_get_result(item_index, ctypes.byref(size))
        if not result_ptr or size.value <= 0:
            return None
        
        # 将结果转换为 numpy 数组
        result_array = np.ctypeslib.as_array(result_ptr, shape=(size.value,))
        return result_array
