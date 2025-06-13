from typing import Union

import numpy as np

from pyengine.inference.basic_engine_wrapper import BaseEngineWrapper
from ctypes import c_char_p, c_bool, c_int, c_float, c_ubyte, POINTER

from pyengine.inference.extend.yolo.data_struct import YoloPose, Yolo, YoloPoint


class YoloEngineWrapper(BaseEngineWrapper):
    def __init__(self, model_path: str, use_pose: bool = False):
        super().__init__(model_path)
        self.yolo_pose = use_pose

        # Map specific C functions to the generic placeholders
        self._c_init_func = self.c_apis.c_yolo_init
        self._c_release_func = self.c_apis.c_yolo_release
        self._c_add_image_func = self.c_apis.c_yolo_add_image
        self._c_inference_func = self.c_apis.c_yolo_inference
        # Yolo has a different get_result flow, so we map these specifically
        self._c_available_results_func = self.c_apis.c_yolo_available_results
        self._c_get_result_single_func = self.c_apis.c_yolo_get_result # Get single detection result

        # Define argtypes and restypes for specific C API functions
        self._c_init_func.argtypes = [c_char_p, c_bool]
        self._c_init_func.restype = None

        self._c_release_func.argtypes = []
        self._c_release_func.restype = c_bool

        self._c_add_image_func.argtypes = [c_int, POINTER(c_ubyte), c_int, c_int, c_int]
        self._c_add_image_func.restype = c_bool

        self._c_inference_func.argtypes = []
        self._c_inference_func.restype = c_bool

        self._c_available_results_func.argtypes = [c_int, c_float, c_float]
        self._c_available_results_func.restype = c_int

        self._c_get_result_single_func.argtypes = [c_int]
        self._c_get_result_single_func.restype = POINTER(c_float)

        # Initialize the underlying C engine
        self._init_engine(use_pose)

    def available_results(self, index: int, cls_thresh: float, nms_thresh: float) -> int:
        """
        Queries the C engine for the number of available results for a given image
        after applying confidence and NMS thresholds.
        """
        if not self._is_initialized:
            print("Error: Engine not initialized. Call __init__ first.")
            return 0
        if self._c_available_results_func is None:
            raise NotImplementedError("C available_results function not defined.")
        return self._c_available_results_func(index, cls_thresh, nms_thresh)

    def get_result(self, item_index: int) -> Union[None, YoloPose, Yolo]:
        """
        Retrieves a specific detection result (Yolo or YoloPose object)
        from the C engine based on its internal item_index.
        """
        if not self._is_initialized:
            print("Error: Engine not initialized. Call __init__ first.")
            return None
        if self._c_get_result_single_func is None:
            raise NotImplementedError("C get_result function not defined.")

        result_ptr = self._c_get_result_single_func(item_index)
        if not result_ptr:
            return None

        if self.yolo_pose:
            # Assuming a fixed size for pose output (e.g., 57: 4 bbox + 1 conf + 1 cls + 17*3 kpts)
            # You might need to make this size dynamic if your C API can provide it.
            result_array = np.ctypeslib.as_array(result_ptr, shape=(57,))

            pts = []
            for i in range(17):
                pts.append(YoloPoint(x=int(result_array[6 + i * 3]), y=int(result_array[7 + i * 3]),
                                     conf=float(result_array[8 + i * 3])))

            pose = YoloPose(lx=int(result_array[0]),
                            ly=int(result_array[1]),
                            rx=int(result_array[2]),
                            ry=int(result_array[3]),
                            cls=int(result_array[4]),
                            conf=float(result_array[5]),
                            pts=pts)
            return pose

        else:
            # Assuming a fixed size for detection output (e.g., 6: lx, ly, rx, ry, conf, cls)
            result_array = np.ctypeslib.as_array(result_ptr, shape=(6,))

            yolo = Yolo(lx=int(result_array[0]),
                        ly=int(result_array[1]),
                        rx=int(result_array[2]),
                        ry=int(result_array[3]),
                        cls=int(result_array[4]),
                        conf=float(result_array[5]))
            return yolo