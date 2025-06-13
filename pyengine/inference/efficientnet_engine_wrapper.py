from ctypes import c_char_p, c_bool, c_int, c_float, c_ubyte, byref, POINTER
from typing import Optional

import numpy as np
from pyengine.inference.basic_engine_wrapper import BaseEngineWrapper


class EfficientNetEngineWrapper(BaseEngineWrapper):

    def __init__(self, model_path: str, max_batch_size: int = 1):
        super().__init__(model_path)

        # Map specific C functions to the generic placeholders
        self._c_init_func = self.c_apis.c_efficient_net_init
        self._c_release_func = self.c_apis.c_efficient_net_release
        self._c_add_image_func = self.c_apis.c_efficient_net_add_image
        self._c_inference_func = self.c_apis.c_efficient_net_inference
        self._c_get_result_func = self.c_apis.c_efficient_net_get_result

        # Define argtypes and restypes for specific C API functions
        self._c_init_func.argtypes = [c_char_p, c_int]
        self._c_init_func.restype = None

        self._c_release_func.argtypes = []
        self._c_release_func.restype = None

        self._c_add_image_func.argtypes = [c_int, POINTER(c_ubyte), c_int, c_int, c_int]
        self._c_add_image_func.restype = c_bool

        self._c_inference_func.argtypes = []
        self._c_inference_func.restype = c_bool

        self._c_get_result_func.argtypes = [c_int, POINTER(c_int)]
        self._c_get_result_func.restype = POINTER(c_float)

        # Initialize the underlying C engine
        self._init_engine(max_batch_size)

    def get_result(self, item_index: int) -> Optional[np.ndarray]:
        """
        Retrieves the inference result for a specific image index from the C engine.
        """
        if not self._is_initialized:
            print("Error: Engine not initialized. Call __init__ first.")
            return None
        if self._c_get_result_func is None:
            raise NotImplementedError("C get_result function not defined.")

        size = c_int(0)
        result_ptr = self._c_get_result_func(item_index, byref(size))
        if not result_ptr or size.value <= 0:
            return None

        # Convert C array to numpy array
        result_array = np.ctypeslib.as_array(result_ptr, shape=(size.value,))

        # Return a copy to prevent external modification of internal C memory
        return np.copy(result_array)