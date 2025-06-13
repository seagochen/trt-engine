import os
from ctypes import c_bool, c_int, c_ubyte, POINTER, cdll
from typing import Union

import numpy as np

from pyengine import load_build_dir


class BaseEngineWrapper:
    """
    Base class for Python wrappers around C/C++ inference engines
    loaded via  Handles common library loading and function mapping.
    """

    def __init__(self, model_path: str, lib_name: str = 'libjetson.so'):
        """
        Initializes the Ctypes wrapper by loading the shared library
        and defining common function prototypes.

        Args:
            model_path: Path to the model engine file.
            lib_name: The name of the shared library file (e.g., 'libjetson.so').
        """
        # Load the dynamic library
        lib_path = os.path.join(load_build_dir(), lib_name)
        try:
            self.c_apis = cdll.LoadLibrary(lib_path)
            print(f"Loaded C shared library from: {lib_path}")
        except OSError as e:
            raise RuntimeError(f"Error loading shared library {lib_path}: {e}")

        # Common function prototype definitions (prefix will be set by derived classes)
        self._define_common_c_api_prototypes()

        self._model_path = model_path
        self._is_initialized = False # Track if the underlying C engine is initialized

    def _define_common_c_api_prototypes(self):
        """
        Defines the argtypes and restypes for common C API functions.
        Concrete function names will be set in derived classes.
        """
        # Placeholders for the actual C function objects
        self._c_init_func = None
        self._c_release_func = None
        self._c_add_image_func = None
        self._c_inference_func = None
        self._c_get_result_func = None # Optional, for classification-like results

        # Common argument types (actual assignment happens in derived classes)
        # add_image: (int index, byte* image_ptr, int channels, int width, int height)
        c_add_image_argtypes = [c_int, POINTER(c_ubyte), c_int, c_int, c_int]
        c_add_image_restype = c_bool

        # inference: ()
        c_inference_argtypes = []
        c_inference_restype = c_bool

        # release: ()
        c_release_argtypes = []
        c_release_restype = None # Or c_bool if it returns success/failure

        # Derived classes will then do something like:
        # self.c_apis.c_efficient_net_add_image.argtypes = c_add_image_argtypes
        # self.c_apis.c_efficient_net_add_image.restype = c_add_image_restype


    def _init_engine(self, *args):
        """
        Internal method to call the C initialization function.
        Derived classes must set self._c_init_func.
        """
        if self._c_init_func is None:
            raise NotImplementedError("C initialization function not defined in derived class.")
        if self._is_initialized:
            print("Warning: Engine already initialized. Skipping re-initialization.")
            return

        try:
            self._c_init_func(self._model_path.encode('utf-8'), *args)
            self._is_initialized = True
            print("C engine initialized successfully.")
        except Exception as e:
            print(f"Error initializing C engine: {e}")
            raise

    def release(self) -> Union[bool, None]:
        """
        Releases the underlying C engine resources.
        Returns the result of the C release function if it has one, otherwise None.
        """
        if not self._is_initialized:
            print("Warning: Engine not initialized. No resources to release.")
            return None
        if self._c_release_func is None:
            raise NotImplementedError("C release function not defined in derived class.")
        try:
            result = self._c_release_func()
            self._is_initialized = False
            print("C engine resources released.")
            return result
        except Exception as e:
            print(f"Error releasing C engine: {e}")
            raise

    def add_image(self, index: int, image: np.ndarray) -> bool:
        """
        Adds an image to the C engine's batch buffer.
        Image is expected to be HWC, BGR, uint8.
        """
        if not self._is_initialized:
            print("Error: Engine not initialized. Call __init__ first.")
            return False
        if self._c_add_image_func is None:
            raise NotImplementedError("C add_image function not defined in derived class.")

        assert image.flags['C_CONTIGUOUS'], "Image must be C-contiguous"
        h, w, c = image.shape
        ptr = image.data_as(POINTER(c_ubyte))
        return self._c_add_image_func(index, ptr, c, w, h)

    def inference(self) -> bool:
        """
        Performs inference on the current batch of images in the C engine.
        """
        if not self._is_initialized:
            print("Error: Engine not initialized. Call __init__ first.")
            return False
        if self._c_inference_func is None:
            raise NotImplementedError("C inference function not defined in derived class.")
        return self._c_inference_func()

    # The get_result method is too different between EfficientNet and YOLO
    # so it will remain abstract/implemented directly in derived classes.
    # def get_result(self, item_index: int) -> Any:
    #     raise NotImplementedError("Subclasses must implement get_result method.")
