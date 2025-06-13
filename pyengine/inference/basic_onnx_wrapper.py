import traceback
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import onnxruntime

from pyengine.inference.extend.yolo.data_struct import Yolo, YoloPose


class BaseOnnxWrapper:
    """
    Base class for ONNX model wrappers, providing common functionalities
    like session management, input/output parsing, and batching.
    """

    def __init__(self, model_path: str, max_batch_size: int = 1,
                 providers: List[str] = None):
        """
        Initializes the ONNX Runtime session and prepares for inference.

        Args:
            model_path: Path to the ONNX model file.
            max_batch_size: The maximum number of items that can be processed
                            in a single inference call.
            providers: List of ONNX Runtime execution providers to use
                       (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']).
        """
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive.")

        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.max_batch_size = max_batch_size
        self.session_options = onnxruntime.SessionOptions()

        try:
            print(f"Loading ONNX model from: {model_path}")
            print(f"Using Execution Providers: {providers}")
            self.session = onnxruntime.InferenceSession(
                model_path,
                sess_options=self.session_options,
                providers=providers
            )
            print("ONNX model loaded successfully.")
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            raise

        # Get model input details
        model_inputs = self.session.get_inputs()
        if not model_inputs:
            raise RuntimeError("Could not get model inputs from ONNX file.")
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape  # e.g., [None, 3, 224, 224] or [1, 3, 640, 640]

        # Determine input height and width from model input shape
        try:
            # Assuming shape is [batch, channels, height, width]
            self.input_height = int(self.input_shape[2])
            self.input_width = int(self.input_shape[3])
            if not isinstance(self.input_height, int) or not isinstance(self.input_width, int):
                # Fallback for dynamic H/W (though most models have fixed H/W)
                raise ValueError("Model input height/width are not fixed dimensions.")
            print(f"Model expects input shape like: [Batch, Channels, {self.input_height}, {self.input_width}]")
        except (IndexError, TypeError, ValueError) as e:
            # This can happen if the model has dynamic H/W or unexpected input shape
            print(f"Warning: Could not reliably determine input H/W from shape {self.input_shape}. "
                  "You might need to set `input_height` and `input_width` manually in derived class if dynamic. Error: {e}")
            self.input_height = None
            self.input_width = None

        # Get model output details
        model_outputs = self.session.get_outputs()
        if not model_outputs:
            raise RuntimeError("Could not get model outputs from ONNX file.")
        self.output_name = model_outputs[0].name  # Assuming single output
        self.output_shape = model_outputs[0].shape
        print(f"Model output name: {self.output_name}, shape: {self.output_shape}")

        # Internal buffer for batching. Derived classes will define what exactly is stored.
        # It's a list of (original_index, preprocessed_data, ...)
        self._batch_buffer: List[Tuple] = []
        # Stores raw output tensor from the last inference
        self._raw_inference_output: Optional[np.ndarray] = None
        # Stores metadata associated with the raw results (e.g., original indices, original sizes)
        self._inference_meta: Dict = {}

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Abstract method for preprocessing a single image.
        Derived classes must implement this.
        Returns the preprocessed tensor and any metadata needed for postprocessing.
        """
        raise NotImplementedError("Subclasses must implement _preprocess method.")

    def _postprocess(self, raw_output: np.ndarray, inference_meta: Dict) -> Dict[int, List[Union[Yolo, YoloPose, np.ndarray]]]:
        """
        Abstract method for postprocessing the raw model output.
        Derived classes must implement this.
        Returns a dictionary mapping original image index to its processed results.
        """
        raise NotImplementedError("Subclasses must implement _postprocess method.")

    def release(self):
        """
        Releases resources. For ONNX Runtime, this primarily clears buffers.
        The session itself is managed by Python's garbage collector.
        """
        print("Releasing ONNX wrapper resources...")
        self.session = None  # Allow session to be garbage collected
        self._batch_buffer.clear()
        self._raw_inference_output = None
        self._inference_meta.clear()
        print("ONNX wrapper resources released.")

    def add_image(self, index: int, image: np.ndarray) -> bool:
        """
        Preprocesses and adds an image to the current batch buffer.

        Args:
            index: An integer identifier for this image.
            image: The input image (NumPy array, expected HWC, BGR, uint8).

        Returns:
            True if the image was added successfully, False if the batch is full or input invalid.
        """
        if self.session is None:
            print("Error: Session has been released. Cannot add image.")
            return False

        if len(self._batch_buffer) >= self.max_batch_size:
            print(f"Warning: Batch buffer is full (max size: {self.max_batch_size}). Cannot add image index {index}.")
            return False

        if len(image.shape) != 3 or image.shape[2] != 3:
            print(f"Error: Image for index {index} has unexpected shape {image.shape}. Expected HWC BGR.")
            return False

        try:
            preprocessed_tensor, meta_data = self._preprocess(image)
            self._batch_buffer.append((index, preprocessed_tensor, meta_data))
            return True

        except Exception as e:
            print(f"Error preprocessing image for index {index}: {e}")
            traceback.print_exc()
            return False

    def inference(self) -> bool:
        """
        Performs inference on the current batch of images.
        Clears the batch buffer afterwards. Stores raw results internally.

        Returns:
            True if inference was successful, False otherwise.
        """
        if self.session is None:
            print("Error: Session has been released. Cannot perform inference.")
            return False
        if not self._batch_buffer:
            print("Warning: Inference called with empty batch buffer.")
            return False

        # Prepare batch tensor and gather metadata for postprocessing
        batch_input_tensors = [item[1] for item in self._batch_buffer]
        original_indices = [item[0] for item in self._batch_buffer]
        # Collect metadata from preprocessing for each image
        preprocessing_meta = [item[2] for item in self._batch_buffer]

        # Concatenate preprocessed images into a single batch tensor
        try:
            batch_input_tensor = np.concatenate(batch_input_tensors, axis=0)
        except ValueError as e:
            print(f"Error concatenating batch input tensors: {e}")
            self._batch_buffer.clear()
            return False

        actual_batch_size = batch_input_tensor.shape[0]

        # Store metadata for later postprocessing
        self._inference_meta = {
            'indices': original_indices,
            'preprocessing_meta': preprocessing_meta,
            'actual_batch_size': actual_batch_size
        }
        self._raw_inference_output = None # Clear previous raw output

        try:
            input_feed = {self.input_name: batch_input_tensor}
            outputs = self.session.run([self.output_name], input_feed)

            if outputs and len(outputs) > 0:
                self._raw_inference_output = outputs[0]
                print(f"Inference successful for batch of {actual_batch_size}.")
                return True
            else:
                print("Error: Inference did not return any outputs.")
                return False

        except Exception as e:
            error_info = traceback.format_exc()
            print(f"Exception during inference: {e}\n{error_info}")
            return False

        finally:
            self._batch_buffer.clear() # Always clear the input buffer
