from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np

from pyengine.inference.basic_onnx_wrapper import BaseOnnxWrapper


class EfficientNetOnnxWrapper(BaseOnnxWrapper):
    """
    Wraps an EfficientNet ONNX model for inference.
    """
    def __init__(self, model_path: str, max_batch_size: int = 1, providers: List[str] = None):

        super().__init__(model_path, max_batch_size, providers)

        # Ensure input height and width are determined from the base class
        # or set sensible defaults if the model shape was dynamic.
        if self.input_height is None or self.input_width is None:
            print("EfficientNetOnnxWrapper: Input H/W not fixed in model, defaulting to 224x224.")
            self.input_height = 224
            self.input_width = 224

        # Specific result storage for EfficientNet
        self._results: Dict[int, np.ndarray] = {}

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Preprocesses a single image (HWC, BGR, uint8) for EfficientNet.
        Returns the preprocessed tensor and an empty dict for metadata (not needed here).
        """
        img_resized = cv2.resize(image, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_float - mean) / std

        img_chw = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_chw, axis=0) # Shape: [1, C, H, W]

        # No specific metadata needed for postprocessing EfficientNet
        return img_batch.astype(np.float32), {}

    def _postprocess(self, raw_output_batch: np.ndarray, inference_meta: Dict) -> Dict[int, np.ndarray]:
        """
        Postprocesses the raw output batch from the EfficientNet model.
        For classification, this typically involves just mapping back to original indices.
        """
        processed_results = {}
        original_indices = inference_meta['indices']
        actual_batch_size = inference_meta['actual_batch_size']

        if raw_output_batch.shape[0] != actual_batch_size:
            print(f"Error: Output batch size ({raw_output_batch.shape[0]}) != Input batch size ({actual_batch_size})")
            return {}

        for i in range(actual_batch_size):
            processed_results[original_indices[i]] = raw_output_batch[i]
        return processed_results

    def get_result(self, item_index: int) -> Optional[np.ndarray]:
        """
        Retrieves the inference result for a specific image index from the last batch.
        This triggers postprocessing if it hasn't been done for the current raw results.

        Args:
            item_index: The original index of the image provided to add_image.

        Returns:
            A NumPy array containing the model's output for that image,
            or None if the index is not found or inference failed.
        """
        if self.session is None:
            print("Error: Session has been released.")
            return None

        # If raw results are available and postprocessing hasn't happened for these results
        if self._raw_inference_output is not None and not self._results:
            self._results = self._postprocess(self._raw_inference_output, self._inference_meta)
            self._raw_inference_output = None # Clear raw results after processing them

        result_array = self._results.get(item_index)

        if result_array is not None:
            return np.copy(result_array) # Return a copy
        else:
            return None

    def release(self):
        """Releases resources and clears internal state."""
        super().release()
        self._results.clear()