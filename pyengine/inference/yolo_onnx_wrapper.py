import traceback
from typing import Union, List, Tuple, Dict, Optional

import cv2
import numpy as np

from pyengine.inference.basic_onnx_wrapper import BaseOnnxWrapper
from pyengine.inference.extend.yolo.data_struct import YoloPose, Yolo, YoloPoint


class YoloOnnxWrapper(BaseOnnxWrapper):
    """
    Wraps a YOLO/YOLO-Pose ONNX model for inference.
    Handles YOLO-specific preprocessing and postprocessing (letterboxing, NMS, scaling).
    """

    def __init__(self, model_path: str, use_pose: bool = False, providers: List[str] = None):
        # max_batch_size for YOLO models is often fixed at 1, but can be dynamic
        # depending on how the model was exported. We'll set it to 1 by default
        # and let the user override if their model supports higher batch size.
        super().__init__(model_path, max_batch_size=1,
                         providers=providers)  # YOLO models often have fixed batch size of 1

        self.use_pose = use_pose

        # Ensure input height and width are determined from the base class
        if self.input_height is None or self.input_width is None:
            raise RuntimeError("YOLO models typically require fixed input height and width.")

        # Determine number of classes and keypoints from output shape
        try:
            output_dims = self.output_shape[1]  # e.g., 84 or 56 for YOLOv8
            if self.use_pose:
                if output_dims == 56:  # Typical YOLOv8-Pose (1 class, 17 keypoints)
                    self.num_classes = 1
                    self.num_keypoints = 17
                    print("Interpreting output as YOLOv8-Pose (1 class, 17 kpts)")
                elif (output_dims - 5) % 3 == 0 and output_dims > 5:  # General check for kpts
                    self.num_keypoints = (output_dims - 5) // 3
                    self.num_classes = 1  # Assume 1 class if kpts present in general pose models
                    print(
                        f"Warning: Assuming YOLO-Pose with {self.num_keypoints} kpts and 1 class based on output dim {output_dims}")
                else:
                    raise ValueError(f"Cannot determine pose structure from output dim {output_dims}")
            else:  # Detection
                self.num_keypoints = 0
                if output_dims > 4:  # YOLOv8 detection: 4 (bbox) + num_classes
                    self.num_classes = output_dims - 4
                    print(f"Interpreting output as YOLOv8-Detection ({self.num_classes} classes)")
                else:
                    raise ValueError(f"Cannot determine detection structure from output dim {output_dims}")
        except Exception as e:
            # Fallback for models with non-standard output shapes or if parsing fails
            print(f"Could not automatically determine num_classes/keypoints from output shape {self.output_shape}: {e}")
            self.num_classes = 80 if not use_pose else 1  # Common defaults
            self.num_keypoints = 17 if use_pose else 0
            print(f"Defaulting to num_classes={self.num_classes}, num_keypoints={self.num_keypoints}")

        # Specific result storage for YOLO
        # Stores post-processed results, keyed by original image index
        self._result_map: Dict[int, List[Union[Yolo, YoloPose]]] = {}
        # Stores the index for which available_results was last called
        self._last_available_index: Optional[int] = None

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Preprocesses a single image (HWC, BGR, uint8) for YOLOv8 models.
        Applies letterboxing and normalization.

        Returns:
            Tuple: (preprocessed_image_tensor [1, C, H, W],
                    dict containing 'original_hw' and 'scaling_pads' for postprocessing)
        """
        img_h, img_w = image.shape[:2]
        target_h, target_w = self.input_height, self.input_width

        # Calculate scaling ratio and new size
        ratio = min(target_w / img_w, target_h / img_h)
        new_unpad_w, new_unpad_h = int(round(img_w * ratio)), int(round(img_h * ratio))
        pad_w, pad_h = (target_w - new_unpad_w) / 2, (target_h - new_unpad_h) / 2

        # Resize
        if (new_unpad_w, new_unpad_h) != (img_w, img_h):
            img_resized = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = image

        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

        # Pad with gray
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=(114, 114, 114))

        # BGR to RGB, HWC to CHW, Normalize to [0, 1]
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_chw = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32)
        img_normalized = img_chw / 255.0

        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)  # Shape: [1, C, H, W]

        # Store necessary info for postprocessing scaling
        meta_data = {
            'original_hw': (img_h, img_w),
            'scaling_pads': (ratio, ratio, top, left)  # scale_h, scale_w, pad_top, pad_left
        }
        return img_batch.astype(np.float32), meta_data

    def _postprocess(self,
                     raw_output_batch: np.ndarray,
                     inference_meta: Dict,
                     conf_threshold: float = 0.25,  # Default values, can be overridden by available_results
                     nms_threshold: float = 0.45) -> Dict[int, List[Union[Yolo, YoloPose]]]:
        """
        Postprocesses the raw output batch from YOLOv8 ONNX model.
        Includes confidence filtering, NMS, and coordinate scaling.
        """
        processed_results = {}
        original_indices = inference_meta['indices']
        preprocessing_meta_list = inference_meta['preprocessing_meta']
        num_images_in_batch = raw_output_batch.shape[0]

        # Output format expected: [batch, 4_xywh + 1_conf + num_classes (+ kpts*3), num_proposals]
        # Transpose to [batch, num_proposals, 4+1+classes (+kpts*3)] for easier processing
        # Note: If your model output is already [batch, num_proposals, Dims], skip transpose
        outputs = np.transpose(raw_output_batch, (0, 2, 1))

        for i in range(num_images_in_batch):
            output_single = outputs[i]  # Shape [num_proposals, dims]
            original_h, original_w = preprocessing_meta_list[i]['original_hw']
            scale_ratio_h, scale_ratio_w, pad_top, pad_left = preprocessing_meta_list[i]['scaling_pads']

            # Filter by confidence
            if self.use_pose:
                conf_scores = output_single[:, 4]  # For pose, usually object confidence
                valid_indices = np.where(conf_scores >= conf_threshold)[0]
                class_ids = np.zeros(len(valid_indices), dtype=int)  # Assume class 0 for pose
            else:
                class_scores = output_single[:, 5:]  # Class scores start after bbox (4) and obj_conf (1)
                max_class_scores = np.max(class_scores, axis=1)
                objectness_scores = output_single[:, 4]  # Objectness score

                # Combine objectness and class scores (common in YOLOv8)
                overall_conf = objectness_scores * max_class_scores
                valid_indices = np.where(overall_conf >= conf_threshold)[0]
                class_ids = np.argmax(class_scores, axis=1)[valid_indices]

            if len(valid_indices) == 0:
                processed_results[original_indices[i]] = []
                continue

            filtered_output = output_single[valid_indices, :]
            final_conf_scores = overall_conf[valid_indices] if not self.use_pose else conf_scores[valid_indices]

            # Extract boxes (cx, cy, w, h) - model coordinates (relative to input size like 640x640)
            boxes_model_xywh = filtered_output[:, :4]

            # Convert xywh to xyxy (lx, ly, rx, ry) - still model coordinates
            boxes_model_xyxy = np.zeros_like(boxes_model_xywh)
            boxes_model_xyxy[:, 0] = boxes_model_xywh[:, 0] - boxes_model_xywh[:, 2] / 2  # lx
            boxes_model_xyxy[:, 1] = boxes_model_xywh[:, 1] - boxes_model_xywh[:, 3] / 2  # ly
            boxes_model_xyxy[:, 2] = boxes_model_xywh[:, 0] + boxes_model_xywh[:, 2] / 2  # rx
            boxes_model_xyxy[:, 3] = boxes_model_xywh[:, 1] + boxes_model_xywh[:, 3] / 2  # ry

            # Extract keypoints if pose model - model coordinates
            keypoints_model = None
            if self.use_pose and filtered_output.shape[1] > 5:  # If there are more dims than 5 (bbox+conf)
                # kpts data starts after box (4) and conf (1)
                kpts_raw = filtered_output[:, 5:].reshape(len(valid_indices), self.num_keypoints,
                                                          3)  # [N, K, 3] x, y, conf(or visibility)
                keypoints_model = kpts_raw

            # --- Apply NMS ---
            # cv2.dnn.NMSBoxes expects boxes as (x, y, w, h)
            boxes_for_nms = [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in boxes_model_xywh]

            # Ensure scores are float for NMSBoxes
            valid_conf_scores_list = final_conf_scores.astype(float).tolist()

            indices_after_nms = cv2.dnn.NMSBoxes(boxes_for_nms, valid_conf_scores_list, conf_threshold, nms_threshold)

            # Filter results after NMS
            final_detections = []
            if len(indices_after_nms) > 0:
                # Handle scalar vs list output of NMSBoxes
                if isinstance(indices_after_nms, np.ndarray):
                    indices_after_nms = indices_after_nms.flatten()

                for idx in indices_after_nms:
                    box_xyxy = boxes_model_xyxy[idx]
                    conf = final_conf_scores[idx]
                    cls_id = class_ids[idx]
                    kpts_single = keypoints_model[idx] if keypoints_model is not None else None

                    # --- Scale coordinates back to original image size ---
                    # Remove padding and scale back
                    lx = (box_xyxy[0] - pad_left) / scale_ratio_w
                    ly = (box_xyxy[1] - pad_top) / scale_ratio_h
                    rx = (box_xyxy[2] - pad_left) / scale_ratio_w
                    ry = (box_xyxy[3] - pad_top) / scale_ratio_h

                    # Clip to image bounds
                    lx_orig = max(0, int(lx))
                    ly_orig = max(0, int(ly))
                    rx_orig = min(original_w, int(rx))
                    ry_orig = min(original_h, int(ry))

                    # Check if box is valid after scaling
                    if rx_orig <= lx_orig or ry_orig <= ly_orig:
                        continue

                    if self.use_pose and kpts_single is not None:
                        pts_orig = []
                        for kpt_idx in range(self.num_keypoints):
                            kpt_x_model, kpt_y_model, kpt_conf = kpts_single[kpt_idx]
                            # Scale keypoints similar to boxes
                            kpt_x = (kpt_x_model - pad_left) / scale_ratio_w
                            kpt_y = (kpt_y_model - pad_top) / scale_ratio_h
                            # Clip keypoints
                            kpt_x_orig = max(0, min(original_w, int(kpt_x)))
                            kpt_y_orig = max(0, min(original_h, int(kpt_y)))
                            pts_orig.append(YoloPoint(x=kpt_x_orig, y=kpt_y_orig, conf=float(kpt_conf)))

                        final_detections.append(YoloPose(lx=lx_orig, ly=ly_orig, rx=rx_orig, ry=ry_orig,
                                                         cls=int(cls_id), conf=float(conf), pts=pts_orig))
                    else:  # Detection
                        final_detections.append(Yolo(lx=lx_orig, ly=ly_orig, rx=rx_orig, ry=ry_orig,
                                                     cls=int(cls_id), conf=float(conf)))

            processed_results[original_indices[i]] = final_detections

        return processed_results

    def available_results(self, index: int, conf_thresh: float = 0.25, nms_thresh: float = 0.45) -> int:
        """
        Triggers postprocessing on the results of the last inference run
        using the provided thresholds and returns the number of detections
        found for the specified image index.

        Args:
            index: The original index of the image to check results for.
            conf_thresh: Confidence threshold for filtering.
            nms_thresh: IoU threshold for NMS.

        Returns:
            The number of detected objects (Yolo/YoloPose) for the given index
            after postprocessing, or 0 if none found or error occurred.
        """
        self._last_available_index = index  # Remember which index was queried

        if self.session is None or self._raw_inference_output is None:
            print("Error: Cannot get available results. Session released or inference not run.")
            return 0

        # Perform postprocessing. We pass the thresholds to _postprocess now.
        try:
            # self._inference_meta contains 'indices', 'preprocessing_meta', 'actual_batch_size'
            self._result_map = self._postprocess(self._raw_inference_output, self._inference_meta, conf_thresh,
                                                 nms_thresh)
            # Clear raw results and inference meta after they've been post-processed and stored in _result_map
            self._raw_inference_output = None
            self._inference_meta.clear()
        except Exception as e:
            print(f"Error during postprocessing for available_results: {e}")
            traceback.print_exc()
            self._result_map = {}  # Clear results if postprocessing fails
            return 0

        # Return the count for the requested index
        count = len(self._result_map.get(index, []))
        return count

    def get_result(self, item_index: int) -> Optional[Union[Yolo, YoloPose]]:
        """
        Retrieves a specific detection result by its index within the list
        of results obtained for the image index last queried by available_results().

        Args:
            item_index: The 0-based index of the detection within the results list
                        for the last queried image.

        Returns:
            The Yolo or YoloPose object for the specified detection, or None if
            the index is out of bounds or results are unavailable.
        """
        if self.session is None:
            print("Error: Session has been released.")
            return None
        if self._last_available_index is None:
            print("Error: available_results() must be called before get_result().")
            return None

        # Get the list of results for the last queried image index
        results_list = self._result_map.get(self._last_available_index)

        if results_list is not None and 0 <= item_index < len(results_list):
            return results_list[item_index]
        else:
            return None

    def release(self):
        """Releases resources and clears internal state."""
        super().release()
        self._result_map.clear()
        self._last_available_index = None

