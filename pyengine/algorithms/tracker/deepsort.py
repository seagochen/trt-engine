# deepsort.py
from typing import List, Dict

import numpy as np
from scipy.optimize import linear_sum_assignment

# 从你的文件导入数据结构
from pyengine.inference.unified_structs.inference_results import ObjectDetection, Rect
from pyengine.algorithms.tracker.tracker import UnifiedTrack
# 导入 logger
from pyengine.utils.logger import logger


class DeepSORTTracker:
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.5, reid_threshold: float = 0.4):
        self.tracks: List[UnifiedTrack] = []
        self.max_age = max_age  # Max frames for a track to be unmatched before deletion
        self.min_hits = min_hits  # Min hits for a new track to be confirmed
        self.iou_threshold = iou_threshold
        self.reid_threshold = reid_threshold  # Cosine similarity threshold for re-ID

    def _iou_cost(self, track_bbox: Rect, det_bbox: Rect) -> float:
        # Calculate IoU distance (1 - IoU)

        # Convert Rect to (x1, y1, x2, y2)
        bb1 = [track_bbox.x1, track_bbox.y1, track_bbox.x2, track_bbox.y2]
        bb2 = [det_bbox.x1, det_bbox.y1, det_bbox.x2, det_bbox.y2]

        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])

        if x_right < x_left or y_bottom < y_top:
            return 1.0  # No overlap, distance is 1 (max)

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

        union_area = bb1_area + bb2_area - intersection_area
        if union_area == 0:
            return 0.0  # Avoid division by zero

        iou = intersection_area / float(union_area)  # Ensure float division
        return 1.0 - iou  # Return 1 - IoU as distance/cost

    def _cosine_distance(self, track_feature: np.ndarray, det_feature: np.ndarray) -> float:
        # Calculate cosine distance (1 - cosine_similarity)
        # Ensure features are normalized if not already
        # Add small epsilon to avoid division by zero if norm is 0
        norm_track = np.linalg.norm(track_feature)
        norm_det = np.linalg.norm(det_feature)

        if norm_track == 0 or norm_det == 0:
            logger.warning("DeepSORTTracker", "Zero norm feature detected, returning max cosine distance.")
            return 1.0  # Max distance if either feature is zero vector

        track_feature = track_feature / norm_track
        det_feature = det_feature / norm_det

        similarity = np.dot(track_feature, det_feature)
        return 1.0 - similarity  # Return 1 - similarity as distance/cost

    def update(self, detections: List[ObjectDetection]) -> Dict[int, Rect]:
        # 1. Predict
        for track in self.tracks:
            track.predict()

        num_tracks = len(self.tracks)
        num_dets = len(detections)

        # --- MODIFIED: Robust handling for empty tracks or detections ---
        if num_tracks == 0 and num_dets == 0:
            logger.debug("DeepSORTTracker", "No tracks and no detections. Returning empty.")
            return {}

        if num_tracks == 0 and num_dets > 0:
            logger.info("DeepSORTTracker", f"No active tracks. Initializing {num_dets} new tracks.")
            for det_idx in range(num_dets):
                if detections[det_idx].confidence >= 0.5:  # Use detection confidence
                    new_track = UnifiedTrack(detections[det_idx], use_reid=True)  # DeepSORT always uses re-ID
                    self.tracks.append(new_track)
            return {track.track_id: track.get_state() for track in self.tracks if
                    track.is_confirmed(self.min_hits) or (track.time_since_update == 0 and track.hits == 1)}

        if num_tracks > 0 and num_dets == 0:
            logger.info("DeepSORTTracker", f"No detections. Predicting tracks and deleting old ones.")
            for track_idx in range(num_tracks):
                self.tracks[track_idx].time_since_update += 1
            self.tracks = [track for track in self.tracks if not track.is_deleted(self.max_age)]
            return {track.track_id: track.get_state() for track in self.tracks if
                    track.is_confirmed(self.min_hits) or (track.time_since_update == 0 and track.hits == 1)}
        # --- END MODIFIED ---

        # 3. Create cost matrix
        cost_matrix = np.full((num_tracks, num_dets), np.inf)

        for i, track in enumerate(self.tracks):
            track_predicted_bbox = track.get_state()
            track_feature = track.get_feature()  # This might be None if track was created without features (e.g. from SORT) or feature deque is empty

            for j, det in enumerate(detections):
                det_bbox_rect = det.rect
                iou_cost = self._iou_cost(track_predicted_bbox, det_bbox_rect)

                reid_cost = np.inf
                # DeepSORT: Only calculate Re-ID cost if both track and detection have valid features
                if track_feature is not None and det.features and len(det.features) > 0:
                    try:
                        # Ensure det.features is treated as numpy array for _cosine_distance
                        reid_cost = self._cosine_distance(track_feature, np.array(det.features))
                    except Exception as e:
                        logger.warning("DeepSORTTracker",
                                       f"Error calculating re-ID cost: {e}. Skipping re-ID for this pair.")
                        reid_cost = np.inf  # If error, treat as no re-ID match

                # Combine costs (DeepSORT's typical combination)
                # Apply thresholds for initial gating before combining
                # If either cost is beyond its threshold, then it's not a compatible match
                is_iou_compatible = iou_cost <= (1.0 - self.iou_threshold)
                is_reid_compatible = reid_cost <= (1.0 - self.reid_threshold)

                # Assign cost only if both components are compatible.
                # A more sophisticated approach would be cascading, but for a combined cost, this works.
                if is_iou_compatible and is_reid_compatible:
                    # Example weighted sum. You might tune weights (e.g., 0.5, 0.5)
                    # Mahalanobis distance would be combined here too in full DeepSORT.
                    cost_matrix[i, j] = 0.5 * iou_cost + 0.5 * reid_cost
                elif is_iou_compatible:  # If only IOU is compatible, it might still be a weak match for initial stages
                    # For simplicity, we make it compatible only if both are good for combined cost
                    # A robust DeepSORT uses cascading, where IOU is primary for short-term, then Re-ID.
                    cost_matrix[i, j] = np.inf  # If not fully compatible, don't consider for combined cost
                else:
                    cost_matrix[i, j] = np.inf  # Not compatible

        logger.debug("DeepSORTTracker", f"Cost matrix before assignment:\n{cost_matrix}")
        logger.debug("DeepSORTTracker", f"Shape of cost matrix: {cost_matrix.shape}")

        # --- MODIFIED: Replace inf with a large finite number for linear_sum_assignment ---
        # This is the crucial fix for "ValueError: cost matrix is infeasible"
        # Find the maximum finite cost in the matrix. If no finite values, use a default.
        max_cost = np.max(cost_matrix[np.isfinite(cost_matrix)]) if np.any(np.isfinite(cost_matrix)) else 1.0
        safe_large_cost = max_cost + 1.0 if np.any(
            np.isfinite(cost_matrix)) else 1000.0  # Use a value guaranteed to be larger

        processed_cost_matrix = np.where(np.isinf(cost_matrix), safe_large_cost, cost_matrix)

        # Also ensure no NaNs (from previous calculation errors) before assignment
        if np.any(np.isnan(processed_cost_matrix)):
            logger.warning("DeepSORTTracker",
                           "Processed cost matrix contains NaN values. Replacing with safe large cost.")
            processed_cost_matrix[np.isnan(processed_cost_matrix)] = safe_large_cost

        # --- ADDED: Check if processed_cost_matrix is entirely problematic ---
        if np.all(processed_cost_matrix == safe_large_cost):
            logger.warning("DeepSORTTracker",
                           "Processed cost matrix is entirely large finite values (effectively infeasible). No 'real' matches found.")
            # Treat all tracks as unmatched, and all detections as new.
            unmatched_tracks_indices = set(range(num_tracks))
            unmatched_dets_indices = set(range(num_dets))

            # Handle unmatched tracks (increment time_since_update and delete old ones)
            for track_idx in unmatched_tracks_indices:
                self.tracks[track_idx].time_since_update += 1
            self.tracks = [track for track in self.tracks if not track.is_deleted(self.max_age)]

            # Initialize new tracks
            for det_idx in unmatched_dets_indices:
                if detections[det_idx].confidence >= 0.5:
                    new_track = UnifiedTrack(detections[det_idx], use_reid=True)
                    self.tracks.append(new_track)

            return {track.track_id: track.get_state() for track in self.tracks if
                    track.is_confirmed(self.min_hits) or (track.time_since_update == 0 and track.hits == 1)}
        # --- END MODIFIED ---

        # 4. Solve Assignment (Hungarian Algorithm)
        row_ind, col_ind = linear_sum_assignment(processed_cost_matrix)  # Use the processed matrix

        matches = []
        unmatched_tracks_indices = set(range(num_tracks))
        unmatched_dets_indices = set(range(num_dets))

        # Filter out "dummy" assignments where cost was safe_large_cost
        for r, c in zip(row_ind, col_ind):
            if processed_cost_matrix[r, c] < safe_large_cost:  # Only consider real matches
                matches.append((r, c))
                unmatched_tracks_indices.discard(r)
                unmatched_dets_indices.discard(c)
            else:  # This track and detection were "matched" to a dummy cost, meaning no real match
                unmatched_tracks_indices.add(r)  # Mark track as truly unmatched
                unmatched_dets_indices.add(c)  # Mark detection as truly unmatched

        # Recalculate unmatched detections based on actual matches
        matched_dets_indices = {c for r, c in matches}
        unmatched_dets_indices = set(range(num_dets)) - matched_dets_indices

        # Recalculate unmatched tracks based on actual matches
        matched_tracks_indices = {r for r, c in matches}
        unmatched_tracks_indices = set(range(num_tracks)) - matched_tracks_indices

        # 5. Update Tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])

        # Handle unmatched tracks (increment time_since_update)
        for track_idx in unmatched_tracks_indices:
            self.tracks[track_idx].time_since_update += 1

        # Remove old, unmatched tracks
        self.tracks = [track for track in self.tracks if
                       track.is_deleted(self.max_age) == False]  # Ensure track deletion logic is correct

        # 6. Initialize New Tracks
        for det_idx in unmatched_dets_indices:
            if detections[det_idx].confidence >= 0.5:
                new_track = UnifiedTrack(detections[det_idx], use_reid=True)
                self.tracks.append(new_track)

        # Filter out "tentative" new tracks that haven't been seen enough times
        final_tracks_output = {}
        for track in self.tracks:
            # DeepSORT outputs confirmed tracks, or tentative tracks that were just updated (hits==1)
            if track.is_confirmed(self.min_hits) or (track.time_since_update == 0 and track.hits == 1):
                final_tracks_output[track.track_id] = track.get_state()

        return final_tracks_output


"""
import sys
import time
import traceback # Keep traceback for comprehensive error logging

import cv2
import os

from pyengine.algorithms.tracker.deepsort import DeepSORTTracker # Changed from SORTTracker
from pyengine.visualization.inference_drawer import GenericInferenceDrawer
from pyengine.inference.unified_structs.pipeline_converter import convert_pipeline_v1_to_skeletons
from pyengine.inference.c_pipeline.pose_pipeline_v1 import PosePipeline
from pyengine.utils.logger import logger
from pyengine.inference.unified_structs.inference_results import Skeleton, ObjectDetection # Explicitly import needed classes


if __name__ == "__main__":
    # Define your paths
    LIBRARY_PATH = "/home/user/projects/TrtEngineToolkits/build/lib/libjetson.so"
    YOLO_POSE_ENGINE = "/opt/models/yolov8n-pose.engine"
    EFFICIENTNET_ENGINE = "/opt/models/efficientnet_b0_feat_logits.engine" # DeepSORT *needs* features
    SCHEMA_FILE = "./configs/schema.json"
    VIDEO_PATH = "/opt/videos/raining_street_02.mp4"

    pipeline = None
    tracker = None
    drawer = None
    cap = None

    try:
        # Initialize C++ pipeline
        pipeline = PosePipeline(LIBRARY_PATH)
        pipeline.register_models()
        pipeline.create_pipeline(
            yolo_engine_path=YOLO_POSE_ENGINE,
            efficient_engine_path=EFFICIENTNET_ENGINE,
            yolo_max_batch=1, # Still 1 for single video stream
            efficient_max_batch=32, # EfficientNet batch size can remain larger
            yolo_cls_thresh=0.5,
            yolo_iou_thresh=0.2
        )
        logger.info("Main", "C++ PosePipeline initialized.")

        # --- MODIFIED: Initialize DeepSORT Tracker ---
        # DeepSORT has additional parameters: iou_threshold and reid_threshold
        # You'll likely need to tune these for optimal performance.
        # min_hits and max_age are usually different for DeepSORT compared to SORT.
        # max_age is typically higher (e.g., 30-70) as Re-ID can bridge larger gaps.
        # min_hits is also typically higher (e.g., 3-5) for more robust track confirmation.
        # However, for a quick test, let's start with common DeepSORT values.
        tracker = DeepSORTTracker(max_age=70, min_hits=3, iou_threshold=0.3, reid_threshold=0.5)
        logger.info("Main", "DeepSORTTracker initialized.")
        # --- END MODIFIED ---

        # Initialize Drawer
        drawer = GenericInferenceDrawer(SCHEMA_FILE)
        logger.info("Main", f"GenericInferenceDrawer initialized with schema: {SCHEMA_FILE}")

        # Open video file
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {VIDEO_PATH}")
        logger.info("Main", f"Video stream opened: {VIDEO_PATH}")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info("Main", f"Video Resolution: {frame_width}x{frame_height}, FPS: {fps}")

        frame_idx = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Main", "End of video stream or failed to read frame.")
                break

            frame_idx += 1

            # Prepare frame for pipeline: resize to 640x640
            frame_for_pipeline = cv2.resize(frame, (640, 640))

            # Perform inference with C++ pipeline
            raw_pipeline_results = pipeline.process_batched_images([frame_for_pipeline])

            # Convert raw results to Skeleton objects
            current_frame_skeletons = []
            if raw_pipeline_results:
                all_skeletons_from_pipeline = convert_pipeline_v1_to_skeletons(raw_pipeline_results)
                if all_skeletons_from_pipeline:
                    current_frame_skeletons = all_skeletons_from_pipeline[0]

            # Update DeepSORT tracker
            # DeepSORTTracker.update expects List[ObjectDetection] (Skeleton inherits from it)
            # and internally uses the 'features' field from ObjectDetection.
            tracked_objects = tracker.update(current_frame_skeletons)

            # --- Visualization ---
            display_frame = frame.copy()
            original_shape = (frame_height, frame_width)

            # Map track_id back to Skeleton objects for drawing
            # This logic remains the same as for SORT
            epsilon = 64.0  # Pixel tolerance, adjusted for larger movements
            for skeleton_det in current_frame_skeletons:
                matched_track_id = -1
                for track_id, tracked_rect in tracked_objects.items():
                    if (abs(skeleton_det.rect.x1 - tracked_rect.x1) < epsilon and
                            abs(skeleton_det.rect.y1 - tracked_rect.y1) < epsilon and
                            abs(skeleton_det.rect.x2 - tracked_rect.x2) < epsilon and
                            abs(skeleton_det.rect.y2 - tracked_rect.y2) < epsilon):
                        matched_track_id = track_id
                        break
                skeleton_det.track_id = matched_track_id

            # Draw all skeletons with their assigned track_ids
            display_frame = drawer.draw_skeletons_batch(
                display_frame,
                current_frame_skeletons,
                original_shape,
                enable_track_id=True,
                label_names=None, # Use classification ID directly or 'person' for skeletons
                enable_pts_names=False,
                enable_skeleton=False
            )

            # Display FPS
            elapsed_time = time.time() - start_time
            current_fps = frame_idx / elapsed_time
            cv2.putText(display_frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("DeepSORT Tracking Demo", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Main", "Quit key 'q' pressed. Exiting.")
                break

    except Exception as e:
        logger.error_trace("Main", f"An error occurred: {e}") # Ensure this is error_trace

    finally:
        # Release resources
        if cap:
            cap.release()
            logger.info("Main", "Video stream released.")
        if pipeline:
            pipeline.destroy_pipeline()
            logger.info("Main", "Pipeline destroyed.")
        cv2.destroyAllWindows()
        logger.info("Main", "Program finished.")
"""