# sort.py
from typing import List, Dict

import numpy as np
from scipy.optimize import linear_sum_assignment

# 从你的数据结构模块导入 ObjectDetection 和 Rect
from pyengine.inference.unified_structs.inference_results import ObjectDetection, Rect
from pyengine.algorithms.tracker.tracker import UnifiedTrack
from pyengine.utils.logger import logger


class SORTTracker:
    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        初始化 SORT 追踪器。
        Args:
            max_age (int): 轨迹在未更新后被删除的最大帧数。
                           SORT通常设置为1或2，因为不使用Re-ID，长时间丢失意味着真正丢失。
            min_hits (int): 新轨迹在被确认为有效轨迹前的最小命中次数。
            iou_threshold (float): 用于匹配的 IoU 阈值。
        """
        self.tracks: List[UnifiedTrack] = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.use_reid = False

    def _calculate_iou(self, bbox1: Rect, bbox2: Rect) -> float:
        """
        计算两个边界框的 IoU。
        Args:
            bbox1 (Rect): 第一个边界框。
            bbox2 (Rect): 第二个边界框。
        Returns:
            float: IoU 值。
        """
        # Rect 已经是 (x1, y1, x2, y2) 形式，直接访问属性
        bb1 = [bbox1.x1, bbox1.y1, bbox1.x2, bbox1.y2]
        bb2 = [bbox2.x1, bbox2.y1, bbox2.x2, bbox2.y2]

        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No overlap

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

        union_area = bb1_area + bb2_area - intersection_area
        if union_area == 0:
            return 0.0  # Avoid division by zero

        iou = intersection_area / union_area
        return iou

    def update(self, detections: List[ObjectDetection]) -> Dict[int, Rect]:
        """
        更新追踪器状态，处理当前帧的检测结果。
        Args:
            detections (List[ObjectDetection]): 当前帧的检测结果列表。
        Returns:
            Dict[int, Rect]: 字典，键是轨迹ID，值是对应轨迹的边界框。
                             只包含被确认的或未达到min_hits但还在追踪中的轨迹。
        """
        # 1. 预测现有轨迹状态
        for track in self.tracks:
            track.predict()

        num_tracks = len(self.tracks)
        num_dets = len(detections)

        # 初始处理空轨迹或空检测的情况
        if num_tracks == 0 and num_dets == 0:
            logger.debug("SORTTracker", "No tracks and no detections. Returning empty.")
            return {}

        if num_tracks == 0 and num_dets > 0:
            logger.info("SORTTracker", f"No active tracks. Initializing {num_dets} new tracks.")
            for det_idx in range(num_dets):
                if detections[det_idx].confidence >= 0.5:
                    new_track = UnifiedTrack(detections[det_idx], use_reid=self.use_reid)
                    self.tracks.append(new_track)
            return {track.track_id: track.get_state() for track in self.tracks if
                    track.is_confirmed(self.min_hits) or (track.time_since_update == 0 and track.hits == 1)}

        if num_tracks > 0 and num_dets == 0:
            logger.info("SORTTracker", f"No detections. Predicting tracks and deleting old ones.")
            for track_idx in range(num_tracks):
                self.tracks[track_idx].time_since_update += 1
            self.tracks = [track for track in self.tracks if not track.is_deleted(self.max_age)]
            return {track.track_id: track.get_state() for track in self.tracks if
                    track.is_confirmed(self.min_hits) or (track.time_since_update == 0 and track.hits == 1)}

        # 成本矩阵 (cost = 1 - IoU)，如果 IoU 低于阈值，则成本为无穷大
        cost_matrix = np.full((num_tracks, num_dets), np.inf)

        for i, track in enumerate(self.tracks):
            track_predicted_bbox = track.get_state()

            for j, det in enumerate(detections):
                det_bbox_rect = det.rect
                iou = self._calculate_iou(track_predicted_bbox, det_bbox_rect)

                if np.isnan(iou):
                    cost_matrix[i, j] = np.inf  # Treat NaN IoU as no overlap
                    logger.debug("SORTTracker", f"NaN IoU for Track {track.track_id} and Det {j}. Cost set to inf.")
                elif iou >= self.iou_threshold:
                    cost_matrix[i, j] = 1.0 - iou

        logger.debug("SORTTracker", f"Cost matrix before assignment:\n{cost_matrix}")
        logger.debug("SORTTracker", f"Shape of cost matrix: {cost_matrix.shape}")

        # --- MODIFIED: Replace inf with a large finite number for linear_sum_assignment ---
        # This is the crucial fix for "ValueError: cost matrix is infeasible"
        max_cost = np.max(cost_matrix[np.isfinite(cost_matrix)]) if np.any(np.isfinite(cost_matrix)) else 1.0
        # Replace inf with a value larger than any possible finite cost
        # A typical IOU cost is between 0 and 1. So 1000 is usually safe.
        # Or, max_cost + 1 if there's any finite value, else a default large number.
        safe_large_cost = max_cost + 1.0 if np.any(np.isfinite(cost_matrix)) else 1000.0

        processed_cost_matrix = np.where(np.isinf(cost_matrix), safe_large_cost, cost_matrix)

        # --- ADDED: Check if processed_cost_matrix is entirely problematic ---
        # If after conversion, it's still all a single large number, linear_sum_assignment
        # might still be problematic if it tries to assign every track to every detection.
        # This happens if there are no 'real' matches (i.e., all original IoUs were < threshold).
        if np.all(processed_cost_matrix == safe_large_cost):
            logger.warning("SORTTracker",
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
                    new_track = UnifiedTrack(detections[det_idx], use_reid=self.use_reid)
                    self.tracks.append(new_track)

            return {track.track_id: track.get_state() for track in self.tracks if
                    track.is_confirmed(self.min_hits) or (track.time_since_update == 0 and track.hits == 1)}
        # --- END MODIFIED ---

        # 3. 使用匈牙利算法进行匹配
        # Pass the processed_cost_matrix
        row_ind, col_ind = linear_sum_assignment(processed_cost_matrix)

        matches = []
        unmatched_tracks_indices = set(range(num_tracks))
        unmatched_dets_indices = set(range(num_dets))

        # Only consider matches where the cost is below the 'safe_large_cost'
        # This filters out the "dummy" assignments made for previously infinite costs
        for r, c in zip(row_ind, col_ind):
            if processed_cost_matrix[r, c] < safe_large_cost:
                matches.append((r, c))
                unmatched_tracks_indices.discard(r)
                unmatched_dets_indices.discard(c)
            else:  # If cost is the safe_large_cost, it means no real match was found
                unmatched_tracks_indices.add(r)
                unmatched_dets_indices.add(c)  # This might add redundant if already in set from other matches

        # Correctly determine unmatched detections from the original set
        # This handles cases where a detection might be matched to a track whose cost was infinite,
        # but we want to treat it as unmatched.
        matched_dets_indices = {c for r, c in matches}
        unmatched_dets_indices = set(range(num_dets)) - matched_dets_indices

        # 4. 更新轨迹状态
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])

        # 5. 处理未匹配的轨迹
        # For unmatched existing tracks, increment time_since_update count
        # This needs to be for all tracks in unmatched_tracks_indices
        for track_idx in unmatched_tracks_indices:
            self.tracks[track_idx].time_since_update += 1

        # 6. 删除老旧的轨迹 (超过 max_age 未更新)
        self.tracks = [track for track in self.tracks if not track.is_deleted(self.max_age)]

        # 7. 初始化新轨迹
        for det_idx in unmatched_dets_indices:
            # Only consider initializing a new track if detection confidence is above a threshold
            if detections[det_idx].confidence >= 0.5:
                new_track = UnifiedTrack(detections[det_idx], use_reid=self.use_reid)
                self.tracks.append(new_track)

        # 8. 收集并返回当前帧的有效跟踪结果
        final_tracked_objects = {}
        for track in self.tracks:
            # SORT only outputs a track if it has been "hit" enough times (min_hits)
            # This prevents creating many false tracks from single erroneous detections
            if track.is_confirmed(self.min_hits) or (track.time_since_update == 0 and track.hits == 1):
                final_tracked_objects[track.track_id] = track.get_state()

        return final_tracked_objects




"""
import sys
import time
import traceback

import cv2
import os  # Import os for path manipulation

from pyengine.algorithms.tracker.sort import SORTTracker
from pyengine.visualization.inference_drawer import GenericInferenceDrawer
from pyengine.inference.unified_structs.pipeline_converter import convert_pipeline_v1_to_skeletons
from pyengine.inference.c_pipeline.pose_pipeline_v1 import PosePipeline
from pyengine.utils.logger import logger


if __name__ == "__main__":
    # Define your paths
    LIBRARY_PATH = "/home/user/projects/TrtEngineToolkits/build/lib/libjetson.so"
    YOLO_POSE_ENGINE = "/opt/models/yolov8n-pose.engine"
    EFFICIENTNET_ENGINE = "/opt/models/efficientnet_b0_feat_logits.engine"  # SORT doesn't strictly need features, but pipeline provides them
    SCHEMA_FILE = "./configs/schema.json"  # Your JSON schema file path
    VIDEO_PATH = "/opt/videos/raining_street_02.mp4"  # Test video path

    pipeline = None
    tracker = None
    drawer = None
    cap = None  # VideoCapture object

    try:
        # Initialize C++ pipeline
        pipeline = PosePipeline(LIBRARY_PATH)
        pipeline.register_models()
        # Set yolo_max_batch to 1 for single video stream
        pipeline.create_pipeline(
            yolo_engine_path=YOLO_POSE_ENGINE,
            efficient_engine_path=EFFICIENTNET_ENGINE,
            yolo_max_batch=1,  # Adjusted for single video stream
            efficient_max_batch=32,  # EfficientNet batch size can remain larger if needed
            yolo_cls_thresh=0.5,
            yolo_iou_thresh=0.4
        )
        logger.info("Main", "C++ PosePipeline initialized.")

        # Initialize SORT Tracker
        tracker = SORTTracker(max_age=1, min_hits=3, iou_threshold=0.3)
        logger.info("Main", "SORTTracker initialized.")

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

        # --- REMOVED: classification_labels is no longer needed ---
        # classification_labels = [...]

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

            # Update SORT tracker
            tracked_objects = tracker.update(current_frame_skeletons)

            # --- Visualization ---
            display_frame = frame.copy()
            original_shape = (frame_height, frame_width)

            # Map track_id back to Skeleton objects for drawing
            for skeleton_det in current_frame_skeletons:
                matched_track_id = -1
                # Iterate through tracked_objects to find a matching Rect (by coordinates)
                epsilon = 64.0  # Pixel tolerance
                for track_id, tracked_rect in tracked_objects.items():
                    if (abs(skeleton_det.rect.x1 - tracked_rect.x1) < epsilon and
                            abs(skeleton_det.rect.y1 - tracked_rect.y1) < epsilon and
                            abs(skeleton_det.rect.x2 - tracked_rect.x2) < epsilon and
                            abs(skeleton_det.rect.y2 - tracked_rect.y2) < epsilon):
                        matched_track_id = track_id
                        break

                skeleton_det.track_id = matched_track_id  # Assign the detected track_id

            # Draw all skeletons with their assigned track_ids
            display_frame = drawer.draw_skeletons_batch(
                display_frame,
                current_frame_skeletons,
                original_shape,
                enable_track_id=True,  # Now display track IDs
                label_names=None,
                # Passed as None, so drawer will use classification ID directly or 'person' for skeletons
                enable_pts_names=False,
                enable_skeleton=True
            )

            # Display FPS
            elapsed_time = time.time() - start_time
            current_fps = frame_idx / elapsed_time
            cv2.putText(display_frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("SORT Tracking Demo", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Main", "Quit key 'q' pressed. Exiting.")
                break

    except Exception as e:
        logger.error_trace("Main", f"An error occurred: {e}")

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