# deepsort.py
from typing import List, Dict

import numpy as np
from scipy.optimize import linear_sum_assignment
from numba import njit

from pyengine.inference.unified_structs.inference_results import ObjectDetection, Rect
from pyengine.algorithms.tracker.tracker import UnifiedTrack


@njit(fastmath=True, cache=True)
def build_cost_matrix(
    num_tracks: int,
    num_dets: int,
    track_boxes: np.ndarray,
    detection_boxes: np.ndarray,
    track_features: np.ndarray,
    detection_features: np.ndarray,
    iou_threshold: float,
    reid_threshold: float,
    lambda_weight: float
) -> np.ndarray:
    """
    Builds the combined cost matrix using Numba for high performance.
    """
    cost_matrix = np.full((num_tracks, num_dets), 1e8)

    for i in range(num_tracks):
        track_box = track_boxes[i]
        track_feature = track_features[i]

        for j in range(num_dets):
            det_box = detection_boxes[j]
            det_feature = detection_features[j]

            # --- IoU Cost Calculation (inlined) ---
            x_left = max(track_box[0], det_box[0])
            y_top = max(track_box[1], det_box[1])
            x_right = min(track_box[2], det_box[2])
            y_bottom = min(track_box[3], det_box[3])
            
            iou_cost = 1.0
            if x_right >= x_left and y_bottom >= y_top:
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                track_area = (track_box[2] - track_box[0]) * (track_box[3] - track_box[1])
                det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                union_area = track_area + det_area - intersection_area
                iou_cost = 1.0 - (intersection_area / (union_area + 1e-6))
            
            # --- Cosine Distance (Re-ID Cost) Calculation (inlined) ---
            dot_product = np.dot(track_feature, det_feature)
            norm_track = np.linalg.norm(track_feature)
            norm_det = np.linalg.norm(det_feature)
            
            reid_cost = 1.0
            if norm_track > 0 and norm_det > 0:
                reid_cost = 1.0 - (dot_product / (norm_track * norm_det))

            # --- Gating and Combined Cost ---
            if iou_cost < iou_threshold and reid_cost < reid_threshold:
                cost_matrix[i, j] = lambda_weight * iou_cost + (1.0 - lambda_weight) * reid_cost
    
    return cost_matrix


class DeepSORTTracker:
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.5, reid_threshold: float = 0.4, lambda_weight: float = 0.5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.reid_threshold = reid_threshold
        self.lambda_weight = lambda_weight
        self.tracks: List[UnifiedTrack] = []
        UnifiedTrack._next_id = 0

    def update(self, detections: List[ObjectDetection]) -> Dict[int, Rect]:
        for track in self.tracks:
            track.predict()

        num_tracks = len(self.tracks)
        num_dets = len(detections)
        
        if num_dets == 0:
            self.tracks = [t for t in self.tracks if not t.is_deleted(self.max_age)]
            return {}

        # --- FIX: Manually create arrays from Rect attributes ---
        if num_tracks > 0:
            track_boxes = np.array([[t.get_state().x1, t.get_state().y1, t.get_state().x2, t.get_state().y2] for t in self.tracks])
        else:
            track_boxes = np.empty((0, 4))
        
        detection_boxes = np.array([[d.rect.x1, d.rect.y1, d.rect.x2, d.rect.y2] for d in detections])
        # --- END FIX ---
        
        track_features = np.array([t.get_mean_feature() if t.get_mean_feature() is not None else np.zeros(1) for t in self.tracks]) if num_tracks > 0 else np.empty((0,1))
        detection_features = np.array([d.features if d.features else np.zeros(1) for d in detections])

        cost_matrix = build_cost_matrix(
            num_tracks, num_dets, track_boxes, detection_boxes, track_features, detection_features,
            self.iou_threshold, self.reid_threshold, self.lambda_weight
        )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_dets = set(range(num_dets))
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e8:
                matches.append((c, r))
                unmatched_dets.discard(c)

        for det_idx, trk_idx in matches:
            self.tracks[trk_idx].update(detections[det_idx])

        for det_idx in unmatched_dets:
            if detections[det_idx].confidence >= 0.5:
                self.tracks.append(UnifiedTrack(detections[det_idx], use_reid=True))

        self.tracks = [t for t in self.tracks if not t.is_deleted(self.max_age)]
        
        final_tracked_objects = {}
        for track in self.tracks:
            if track.is_confirmed(self.min_hits) and track.time_since_update == 0:
                final_tracked_objects[track.track_id] = track.get_state()
        
        return final_tracked_objects