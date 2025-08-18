# deepsort.py (Cascaded Matching Version)
from typing import List, Dict, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from numba import njit

# Assuming these are in your project structure
from pyengine.inference.unified_structs.inference_results import ObjectDetection, Rect
from pyengine.algorithms.tracker.tracker import UnifiedTrack


# --- HELPER FUNCTION 1: IoU Cost Calculation (from SORT) ---
@njit(fastmath=True, cache=True)
def iou_batch_numba(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """
    Computes IoU between two sets of boxes (vectorized and Numba-compiled).
    From sort.py.
    """
    iou_matrix = np.empty((bb_test.shape[0], bb_gt.shape[0]))
    for i in range(bb_test.shape[0]):
        for j in range(bb_gt.shape[0]):
            box_test = bb_test[i]
            box_gt = bb_gt[j]

            xx1 = max(box_test[0], box_gt[0])
            yy1 = max(box_test[1], box_gt[1])
            xx2 = min(box_test[2], box_gt[2])
            yy2 = min(box_test[3], box_gt[3])

            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)

            intersection = w * h
            area_test = (box_test[2] - box_test[0]) * (box_test[3] - box_test[1])
            area_gt = (box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1])
            union = area_test + area_gt - intersection

            iou_matrix[i, j] = intersection / (union + 1e-6)

    return iou_matrix


# --- HELPER FUNCTION 2: DeepSORT Cost Calculation (from your original deepsort.py) ---
@njit(fastmath=True, cache=True)
def build_deepsort_cost_matrix(
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
    From your original deepsort.py.
    """
    num_tracks = track_boxes.shape[0]
    num_dets = detection_boxes.shape[0]
    cost_matrix = np.full((num_tracks, num_dets), 1e8)

    for i in range(num_tracks):
        track_box = track_boxes[i]
        track_feature = track_features[i]

        for j in range(num_dets):
            det_box = detection_boxes[j]
            det_feature = detection_features[j]

            # IoU Cost Calculation
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
                iou = intersection_area / (union_area + 1e-6)
                iou_cost = 1.0 - iou

            # This is the cost gating, not a threshold on the IoU value itself
            if iou_cost > iou_threshold:
                continue

            # Cosine Distance (Re-ID Cost) Calculation
            reid_cost = 1.0
            if track_feature.shape[0] > 1:
                dot_product = np.dot(track_feature, det_feature)
                norm_track = np.linalg.norm(track_feature)
                norm_det = np.linalg.norm(det_feature)
                if norm_track > 0 and norm_det > 0:
                    reid_cost = 1.0 - (dot_product / (norm_track * norm_det))

            if reid_cost < reid_threshold:
                cost_matrix[i, j] = lambda_weight * iou_cost + (1.0 - lambda_weight) * reid_cost

    return cost_matrix


class DeepSORTTracker:
    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.3,  # IoU threshold for the first, fast matching cascade
                 reid_iou_threshold: float = 0.5,  # IoU threshold for the second, Re-ID matching cascade
                 reid_threshold: float = 0.4,  # Re-ID feature distance threshold
                 lambda_weight: float = 0.5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.reid_iou_threshold = reid_iou_threshold
        self.reid_threshold = reid_threshold
        self.lambda_weight = lambda_weight
        self.tracks: List[UnifiedTrack] = []
        UnifiedTrack._next_id = 0

    def _associate(self, detections: List[ObjectDetection], tracks: List[UnifiedTrack], use_reid: bool) -> Tuple[
        List[Tuple[int, int]], List[int], List[int]]:
        """Helper function to perform association."""
        if not tracks or not detections:
            return [], list(range(len(detections))), list(range(len(tracks)))

        det_boxes = np.array([[d.rect.x1, d.rect.y1, d.rect.x2, d.rect.y2] for d in detections])
        trk_boxes = np.array([[t.get_state().x1, t.get_state().y1, t.get_state().x2, t.get_state().y2] for t in tracks])

        if use_reid:
            # --- Second Cascade: DeepSORT association ---
            det_features = np.array([d.features for d in detections])
            trk_features = np.array([t.get_mean_feature() for t in tracks])
            cost_matrix = build_deepsort_cost_matrix(
                trk_boxes, det_boxes, trk_features, det_features,
                self.reid_iou_threshold, self.reid_threshold, self.lambda_weight
            )
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Filter matches based on cost
            matches = [(c, r) for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] < 1e8]
        else:
            # --- First Cascade: IoU-only association ---
            iou_matrix = iou_batch_numba(det_boxes, trk_boxes)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            # Filter matches based on IoU threshold
            matches = [(r, c) for r, c in zip(row_ind, col_ind) if iou_matrix[r, c] >= self.iou_threshold]

        # Find unmatched detections and tracks
        matched_dets = {m[0] for m in matches}
        matched_trks = {m[1] for m in matches}
        unmatched_dets = [i for i, _ in enumerate(detections) if i not in matched_dets]
        unmatched_trks = [i for i, _ in enumerate(tracks) if i not in matched_trks]

        return matches, unmatched_dets, unmatched_trks

    def update(self, detections: List[ObjectDetection]) -> Dict[int, Rect]:
        """
        Performs the cascaded matching update.
        """
        # --- 1. Predict new state for all existing tracks ---
        for t in self.tracks:
            t.predict()

        # --- 2. Partition tracks into recent and old ---
        # A track is "recent" if it was updated in the previous frame (time_since_update is 1 after predict())
        recent_tracks = [t for t in self.tracks if t.time_since_update == 1]
        old_tracks = [t for t in self.tracks if t.time_since_update > 1]

        # --- 3. First Cascade: Match recent tracks with detections using IoU ---
        matches1, unmatched_dets1, unmatched_trks1 = self._associate(detections, recent_tracks, use_reid=False)
        for det_idx, trk_idx in matches1:
            recent_tracks[trk_idx].update(detections[det_idx])

        # --- 4. Second Cascade: Match remaining detections with old tracks using Re-ID ---
        # Get the actual detection objects that were not matched in the first cascade
        unmatched_detections = [detections[i] for i in unmatched_dets1]
        matches2, unmatched_dets2, unmatched_trks2 = self._associate(unmatched_detections, old_tracks, use_reid=True)
        for det_idx, trk_idx in matches2:
            # The det_idx here is for the `unmatched_detections` list, so we need to find the original index
            original_det_idx = unmatched_dets1[det_idx]
            old_tracks[trk_idx].update(detections[original_det_idx])

        # --- 5. Create new tracks for final unmatched detections ---
        for det_idx in unmatched_dets2:
            original_det_idx = unmatched_dets1[det_idx]
            if detections[original_det_idx].confidence >= 0.5:
                self.tracks.append(UnifiedTrack(detections[original_det_idx], use_reid=True))

        # --- 6. Cleanup: Remove dead tracks ---
        self.tracks = [t for t in self.tracks if not t.is_deleted(self.max_age)]

        # --- 7. Prepare final output ---
        final_tracked_objects = {}
        for track in self.tracks:
            if track.is_confirmed(self.min_hits) and track.time_since_update == 0:
                final_tracked_objects[track.track_id] = track.get_state()

        return final_tracked_objects