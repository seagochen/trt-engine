# sort.py
from typing import List, Dict, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from numba import njit # <<< OPTIMIZATION: Import Numba

from pyengine.inference.unified_structs.inference_results import ObjectDetection, Rect
from pyengine.algorithms.tracker.tracker import UnifiedTrack


# <<< OPTIMIZATION: Compile this function to fast machine code
@njit(fastmath=True, cache=True)
def iou_batch_numba(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """
    Computes IoU between two sets of boxes (vectorized and Numba-compiled).
    """
    # Numba prefers explicit loops for clarity and optimization potential
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


class SORTTracker:
    def __init__(self, max_age: int = 5, min_hits: int = 1, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[UnifiedTrack] = []
        UnifiedTrack._next_id = 0

    def update(self, detections: List[ObjectDetection]) -> Dict[int, Rect]:
        # 预测所有 track 的新状态
        predicted_boxes = []
        for track in self.tracks:
            track.predict()
            rect = track.get_state()
            # 验证预测的边界框是否有效
            if rect.x2 > rect.x1 and rect.y2 > rect.y1:
                predicted_boxes.append([rect.x1, rect.y1, rect.x2, rect.y2])
            else:
                # 如果预测的边界框无效，使用零尺寸占位（会导致 IoU=0，不会被匹配）
                predicted_boxes.append([0, 0, 0, 0])

        # 转换为 numpy 数组（需要处理空列表的情况）
        if predicted_boxes:
            predicted_boxes = np.array(predicted_boxes)
        else:
            predicted_boxes = np.empty((0, 4))

        # 提取检测框
        if detections:
            detection_boxes = np.array([[d.rect.x1, d.rect.y1, d.rect.x2, d.rect.y2] for d in detections])
        else:
            detection_boxes = np.empty((0, 4))

        # 执行匹配
        matches, _, _ = self._associate(detection_boxes, predicted_boxes)

        # 更新匹配的 tracks
        for det_idx, trk_idx in matches:
            self.tracks[trk_idx].update(detections[det_idx])

        # 找出未匹配的检测
        matched_det_indices = set(matches[:, 0]) if len(matches) > 0 else set()
        unmatched_dets = [i for i, _ in enumerate(detections) if i not in matched_det_indices]

        # 为未匹配的检测创建新的 tracks
        for det_idx in unmatched_dets:
            if detections[det_idx].confidence >= 0.5:
                self.tracks.append(UnifiedTrack(detections[det_idx], use_reid=False))

        # 删除过时的 tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted(self.max_age)]

        # 返回确认的 tracks
        final_tracked_objects = {}
        for track in self.tracks:
            if track.is_confirmed(self.min_hits) and track.time_since_update == 0:
                 final_tracked_objects[track.track_id] = track.get_state()

        return final_tracked_objects

    def _associate(self, det_boxes: np.ndarray, trk_boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if trk_boxes.shape[0] == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(det_boxes)), np.empty(0, dtype=int)
        
        if det_boxes.shape[0] == 0:
            return np.empty((0, 2), dtype=int), np.empty(0, dtype=int), np.arange(len(trk_boxes))

        # <<< OPTIMIZATION: Call the Numba-compiled function
        iou_matrix = iou_batch_numba(det_boxes, trk_boxes)
        
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(row_ind, col_ind)))

        matches = []
        if matched_indices.size > 0:
            for m in matched_indices:
                if iou_matrix[m[0], m[1]] >= self.iou_threshold:
                    matches.append(m.reshape(1, 2))
        
        matches = np.concatenate(matches, axis=0) if len(matches) > 0 else np.empty((0, 2), dtype=int)

        all_det_indices = set(range(len(det_boxes)))
        matched_det_indices = set(matches[:, 0]) if matches.size > 0 else set()
        unmatched_dets = np.array(list(all_det_indices - matched_det_indices))

        all_trk_indices = set(range(len(trk_boxes)))
        matched_trk_indices = set(matches[:, 1]) if matches.size > 0 else set()
        unmatched_trks = np.array(list(all_trk_indices - matched_trk_indices))

        return matches, unmatched_dets, unmatched_trks