import numpy as np
from scipy.optimize import linear_sum_assignment

from common.tracker.tracker_manager import TrackerManager



class SortTracker:
    
    def __init__(self, max_age=5, min_hits=1, iou_threshold=0.3, max_objects=1000):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.frame_count = 0  # 帧计数器
        self.tracker_manager = TrackerManager(max_age, max_objects)  # 用 TrackerManager 管理追踪器
        

    def update(self, detections: np.ndarray) -> np.array:
        self.frame_count += 1
        trks = self.tracker_manager.get_tracker_states()  # 获取当前所有追踪器的状态

        # 将检测结果与追踪器进行匹配
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(detections, trks)
        # 对于matched的结构来说：每一行是一个匹配对，包含两个值：
        # 第一列：检测结果的索引（在输入 detections 中的位置）。
        # 第二列：追踪器的索引（在 trackers 数组中的位置）。
        # 例:
        #     [0, 0],  # 第 0 个检测结果匹配第 0 个追踪器
        #     [1, 1]   # 第 1 个检测结果匹配第 1 个追踪器


        # 更新匹配上的追踪器
        detection_map = self.tracker_manager.update_trackers(matched, detections) # 今度の修正

        # 添加未匹配的追踪器
        unmatched_detection_map = self.tracker_manager.add_trackers(unmatched_dets, detections)
        detection_map.update(unmatched_detection_map)        

        # 更新追踪器状态
        self.tracker_manager.commit_changes()

        # 构造返回的结果
        ret = []
        for tracker in self.tracker_manager.trackers:
            bbox = tracker.get_state()
            detection_idx = detection_map.get(tracker.id)
            if detection_idx is not None and isinstance(detection_idx.item(), int):
                kpts_combined = detections[detection_idx][4:]  # 获取检测的关键点信息
                ret.append([tracker.id] + bbox + kpts_combined.tolist())
        return np.array(ret)        


    def _associate_detections_to_trackers(self, detections, trackers):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det[:4], trk[:4])

        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(*matched_indices)))

        if matched_indices.size == 0:
            matched_indices = np.empty((0, 2), dtype=int)

        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


    @staticmethod
    def _iou(bb_test, bb_gt):
        xx1 = max(bb_test[0], bb_gt[0])
        yy1 = max(bb_test[1], bb_gt[1])
        xx2 = min(bb_test[2], bb_gt[2])
        yy2 = min(bb_test[3], bb_gt[3])

        union_w = max(0., xx2 - xx1)
        union_h = max(0., yy2 - yy1)
        intersection = union_w * union_h
        area_bb_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        area_bb_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        union = area_bb_test + area_bb_gt - intersection
        return intersection / (union + 1e-6)
