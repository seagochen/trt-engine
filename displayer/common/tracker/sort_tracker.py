import numpy as np
from common.tracker.kalman_filter import KalmanFilter
from scipy.optimize import linear_sum_assignment


class SortTracker:
    def __init__(self, max_age=5, min_hits=1, iou_threshold=0.3, max_objects=1000):
        # 初始化追踪器的参数
        self.max_age = max_age  # 最大允许的未更新帧数
        self.min_hits = min_hits  # 需要的最小匹配次数来确认一个追踪器
        self.iou_threshold = iou_threshold  # 匹配时使用的 IOU 阈值
        self.trackers = []  # 存储所有活动的追踪器
        self.frame_count = 0  # 帧计数器
        self.available_ids = set()  # 可用ID的池子，用于重用ID

        # 初始化可用ID池子
        for i in range(max_objects):
            self.available_ids.add(i)

    def _get_next_id(self):
        """获取下一个可用的ID"""
        if self.available_ids:
            return self.available_ids.pop()
        else:
            raise Exception("No available ID left!")

    def _release_id(self, trk_id):
        """将ID放入可用池子"""
        self.available_ids.add(trk_id)

    def update(self, detections):
        self.frame_count += 1  # 增加帧计数器
        trks = np.zeros((len(self.trackers), 5))  # 用来存储当前所有追踪器的状态信息
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()  # 预测当前帧中的位置
            trks[t][:4] = trk.get_state()  # 获取追踪器的状态（bounding box）
            trks[t][4] = trk.id  # 存储追踪器的 ID

            # 检查预测的位置是否为 NaN 并删除相应的追踪器
            if np.any(np.isnan(pos)):
                self._release_id(trk.id)  # 释放ID
                self.trackers.pop(t)

            # 去除含有 NaN 的行
            trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        # 将检测结果与追踪器进行匹配
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(detections, trks)

        detection_map = {}  # 追踪器 ID 与检测结果索引的映射
        for m in matched:
            self.trackers[m[1]].update(detections[m[0]][:4])  # 更新匹配上的追踪器
            detection_map[self.trackers[m[1]].id] = m[0]  # 记录匹配信息

        for i in unmatched_dets:
            # 对于未匹配上的检测结果，创建新的 Kalman 追踪器
            trk = KalmanFilter(self._get_next_id(), detections[i][:4])  # 获取新的ID
            self.trackers.append(trk)  # 添加到追踪器列表中
            detection_map[trk.id] = i  # 记录新追踪器的信息

        ret = []  # 存储更新后的追踪器结果
        # 创建临时列表来保存活动的跟踪器
        active_trackers = []
        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:  # 检查追踪器是否在有效时间范围内
                bbox = trk.get_state()  # 获取追踪器的状态
                detection_idx = detection_map.get(trk.id)  # 获取对应的检测结果索引
                if detection_idx is not None and detection_idx < len(detections):
                    kpts_combined = detections[detection_idx][4:]  # 获取检测的关键点信息
                    ret.append([trk.id] + bbox + kpts_combined.tolist())  # 保存追踪器 ID、bounding box 和关键点信息

                active_trackers.append(trk)  # 将活跃的追踪器添加到新列表中
            else:
                # 如果超出 max_age，释放其 ID
                self._release_id(trk.id)

        # 更新追踪器列表，仅保留活跃的追踪器
        self.trackers = active_trackers

        return np.array(ret)  # 返回更新后的追踪结果

    def _associate_detections_to_trackers(self, detections, trackers):
        # 如果没有追踪器存在，直接返回所有检测结果为未匹配
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        # 计算每个检测结果与追踪器之间的 IOU（交并比）矩阵
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det[:4], trk[:4])

        # 使用匈牙利算法（线性分配问题）匹配检测结果和追踪器
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
            # 检查 IOU 是否低于阈值，如果低于则标记为未匹配
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

    def _iou(self, bb_test, bb_gt):
        # 计算两个 bounding box 之间的交并比 (IoU)
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
        return intersection / (union + 1e-6)  # 返回 IOU 值，防止除零
