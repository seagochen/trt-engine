import numpy as np

from common.tracker.kalman_filter import KalmanFilter


class TrackerManager:

    def __init__(self, max_age: int, max_objects: int):
        self.max_age = max_age
        self.trackers = []  # 存储所有活跃的追踪器
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


    def update_trackers(self, matched, detections):
        """更新匹配上的追踪器状态"""
        detection_map = {}
        for item in matched:

            # item[0] 为检测结果索引，item[1] 为追踪器索引
            index_det = item[0]
            index_trk = item[1]

            # detectionsからbboxを取得
            bbox = detections[index_det][:4]

            # 目標の追跡器を更新
            self.trackers[index_trk].update(bbox)

            # 検出結果のインデックスを保存
            detection_map[self.trackers[index_trk].id] = index_det

        return detection_map


    def add_trackers(self, unmatched_dets, detection):
        """
        为未匹配的检测结果创建新的追踪器
        """
        detection_map = {}
        for idx in unmatched_dets:

            # bboxを取得
            bbox = detection[idx][:4]

            # 新しいIDを取得
            new_id = self._get_next_id()

            # 新しい追跡器を作成
            tracker = KalmanFilter(new_id, bbox)

            # 追跡器を追加
            self.trackers.append(tracker)

            # 検出結果のインデックスを保存
            detection_map[tracker.id] = idx 

        return detection_map


    def commit_changes(self):
        """更新所有追踪器状态，删除超时未更新的追踪器"""
        active_trackers = []

        # 更新所有追踪器状态
        for tracker in self.trackers:

            # 如果追踪器未超时，保留
            if tracker.time_since_update <= self.max_age:
                active_trackers.append(tracker)

            # 释放超时未更新的ID
            else:
                self._release_id(tracker.id)  
        
        # 更新追踪器列表
        self.trackers = active_trackers


    def get_tracker_states(self):
        """返回所有追踪器的状态"""
        states = []
        for tracker in self.trackers:

            # 预测追踪器状态
            pos = tracker.predict()

            # 过滤NaN值
            if not np.any(np.isnan(pos)):
                states.append([*tracker.get_state(), tracker.id]) # 追踪器状态: [x1, y1, x2, y2, id]

            # 释放NaN状态的追踪器ID
            else:
                self._release_id(tracker.id) 

        # 返回追踪器状态: [x1, y1, x2, y2, id]
        return np.array(states)

