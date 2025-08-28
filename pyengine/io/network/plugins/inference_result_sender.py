# inference_result_sender.py
import json
from typing import List

import numpy as np


from pyengine.inference.unified_structs.inference_results import Skeleton
from pyengine.io.network import protobufs
from pyengine.io.network.mqtt_plugins import MqttPlugin


class InferenceResultSenderPlugin(MqttPlugin):
    def __init__(self, topic: str, pb2_dir: str = "./protobufs", qos: int = 1, retain: bool = False):
        self.topic = topic
        self.pb2_dir = pb2_dir
        self.qos = int(qos)
        self.retain = bool(retain)
        self._InferenceResult = protobufs.import_inference_result(pb2_dir)
        self._host = None

    def start(self, host) -> None:
        self._host = host

    def stop(self) -> None:
        self._host = None

    # ---- 仅支持单层 List[Skeleton] → JSON bytes(紧凑，UTF-8)----
    @staticmethod
    def _dump_inference_results(items: List[Skeleton]) -> bytes:
        def to_dict_safe(x):
            return x.to_dict() if hasattr(x, "to_dict") else x
        payload = [to_dict_safe(s) for s in (items or [])]
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    def send(self,
             frame_number: int,
             frame_width: int,
             frame_height: int,
             frame_channels: int,
             frame_raw_data: np.ndarray,
             publish_by: str,
             inference_result: List[Skeleton]) -> bool:  # ← 单层 List[Skeleton]
        """
        发布一条 InferenceResult(proto 字段：frame_number/width/height/channels/frame_raw_data/publish_by/inference_results)。
        注意：frame_raw_data 发送裸像素(无格式字段)。
        """
        if self._host is None:
            return False

        # --- 基本校验 ---
        if not isinstance(frame_number, (int, np.integer)):
            return False
        if not (isinstance(frame_width, (int, np.integer)) and isinstance(frame_height, (int, np.integer)) and isinstance(frame_channels, (int, np.integer))):
            return False
        if not isinstance(publish_by, str):
            return False
        if not isinstance(frame_raw_data, np.ndarray) or frame_raw_data.dtype != np.uint8:
            return False

        # --- 处理帧字节 ---
        try:
            if frame_raw_data.ndim == 1:
                raw_bytes = bytes(frame_raw_data)
            else:
                if frame_channels == 1:
                    if frame_raw_data.ndim != 2 or frame_raw_data.shape != (frame_height, frame_width):
                        return False
                else:
                    if frame_raw_data.ndim != 3 or frame_raw_data.shape[:2] != (frame_height, frame_width) or frame_raw_data.shape[2] != frame_channels:
                        return False
                raw_bytes = frame_raw_data.tobytes(order="C")
        except Exception:
            return False

        # --- 序列化推理结果(单层 List[Skeleton])---
        try:
            results_bytes = self._dump_inference_results(inference_result)
        except Exception:
            results_bytes = b""

        # --- 组装并发布 ---
        msg = self._InferenceResult()
        msg.frame_number = int(frame_number)
        msg.frame_width = int(frame_width)
        msg.frame_height = int(frame_height)
        msg.frame_channels = int(frame_channels)
        msg.frame_raw_data = raw_bytes
        msg.publish_by = publish_by
        msg.inference_results = results_bytes

        payload = msg.SerializeToString()
        try:
            return bool(self._host.publish(self.topic, payload, qos=self.qos, retain=self.retain))
        except Exception:
            return False
