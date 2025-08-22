# inference_result_receiver.py
import queue
import threading
import time
from typing import Optional, Callable

import extends
from pyengine.io.network.mqtt_plugins import MqttPlugin, IMqttHost


class InferenceResultReceiverPlugin(MqttPlugin):
    """
    订阅 InferenceResult(protobuf)，直接返回/分发 Protobuf 实例：
      - read(): 返回最新一条 InferenceResult(protobuf)
      - on_result: 每到一条回调(参数为 protobuf)
      - result_queue: 可选，将 protobuf 入队
    """
    def __init__(self,
                 topic: str,
                 pb2_dir: str = "./protobufs",
                 on_result: Optional[Callable[[object], None]] = None,
                 result_queue: Optional[queue.Queue[object]] = None):
        self.topic = topic
        self.pb2_dir = pb2_dir
        self.on_result = on_result
        self.result_queue = result_queue

        # 动态加载 pb2: returns class of InferenceResult
        self._InferenceResult = extends.import_inference_result(pb2_dir)  # :contentReference[oaicite:3]{index=3}
        self._host: Optional[IMqttHost] = None
        self._sub_handle = None

        self._lock = threading.Lock()
        self._latest = None  # type: Optional[object]  # Protobuf实例
        self._new_ready = False
        self._started = False
        self._last_rx_ts = 0.0

    def start(self, host: IMqttHost) -> None:
        self._host = host
        # 每个订阅独立 worker，不阻塞网络线程
        self._sub_handle = host.subscribe(self.topic, self._on_message, qos=0)  # :contentReference[oaicite:4]{index=4}
        self._started = True

    def stop(self) -> None:
        self._started = False
        if self._host and self._sub_handle:
            try:
                self._host.unsubscribe(self._sub_handle)  # :contentReference[oaicite:5]{index=5}
            except Exception:
                pass
        self._sub_handle = None
        self._host = None

    def is_alive(self, recent_sec: float = 5.0) -> bool:
        recent = (time.time() - self._last_rx_ts) < recent_sec
        online = getattr(self._host, "is_connected", True)
        return bool(online and recent)

    def read(self):
        """
        返回最新一条 Protobuf InferenceResult(没有新数据时返回 None)。
        线程安全：返回的是同一对象的引用，如需持久化请自行复制其字段。
        """
        if not self._started:
            return None
        with self._lock:
            if not self._new_ready or self._latest is None:
                return None
            msg = self._latest
            self._new_ready = False
            return msg

    # ---- MQTT 回调：反序列化 → protobuf 直返
    def _on_message(self, topic: str, payload: bytes):
        if topic != self.topic:
            return
        try:
            msg = self._InferenceResult()
            msg.ParseFromString(payload)
        except Exception:
            return

        self._last_rx_ts = time.time()

        # 缓存最新版(protobuf)
        with self._lock:
            self._latest = msg
            self._new_ready = True

        # 回调/入队(传 protobuf)
        if self.on_result:
            try:
                self.on_result(msg)
            except Exception:
                pass
        if self.result_queue is not None:
            try:
                self.result_queue.put_nowait(msg)
            except Exception:
                pass
