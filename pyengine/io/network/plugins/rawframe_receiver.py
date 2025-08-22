# ---- Add to mqtt_plugins.py ----
"""
import time

import cv2

from pyengine.io.network.mqtt_bus import MqttBus
from pyengine.io.network.mqtt_plugins import MqttPluginManager
from pyengine.io.network.plugins.frame_reader import FrameReaderMqttPlugin
from pyengine.io.network.plugins.hello_world import HelloWorldSenderPlugin, HelloWorldReceiverPlugin

# 1) 启动 Bus(单处创建，统一管理连接/重连/扇出)
bus = MqttBus(host="127.0.0.1", port=1883, client_id="demo_bus")
bus.start()  # 内含重连与订阅恢复逻辑，发布接口可被插件直接复用。:contentReference[oaicite:3]{index=3}

# 2) 装插件
pm = MqttPluginManager(bus)  # 多插件共享一个宿主(bus)运行。:contentReference[oaicite:4]{index=4}
reader = FrameReaderMqttPlugin(topic="raw/frames/cam1", pb2_dir="./protobufs",
                                width=1280, height=720)
pm.register(reader)

# 3) 启动插件
pm.start()

# 等首帧(最多等 10 秒)
t0 = time.time()
first = None
while first is None and time.time() - t0 < 10:
    first = reader.read_frame()
    cv2.waitKey(1)  # 让UI线程处理消息

if first is None:
    print("[demo] timeout: no frame received on topic 'raw/frames/cam1'")
    pm.stop(); bus.stop()
    raise SystemExit(1)

# 正常循环
while True:
    frame = reader.read_frame()
    if frame is not None:
        cv2.imshow("demo", frame)
    # 没有新帧时，不要把 None 传给 imshow；仅处理事件即可
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pm.stop(); bus.stop()
cv2.destroyAllWindows()
"""

import queue
import threading
import time
from typing import Optional, Callable

import cv2
import numpy as np
import extends

from pyengine.io.network.mqtt_plugins import MqttPlugin, IMqttHost


class RawFrameReceiverPlugin(MqttPlugin):
    """
    订阅 MQTT 图像流(RawFrame)，解包为 np.ndarray 并提供：
      - read_frame(): 拉取最新帧
      - 可选 on_frame 回调：每次收到帧回调一次
      - 可选 frame_queue：将帧入队 (frame, source_id)

    依赖：
      - 宿主需支持 subscribe()/unsubscribe()(MqttBus OK)
      - RawFrame protobuf：extends.import_rawframe(pb2_dir)
    """
    def __init__(self,
                 topic: str,
                 pb2_dir: str = "./protobufs",
                 width: int = -1,
                 height: int = -1,
                 fps: int = -1,
                 source_id: str = "mqtt",
                 on_frame: Optional[Callable[[np.ndarray, str], None]] = None,
                 frame_queue: Optional[queue.Queue] = None):
        self.topic = topic
        self.pb2_dir = pb2_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_time = (1.0 / fps) if isinstance(fps, (int, float)) and fps > 0 else 0.0
        self.source_id = source_id
        self.on_frame = on_frame
        self.frame_queue = frame_queue

        self._RawFrame = extends.import_rawframe(pb2_dir)
        self._host: Optional[IMqttHost] = None
        self._sub_handle = None

        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._last_rx_ts: float = 0.0
        self._last_read_ts: float = 0.0
        self._new_ready: bool = False
        self._started: bool = False

    def start(self, host: IMqttHost) -> None:
        self._host = host
        # 订阅：MqttBus 每个订阅有独立队列/worker，不会阻塞网络回调
        self._sub_handle = host.subscribe(self.topic, self._on_message, qos=0)
        self._started = True

    def stop(self) -> None:
        self._started = False
        if self._host and self._sub_handle:
            try:
                self._host.unsubscribe(self._sub_handle)
            except Exception:
                pass
        self._sub_handle = None
        self._host = None

    # 外部可轮询读取最新帧(复制一份，线程安全)
    def read_frame(self) -> Optional[np.ndarray]:
        if not self._started:
            return None
        now = time.time()
        if self.frame_time > 0.0 and (now - self._last_read_ts) < self.frame_time:
            return None
        with self._lock:
            if not self._new_ready or self._latest is None:
                return None
            frame = self._latest.copy()
            self._new_ready = False
        self._last_read_ts = now
        if self.width > 0 and self.height > 0:
            frame = cv2.resize(frame, (int(self.width), int(self.height)))
        return frame

    # 最近5秒有帧 + 宿主在线(若宿主暴露 is_connected)
    def is_alive(self, recent_sec: float = 5.0) -> bool:
        recent = (time.time() - self._last_rx_ts) < recent_sec
        online = getattr(self._host, "is_connected", True)
        return bool(online and recent)

    # MQTT 消息回调(RawFrame → np.ndarray)
    def _on_message(self, topic: str, payload: bytes):
        if topic != self.topic:
            return
        try:
            msg = self._RawFrame()
            msg.ParseFromString(payload)
        except Exception:
            return

        w = int(getattr(msg, "frame_width", 0))
        h = int(getattr(msg, "frame_height", 0))
        c = int(getattr(msg, "frame_channels", 3))
        data = bytes(getattr(msg, "frame_raw_data", b""))

        frame = None
        if data:
            # 优先当JPEG解码；失败再按裸像素兜底
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None and w > 0 and h > 0 and c in (1, 3):
                try:
                    raw = np.frombuffer(data, dtype=np.uint8)
                    frame = raw.reshape((h, w, c)).copy()
                except Exception:
                    frame = None
        if frame is None:
            return

        self._last_rx_ts = time.time()
        # 写入最新帧缓存
        with self._lock:
            self._latest = frame
            self._new_ready = True
        # 可选回调/入队(尽量轻量，不做重活)
        if self.on_frame:
            try:
                self.on_frame(frame, self.source_id)
            except Exception:
                pass
        if self.frame_queue is not None:
            try:
                self.frame_queue.put_nowait((frame, self.source_id))
            except Exception:
                # 背压策略：满了就丢
                pass
