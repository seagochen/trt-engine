# -*- coding: utf-8 -*-
"""
StreamReaderMQTT
----------------
与 pyengine.io.streamer.stream_reader.StreamReader 的用法对齐，但数据来自 MQTT。
消息格式使用 ./protobufs/raw_frames_pb2.RawFrame（你的 5 个字段版本）：
  uint64 frame_fps
  int32  frame_width
  int32  frame_height
  int32  frame_channels
  bytes  frame_raw_data    # 建议放 JPEG 字节；也兼容原始 BGR 字节

依赖：
- pyengine.io.network.mqtt_client.MQTTClient  (你的封装)
- protobufs/raw_frames_pb2.py                  (已编译的 pb2)
"""

import time
import threading
from typing import Optional

import numpy as np
import cv2

import plugin
from pyengine.utils.logger import logger


class StreamReaderMQTT:
    def __init__(self,
                 host: str,
                 port: int,
                 topic: str,
                 client_id: str = "stream_reader_mqtt",
                 pb2_dir: str = "./protobufs",
                 width: int = -1,
                 height: int = -1,
                 fps: int = -1,
                 connect_timeout: int = 10):
        """
        :param host: MQTT broker host
        :param port: MQTT broker port
        :param topic: 订阅的图像主题（例如 raw/frames/cam1）
        :param client_id: MQTT client id
        :param pb2_dir: 包含 raw_frames_pb2.py 的目录
        :param width,height: 若 >0 则对输出帧 resize；否则保持原尺寸
        :param fps: 若 >0 则在 read_frame() 处节流输出帧率；否则不节流
        :param connect_timeout: 连接超时秒数
        """
        self.host = host
        self.port = port
        self.topic = topic
        self.client_id = client_id
        self.pb2_dir = pb2_dir

        self.width = width
        self.height = height
        self.fps = fps
        self.frame_time = (1.0 / fps) if (isinstance(fps, (int, float)) and fps > 0) else 0.0

        self._RawFrame = plugin.import_rawframe(pb2_dir)

        # 内部状态
        self.latest_frame: Optional[np.ndarray] = None
        self.last_rx_ts: float = 0.0           # 最近一次收到 MQTT 帧的时间
        self.last_consumed_rx_ts: float = 0.0  # 最近一次被 read_frame() 消费的时间
        self.last_frame_time: float = 0.0
        self.new_frame_available: bool = False
        self.lock = threading.Lock()
        self.started = False

        # MQTT 客户端
        from pyengine.io.network.mqtt_client import MQTTClient
        self.mqtt = MQTTClient(host=self.host, port=self.port, client_id=self.client_id)
        if hasattr(self.mqtt, "set_message_callback"):
            self.mqtt.set_message_callback(self._on_message)
        else:
            # 理论不会发生：你的封装就叫 set_message_callback
            raise RuntimeError("MQTTClient missing set_message_callback()")

        # 连接（同步阻塞，内部会 loop_start）
        if not self.mqtt.connect(timeout=connect_timeout):
            raise ConnectionError(f"MQTT connect to {host}:{port} failed")

        # 订阅主题
        ok = self.mqtt.subscribe(self.topic)
        if not ok:
            logger.warning("StreamReaderMQTT", f"Subscribe failed: {self.topic}")

        # 订阅心跳可选：self.mqtt.subscribe(self.topic.rstrip('/') + '/status')

        logger.info("StreamReaderMQTT", f"Initialized and subscribed to '{self.topic}'")

    # 兼容 StreamReader 的 start() 接口：无需真的起线程（MQTT loop 在 connect() 里已经启动）
    def start(self):
        self.started = True
        logger.info("StreamReaderMQTT", "Started.")
        return self

    def _on_message(self, topic: str, payload: bytes):
        # 只关心绑定的图像主题（若你同一个 client 要多路，可在外层建多个实例）
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
            # 优先按 JPEG 解码
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            # 不是 JPEG（可能发的是原始 BGR）
            if frame is None and w > 0 and h > 0 and c in (1, 3):
                try:
                    raw = np.frombuffer(data, dtype=np.uint8)
                    frame = raw.reshape((h, w, c)).copy()
                except Exception:
                    frame = None

        if frame is None:
            return

        with self.lock:
            self.latest_frame = frame
            self.last_rx_ts = time.time()
            # self.last_read_success = True
            self.new_frame_available = True

    def read_frame(self) -> Optional[np.ndarray]:
        """
        类似摄像头版：按 fps（如设定）节流输出最新帧；若没新帧/未连接则返回 None。
        """
        if not self.started:
            return None

        now = time.time()
        if self.frame_time > 0.0 and (now - self.last_frame_time) < self.frame_time:
            return None

        with self.lock:
            # if not self.last_read_success or self.latest_frame is None:
            # 没有新帧就不返回，避免重复消费同一帧导致 FPS 虚高
            if not self.new_frame_available or self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()
            self.last_consumed_rx_ts = self.last_rx_ts
            self.new_frame_available = False

        if self.frame_time > 0.0:
            self.last_frame_time = now

        # 按需 resize
        if (self.width and self.width > 0) and (self.height and self.height > 0):
            frame = cv2.resize(frame, (int(self.width), int(self.height)))

        return frame

    def is_connected(self) -> bool:
        """
        判定为连接：MQTT 客户端在线并且最近收到过帧（5s 之内）。
        """
        alive = getattr(self.mqtt, "is_connected", False)
        recent = (time.time() - self.last_rx_ts) < 5.0
        return bool(alive and recent)

    def stop(self):
        """
        关闭订阅与连接；释放资源。
        """
        logger.info("StreamReaderMQTT", "Stopping...")
        self.started = False
        try:
            if hasattr(self.mqtt, "disconnect"):
                self.mqtt.disconnect()
        except Exception:
            pass
        logger.info("StreamReaderMQTT", "Stopped.")
