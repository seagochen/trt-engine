# rawframe_sender.py

import time
import threading
from typing import Optional

import cv2

from pyengine.io.network import protobufs
from pyengine.io.network.mqtt_plugins import MqttPlugin
from pyengine.io.streamer.stream_reader_old import StreamReader


class _FPSAverager:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.last_t = None
        self._fps = 0.0

    def tick(self):
        now = time.time()
        if self.last_t is not None:
            inst = 1.0 / max(1e-6, (now - self.last_t))
            self._fps = self.alpha * inst + (1 - self.alpha) * self._fps
        self.last_t = now

    @property
    def fps(self) -> float:
        return self._fps


class RawFrameSenderPlugin(MqttPlugin):
    """
    插件：从 url 取流 -> 打包为 RawFrame(JPEG) -> 通过 host.publish 发送到 topic

    注意：
    - 插件不管理连接(不 connect/disconnect)，只调用 host.publish(...)
    - 心跳/遗嘱建议交给 HeartbeatPlugin 或 MqttBus.set_will 在外部统一处理
    """

    def __init__(self,
                 url: str,
                 topic: str,
                 pb2_dir: str = "./protobufs",
                 jpeg_quality: int = 85,
                 send_fps: float = -1,
                 qos: int = 1,
                 retain: bool = False,
                 first_frame_timeout: float = 3.0,
                 idle_disconnect_sec: float = 5.0):
        self.url = url
        self.topic = topic
        self.pb2_dir = pb2_dir
        self.jpeg_quality = int(jpeg_quality)
        self.send_fps = float(send_fps)  # <=0 跟随源FPS(不额外限速)
        self.qos = int(qos)
        self.retain = bool(retain)
        self.first_frame_timeout = float(first_frame_timeout)
        self.idle_disconnect_sec = float(idle_disconnect_sec)

        self._RawFrame = protobufs.import_rawframe(pb2_dir)
        self._host = None

        self._reader: Optional[StreamReader] = None
        self._t: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # MqttPlugin 接口：start/stop
    def start(self, host) -> None:
        """
        host: 需要具备 publish(topic, payload, qos=..., retain=...) 方法
              (MqttBus 与 MQTTClient 都满足)
        """
        self._host = host
        self._stop.clear()
        self._t = threading.Thread(target=self._run, name="RawFrameSender", daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0)
        self._t = None
        try:
            if self._reader:
                self._reader.stop()
        except Exception:
            pass
        self._reader = None
        self._host = None

    # 工作线程：读取、节流、编码、发布
    def _run(self):
        # 1) 启动读取器(保持原生尺寸/FPS)
        self._reader = StreamReader(url=self.url, width=-1, height=-1, fps=-1)
        self._reader.start()

        # 2) 等首帧
        t0, first = time.time(), None
        while not self._stop.is_set() and time.time() - t0 < self.first_frame_timeout and first is None:
            first = self._reader.read_frame()
            if first is None:
                time.sleep(0.01)
        if first is None:
            # 首帧超时，直接退出线程
            return

        fps_avg = _FPSAverager()
        send_interval = 1.0 / self.send_fps if self.send_fps and self.send_fps > 0 else 0.0
        last_send_ts = 0.0
        last_ok_ts = time.time()

        # 3) 主循环
        while not self._stop.is_set():
            frame = self._reader.read_frame()
            if frame is None:
                # 源长时间无帧 → 退出
                if time.time() - last_ok_ts > self.idle_disconnect_sec:
                    break
                time.sleep(0.005)
                continue
            last_ok_ts = time.time()

            # 限速(默认不额外限速)
            now = time.time()
            if send_interval > 0.0 and (now - last_send_ts) < send_interval:
                continue
            last_send_ts = now

            # 编码 JPEG
            h, w = frame.shape[:2]
            c = 1 if frame.ndim == 2 else frame.shape[2]
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
            if not ok:
                continue
            jpeg_bytes = bytes(buf)

            # 估算 FPS
            fps_avg.tick()
            frame_fps = int(round(fps_avg.fps)) if fps_avg.fps > 0 else 0

            # 组 RawFrame
            msg = self._RawFrame()
            msg.frame_fps = frame_fps
            msg.frame_width = int(w)
            msg.frame_height = int(h)
            msg.frame_channels = int(c)
            msg.frame_raw_data = jpeg_bytes

            payload = msg.SerializeToString()

            # 发布(host 可以是 MqttBus 或 MQTTClient)
            try:
                if self._host:
                    self._host.publish(self.topic, payload, qos=self.qos, retain=self.retain)
            except Exception:
                # 断线时 publish 可能返回 False/抛异常；交给宿主重连与上层监控
                pass
