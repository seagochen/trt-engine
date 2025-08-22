import time
from typing import Callable, Any, Optional

from pyengine.io.streamer.video_plugins import VideoPlugin
from pyengine.utils.logger import logger


class VideoInferencePlugin(VideoPlugin):
    """
    推理插件示例：从总线收帧，调用用户提供的 infer(frame) → result；
    - on_result 回调可把结果发 MQTT/画框/落库等
    - 可在内部做限速(例如每 N 帧/每 T 秒推理一次)
    """
    def __init__(self,
                 infer_fn: Callable[[Any], Any],
                 on_result: Optional[Callable[[Any, Any], None]] = None,
                 throttle_sec: float = 0.0):
        self.infer_fn = infer_fn
        self.on_result = on_result
        self.throttle_sec = throttle_sec
        self._bus = None
        self._sub = None
        self._last_ts = 0.0

    def start(self, bus) -> None:
        self._bus = bus

        def _on_frame(frame):
            now = time.time()
            if self.throttle_sec > 0 and (now - self._last_ts) < self.throttle_sec:
                return
            self._last_ts = now
            try:
                result = self.infer_fn(frame)
                if self.on_result:
                    self.on_result(frame, result)
            except Exception as e:
                logger.error_trace("VideoInferencePlugin", f"infer error: {e}")

        self._sub = self._bus.subscribe(_on_frame)

    def stop(self) -> None:
        if self._bus and self._sub:
            self._bus.unsubscribe(self._sub)
        self._sub = None
        self._bus = None