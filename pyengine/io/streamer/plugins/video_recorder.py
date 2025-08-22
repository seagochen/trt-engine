# video_plugins.py
from typing import Optional

from pyengine.io.streamer.video_maker import VideoMaker
from pyengine.io.streamer.video_plugins import VideoPlugin
from pyengine.utils.logger import logger


class VideoRecorderPlugin(VideoPlugin):
    """
    录制插件：启动时注册订阅者，拿到第一帧后按 bus.get_capture() 初始化 VideoMaker 并持续写帧
    """
    def __init__(self, filename_trunk: Optional[str] = "record", append_date: bool = True):
        self.filename_trunk = filename_trunk
        self.append_date = append_date
        self._bus = None
        self._sub = None
        self._vm: Optional[VideoMaker] = None
        self._started = False

    def start(self, bus) -> None:
        self._bus = bus
        self._started = True

        def _on_frame(frame):
            # 首帧时按实际 cap 初始化 VideoMaker(沿用你的自动 codec 选择逻辑)
            if self._vm is None:
                cap = self._bus.get_capture()
                if cap is None or not cap.isOpened():
                    logger.warning("VideoRecorderPlugin", "no valid cap; cannot init writer yet"); return
                self._vm = VideoMaker(cap, output_trunk_name=self.filename_trunk, append_date=self.append_date)
                logger.info("VideoRecorderPlugin", f"writer ready: {self._vm.generated_filename()}")
            # 写帧(VideoMaker 内部会自动 resize 到 writer 尺寸)
            try:
                self._vm.add_frame(frame)
            except Exception as e:
                logger.error_trace("VideoRecorderPlugin", f"write error: {e}")

        self._sub = self._bus.subscribe(_on_frame)

    def stop(self) -> None:
        if self._bus and self._sub:
            self._bus.unsubscribe(self._sub)
        self._sub = None
        if self._vm:
            try:
                self._vm.release()
            except Exception:
                pass
        self._vm = None
        self._bus = None
        self._started = False

