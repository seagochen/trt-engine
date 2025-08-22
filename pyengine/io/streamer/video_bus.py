# video_bus.py
import time, threading
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Callable, List, Optional, Any

import cv2
from pyengine.utils.logger import logger
from .stream_reader import StreamReader

FrameHandler = Callable[[Any], None]  # handler(frame: np.ndarray) → None

@dataclass
class _Sub:
    queue: Queue
    worker: threading.Thread
    active: bool = True

class VideoBus:
    """
    管理视频流与帧扇出：
    - 统一 start(url)/stop()
    - 订阅：每个订阅者一个独立队列+worker，避免阻塞抓帧
    - 背压：队列满时丢最旧
    """
    def __init__(self, name: str = "video_bus", max_queue_per_sub: int = 64, poll_interval: float = 0.0):
        self.name = name
        self._reader: Optional[StreamReader] = None
        self._subs: List[_Sub] = []
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._loop: Optional[threading.Thread] = None
        self.max_queue_per_sub = max_queue_per_sub
        self.poll_interval = poll_interval  # 0 表示尽快

    # 生命周期
    def start(self, url: str, width: int = -1, height: int = -1, fps: int = -1):
        if self._reader:
            logger.warning("VideoBus", "already started")
            return self
        self._stop.clear()
        self._reader = StreamReader(url=url, width=width, height=height, fps=fps).start()
        self._loop = threading.Thread(target=self._run, name=f"{self.name}-loop", daemon=True)
        self._loop.start()
        logger.info("VideoBus", f"started url={url}")
        return self

    def stop(self):
        self._stop.set()
        if self._loop:
            self._loop.join(timeout=2.0)
        # 停掉订阅者
        with self._lock:
            for sub in self._subs:
                sub.active = False
                try:
                    sub.queue.put_nowait(None)
                except Exception:
                    pass
            self._subs.clear()
        # 关 reader
        try:
            if self._reader:
                self._reader.stop()
        except Exception:
            pass
        self._reader = None
        logger.info("VideoBus", "stopped")

    def is_running(self) -> bool:
        return self._reader is not None

    def get_capture(self) -> Optional[cv2.VideoCapture]:
        # 供录制插件复用 cap 参数(fps/width/height/编码器自适配等)
        return getattr(self._reader, "cap", None)

    # 订阅/退订
    def subscribe(self, handler: FrameHandler) -> _Sub:
        q = Queue(maxsize=self.max_queue_per_sub)

        def _worker():
            while True:
                try:
                    item = q.get(timeout=0.5)
                except Empty:
                    if not sub.active and q.empty():
                        break
                    continue
                if item is None:
                    break
                frame = item
                try:
                    handler(frame)
                except Exception as e:
                    logger.error_trace("VideoBus", f"handler error: {e}")

        sub = _Sub(queue=q, worker=threading.Thread(target=_worker, name=f"{self.name}-sub", daemon=True))
        with self._lock:
            self._subs.append(sub)
        sub.worker.start()
        return sub

    def unsubscribe(self, sub: _Sub):
        with self._lock:
            if sub in self._subs:
                sub.active = False
                try:
                    sub.queue.put_nowait(None)
                except Exception:
                    pass
                self._subs.remove(sub)

    # 内部：从 StreamReader 拉帧并扇出
    def _run(self):
        while not self._stop.is_set():
            if not self._reader:
                time.sleep(0.05); continue
            frame = self._reader.read_frame()
            if frame is None:
                time.sleep(self.poll_interval)
                continue
            # 扇出：丢最旧策略
            with self._lock:
                for sub in list(self._subs):
                    try:
                        sub.queue.put_nowait(frame)
                    except Exception:
                        # 丢旧帧腾位
                        try:
                            sub.queue.get_nowait()
                            sub.queue.put_nowait(frame)
                        except Exception:
                            pass
