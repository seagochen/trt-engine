import cv2
import time
import os
import threading
from pyengine.utils.logger import logger


class StreamReader:
    def __init__(self, url,
                 width: int = -1,
                 height: int = -1,
                 fps: int = -1,
                 max_retries: int = 5,
                 delay: int = 2):
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        self.max_retries = max_retries
        self.delay = delay

        # 打开视频流，但不在这里启动读取
        self.cap = self.open_camera_stream(url)
        if not self.cap:
            raise ConnectionError(f"Failed to initialize camera stream from {url}.")

        # === 新：按项决定是否使用原生值 ===
        self.use_native_width = (width == -1)
        self.use_native_height = (height == -1)
        self.use_native_fps = (fps == -1)
        self._use_any_native = self.use_native_width or self.use_native_height or self.use_native_fps

        # 线程相关
        self.latest_frame = None
        self.last_read_success = False
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True

        # 初始化参数：若任一项走原生，则刷新一次原生参数；
        # 否则仅根据传入 fps 设定节流。
        if self._use_any_native:
            self._refresh_native_params()
        self._recompute_frame_time()

        self.last_frame_time = 0
        logger.info("StreamReader", f"Camera stream for {url} initialized. Call start() to begin reading.")

    def _recompute_frame_time(self):
        """根据当前 self.fps 重新计算帧时间(fps<=0 表示不限帧率)"""
        if self.fps is not None and isinstance(self.fps, (int, float)) and self.fps > 0:
            self.frame_time = 1.0 / float(self.fps)
        else:
            self.frame_time = 0.0  # 不限帧率

    def _refresh_native_params(self):
        """从当前 cap 中读取原生 width/height/fps，并仅对选择了原生的项生效"""
        native_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        native_fps = float(self.cap.get(cv2.CAP_PROP_FPS))

        # 宽高：只在对应 use_native_* 为 True 时覆盖
        if self.use_native_width:
            self.width = native_w if native_w > 0 else -1
        if self.use_native_height:
            self.height = native_h if native_h > 0 else -1

        # FPS：只在 use_native_fps 为 True 时覆盖
        if self.use_native_fps:
            if native_fps and native_fps > 0 and native_fps == native_fps:  # 排除 NaN
                # 有些流返回小数 FPS，这里统一取四舍五入的 int，节流足够用
                self.fps = int(round(native_fps))
            else:
                self.fps = -1  # 拿不到原生 FPS 就表示不限帧率

        # 重新计算帧间隔
        self._recompute_frame_time()

        logger.info(
            "StreamReader",
            f"[params] width={self.width} ({'native' if self.use_native_width else 'user'}), "
            f"height={self.height} ({'native' if self.use_native_height else 'user'}), "
            f"fps={self.fps} ({'native' if self.use_native_fps else 'user' if self.fps>0 else 'unlimited'})"
        )

    @staticmethod
    def open_camera_stream(url):
        """
        打开摄像头或视频文件流。
        如果是本地文件且不存在，则直接抛出 FileNotFoundError。
        """
        if isinstance(url, str) and url.isdigit():
            logger.info("StreamReader", f"Detected camera index: {url}.")
            cap_url = int(url)
        elif isinstance(url, str) and url.startswith(("rtsp://", "http://", "https://")):
            logger.info("StreamReader", f"Detected streaming URL: {url}.")
            cap_url = url
        else:
            if not os.path.exists(url):
                error_msg = f"File or device '{url}' does not exist. Please check the path."
                logger.error("StreamReader", error_msg)
                raise FileNotFoundError(error_msg)
            else:
                logger.info("StreamReader", f"Detected local file path: {url}.")
                cap_url = url

        cap = cv2.VideoCapture(cap_url)
        if cap.isOpened():
            logger.info("StreamReader", f"Camera stream opened successfully from {url}.")
            return cap
        else:
            logger.error("StreamReader", f"Failed to open stream from {url}. Check address/device accessibility.")
            return None

    def start(self):
        """启动后台线程抓帧"""
        if self.stopped:
            logger.warning("StreamReader", "Cannot start a stopped stream. Please create a new instance.")
            return
        self.stopped = False
        self.thread.start()
        logger.info("StreamReader", f"Background frame reader started for {self.url}.")
        return self

    def _update(self):
        """后台线程：持续读取最新一帧"""
        while not self.stopped:
            if not (self.cap and self.cap.isOpened()):
                logger.warning("StreamReader", "Stream is disconnected. Attempting to reconnect...")
                self.reconnect_camera()
                if not (self.cap and self.cap.isOpened()):
                    time.sleep(self.delay)
                    continue

            ret, frame = self.cap.read()
            with self.lock:
                if ret:
                    self.last_read_success = True
                    self.latest_frame = frame
                else:
                    self.last_read_success = False
                    self.cap.release()
                    logger.warning("StreamReader", f"Failed to read frame from {self.url}. Will attempt to reconnect.")

    def read_frame(self):
        """根据 FPS 节流，从后台线程拿最新帧；若尺寸未指定(-1)，不做 resize。"""
        current_time = time.time()
        # 没有限制或到了该取下一帧的时间
        if self.frame_time == 0.0 or (current_time - self.last_frame_time) >= self.frame_time:
            with self.lock:
                if not self.last_read_success:
                    return None
                frame = self.latest_frame.copy()

            if self.frame_time > 0.0:
                self.last_frame_time = current_time

            # 只在 width>0 且 height>0 时进行 resize；否则按源尺寸返回
            if (self.width and self.width > 0) and (self.height and self.height > 0):
                return cv2.resize(frame, (int(self.width), int(self.height)))
            else:
                return frame
        else:
            return None

    def reconnect_camera(self):
        """重连逻辑：若任一项走原生，重连后刷新原生参数"""
        logger.info("StreamReader", f"Attempting to reconnect to {self.url}...")
        if self.cap:
            self.cap.release()

        for attempt in range(self.max_retries):
            logger.info("StreamReader", f"Reconnection attempt {attempt + 1}/{self.max_retries}...")
            try:
                new_cap = self.open_camera_stream(self.url)
                if new_cap and new_cap.isOpened():
                    self.cap = new_cap
                    # 重连成功：若需要原生参数，刷新一遍
                    if self._use_any_native:
                        self._refresh_native_params()
                    else:
                        self._recompute_frame_time()
                    logger.info("StreamReader", "Successfully reconnected.")
                    return
            except FileNotFoundError:
                logger.error("StreamReader", f"Cannot reconnect: file '{self.url}' not found. Stopping retries.")
                break
            time.sleep(self.delay)

        logger.error("StreamReader", f"Max retries reached. Could not reconnect to {self.url}.")

    def stop(self):
        """停止线程并释放资源"""
        logger.info("StreamReader", f"Stopping stream reader for {self.url}...")
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("StreamReader", "Camera stream closed successfully.")

    def is_connected(self):
        """检查流是否连接"""
        with self.lock:
            return self.last_read_success and self.cap and self.cap.isOpened()
