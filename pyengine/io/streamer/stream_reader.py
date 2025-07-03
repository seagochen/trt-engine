import cv2
import time
import os
from pyengine.utils.logger import logger


class StreamReader:
    def __init__(self, url, width, height, fps, max_retries=5, delay=2):
        self.url = url
        self.cap = self.open_camera_stream(url)
        self.max_retries = max_retries
        self.delay = delay
        self.width = width
        self.height = height
        self.fps = fps

        if fps != -1:
            self.frame_time = 1.0 / fps
        else:
            # fps == -1 表示不限制帧率，不使用延时功能
            self.frame_time = 0

        self.last_frame_time = time.time()

        if self.cap:
            logger.info("StreamReader", f"Camera stream initialized successfully from {url}.")
        else:
            # 在 open_camera_stream 中已经抛出异常，理论上不会到这里，但作为保险
            logger.error("StreamReader", f"Failed to initialize camera stream from {url}.")

    @staticmethod
    def open_camera_stream(url):
        """
        打开摄像头或视频文件流。
        (修改) 如果是本地文件且不存在，则直接抛出 FileNotFoundError。
        """
        # 判断是否为摄像头索引（数字字符串）
        if isinstance(url, str) and url.isdigit():
            logger.info("StreamReader", f"Detected camera index: {url}.")
        # 判断是否为流媒体 URL
        elif url.startswith(("rtsp://", "http://", "https://")):
            logger.info("StreamReader", f"Detected streaming URL: {url}.")
        # 否则，一律当作文件路径处理
        else:
            if not os.path.exists(url):
                # (修改) 不再返回 None，而是直接抛出异常
                error_msg = f"File or device '{url}' does not exist. Please check the path."
                logger.error("StreamReader", error_msg)
                raise FileNotFoundError(error_msg)
            else:
                logger.info("StreamReader", f"Detected local file path: {url}.")

        # 尝试使用 OpenCV 打开流
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            logger.info("StreamReader", f"Camera stream opened successfully from {url}.")
            return cap
        else:
            logger.error(
                "StreamReader",
                f"Failed to open camera stream from {url}. "
                "Please check if the address is correct or if the stream device is accessible."
            )
            return None

    def is_connected(self):
        # 增加对 self.cap 是否为 None 的检查
        return self.cap and self.cap.isOpened()

    def read_frame(self):
        """从摄像头流中读取一帧图像"""
        if not self.is_connected():
            logger.warning("StreamReader", "Stream is not connected. Attempting to reconnect.")
            self.cap = self.reconnect_camera(self.url, self.max_retries, self.delay)
            if not self.is_connected():
                logger.error("StreamReader", "Failed to read frame after reconnection attempts.")
                return None

        current_time = time.time()
        # 如果 fps 设置为 -1，则不使用延时控制，总是尝试读取
        if self.fps == -1 or current_time - self.last_frame_time >= self.frame_time:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (self.width, self.height))
                if self.fps != -1:
                    self.last_frame_time = current_time
                return frame
            else:
                logger.warning("StreamReader", "Failed to read frame, attempting to reconnect.")
                self.cap = self.reconnect_camera(self.url, self.max_retries, self.delay)
                if self.cap:
                    ret, frame = self.cap.read()
                    if ret:
                        frame = cv2.resize(frame, (self.width, self.height))
                        if self.fps != -1:
                            self.last_frame_time = current_time
                        logger.info("StreamReader", "Frame read successfully after reconnection.")
                        return frame
                logger.error("StreamReader", "Failed to read frame after reconnection.")
                return None
        else:
            return None

    def reconnect_camera(self, url, max_retries=5, delay=2):
        """重连摄像头流，最多尝试 max_retries 次，每次间隔 delay 秒"""
        attempts = 0
        while attempts < max_retries:
            logger.info("StreamReader", f"Reconnection attempt {attempts + 1}/{max_retries}...")
            try:
                # 尝试重新打开流，同样会触发文件存在性检查
                cap = self.open_camera_stream(url)
                if cap and cap.isOpened():
                    logger.info("StreamReader", f"Successfully reconnected to the stream on attempt {attempts + 1}.")
                    return cap
            except FileNotFoundError:
                # 如果在重连时文件仍然不存在，则直接失败
                logger.error("StreamReader", f"Cannot reconnect: file '{url}' not found.")
                return None

            logger.warning("StreamReader", f"Reconnection attempt failed. Retrying in {delay} seconds...")
            attempts += 1
            time.sleep(delay)

        logger.error("StreamReader", "Max retries reached. Could not reconnect to the stream.")
        return None

    def close_camera_stream(self):
        """关闭摄像头流"""
        if self.cap:
            self.cap.release()
            logger.info("StreamReader", "Camera stream closed successfully.")
        else:
            logger.warning("StreamReader", "Attempted to close a camera stream that was not open.")