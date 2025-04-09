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
            logger.error("StreamReader", f"Failed to initialize camera stream from {url}.")

    @staticmethod
    def open_camera_stream(url):
        """
        打开摄像头或视频文件流：
          - 如果 URL 是数字字符串，则认为是摄像头索引；
          - 如果 URL 对应一个本地视频文件，则提前检测该文件是否存在；
          - 如果 URL 是流媒体地址（以 rtsp://、http:// 或 https:// 开头），则直接尝试打开；
          - 否则，若检测不到文件，则直接返回 None 并打印错误信息。
        """
        # 判断是否为摄像头索引（数字字符串）
        if isinstance(url, str) and url.isdigit():
            logger.info("StreamReader", f"Detected camera index: {url}.")
        # 判断是否为本地视频文件
        elif os.path.isfile(url):
            logger.info("StreamReader", f"Detected local video file: {url}.")
        # 判断是否为流媒体 URL
        elif url.startswith(("rtsp://", "http://", "https://")):
            logger.info("StreamReader", f"Detected streaming URL: {url}.")
        else:
            # 如果既不符合以上情况，则检测该路径是否存在
            if not os.path.exists(url):
                logger.error("StreamReader", f"File '{url}' does not exist. Please check the file path.")
                return None
            else:
                logger.info("StreamReader", f"Detected file path: {url}.")

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
        connected = self.cap.isOpened()
        if not connected:
            logger.debug("StreamReader", "Camera stream is not connected.")
        return connected

    def read_frame(self):
        """从摄像头流中读取一帧图像"""
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
            cap = self.open_camera_stream(url)
            if cap:
                logger.info("StreamReader", f"Successfully reconnected to the camera on attempt {attempts + 1}.")
                return cap
            else:
                logger.warning("StreamReader", f"Reconnection attempt {attempts + 1}/{max_retries} failed. Retrying in {delay} seconds...")
                attempts += 1
                time.sleep(delay)
        logger.error("StreamReader", "Max retries reached. Could not reconnect to the camera.")
        return None

    def close_camera_stream(self):
        """关闭摄像头流"""
        if self.cap:
            self.cap.release()
            logger.info("StreamReader", "Camera stream closed successfully.")
        else:
            logger.warning("StreamReader", "Attempted to close a camera stream that was not open.")
