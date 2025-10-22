import os
import time
import cv2

from pyengine.utils.logger import logger


class StreamReader:
    """
    Simplified StreamReader for reading video streams/files.

    Usage:
        reader = StreamReader(url, width=640, height=640, fps=30).open()

        with StreamReader(url).open() as reader:
           while reader.is_connected():
               frame = reader.read()
               ...

        or:

        while reader.is_connected():
            frame = reader.read()

        reader.close()
    """

    def __init__(self, url,
                 *, width: int = -1,
                 height: int = -1,
                 fps: int = -1,
                 max_retries: int = 5,
                 retry_delay: int = 2):
        """
        Initialize StreamReader.

        Args:
            url: Video source (RTSP URL, HTTP URL, camera index, or file path)
            width: Target width (-1 for native resolution)
            height: Target height (-1 for native resolution)
            fps: Target FPS (-1 for native/unlimited FPS)
            max_retries: Maximum reconnection attempts
            retry_delay: Delay between reconnection attempts (seconds)
        """
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.cap = None
        self.last_frame_time = 0
        self.frame_time = 0.0
        self._is_opened = False

        # Flags for native parameters
        self.use_native_width = (width == -1)
        self.use_native_height = (height == -1)
        self.use_native_fps = (fps == -1)
        self._use_any_native = self.use_native_width or self.use_native_height or self.use_native_fps

    def open(self) -> 'StreamReader':
        """
        Open the video stream.

        Returns:
            Self for method chaining

        Raises:
            ConnectionError: If failed to open stream
            FileNotFoundError: If local file doesn't exist
        """
        self.cap = self._open_stream(self.url)
        if not self.cap:
            raise ConnectionError(f"Failed to open stream from {self.url}.")

        self._is_opened = True

        # Initialize parameters
        if self._use_any_native:
            self._refresh_native_params()
        else:
            self._recompute_frame_time()

        logger.info("StreamReader", f"Stream opened: {self.url}")
        return self

    def read(self):
        """
        Read next frame from stream with FPS throttling.

        Returns:
            Frame (numpy array) if successful and FPS throttle allows, None otherwise
        """
        if not self._is_opened or not self.cap or not self.cap.isOpened():
            return None

        current_time = time.time()

        # FPS throttling
        if self.frame_time > 0.0 and (current_time - self.last_frame_time) < self.frame_time:
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None

        if self.frame_time > 0.0:
            self.last_frame_time = current_time

        # Resize if dimensions are specified
        if (self.width and self.width > 0) and (self.height and self.height > 0):
            return cv2.resize(frame, (int(self.width), int(self.height)))
        else:
            return frame

    def is_connected(self) -> bool:
        """
        Check if stream is connected and ready to read.

        Returns:
            True if connected, False otherwise
        """
        return self._is_opened and self.cap is not None and self.cap.isOpened()

    def close(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()
            logger.info("StreamReader", f"Stream closed: {self.url}")
        self._is_opened = False
        self.cap = None

    def reconnect(self) -> bool:
        """
        Attempt to reconnect to the stream.

        Returns:
            True if reconnection successful, False otherwise
        """
        logger.info("StreamReader", f"Attempting to reconnect to {self.url}...")

        if self.cap:
            self.cap.release()
            self.cap = None

        for attempt in range(self.max_retries):
            logger.info("StreamReader", f"Reconnection attempt {attempt + 1}/{self.max_retries}...")
            try:
                new_cap = self._open_stream(self.url)
                if new_cap and new_cap.isOpened():
                    self.cap = new_cap
                    self._is_opened = True

                    # Refresh native parameters if needed
                    if self._use_any_native:
                        self._refresh_native_params()
                    else:
                        self._recompute_frame_time()

                    logger.info("StreamReader", "Successfully reconnected.")
                    return True

            except FileNotFoundError:
                logger.error("StreamReader", f"Cannot reconnect: file '{self.url}' not found.")
                break
            except Exception as e:
                logger.error("StreamReader", f"Reconnection error: {e}")

            time.sleep(self.retry_delay)

        logger.error("StreamReader", f"Max retries reached. Could not reconnect to {self.url}.")
        self._is_opened = False
        return False

    def _open_stream(self, url):
        """
        Open video capture from URL/file/camera.

        Returns:
            cv2.VideoCapture object if successful, None otherwise

        Raises:
            FileNotFoundError: If local file doesn't exist
        """
        # Detect camera index
        if isinstance(url, str) and url.isdigit():
            logger.info("StreamReader", f"Detected camera index: {url}.")
            cap_url = int(url)
        # Detect streaming URL
        elif isinstance(url, str) and url.startswith(("rtsp://", "http://", "https://")):
            logger.info("StreamReader", f"Detected streaming URL: {url}.")
            cap_url = url
        # Local file
        else:
            if not os.path.exists(url):
                error_msg = f"File or device '{url}' does not exist."
                logger.error("StreamReader", error_msg)
                raise FileNotFoundError(error_msg)
            logger.info("StreamReader", f"Detected local file: {url}.")
            cap_url = url

        cap = cv2.VideoCapture(cap_url)
        if cap.isOpened():
            logger.info("StreamReader", f"Stream opened successfully from {url}.")
            return cap
        else:
            logger.error("StreamReader", f"Failed to open stream from {url}.")
            return None

    def _recompute_frame_time(self):
        """Compute frame interval based on FPS (0 means unlimited)."""
        if self.fps is not None and isinstance(self.fps, (int, float)) and self.fps > 0:
            self.frame_time = 1.0 / float(self.fps)
        else:
            self.frame_time = 0.0

    def _refresh_native_params(self):
        """Refresh native width/height/fps from capture device."""
        if not self.cap:
            return

        native_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        native_fps = float(self.cap.get(cv2.CAP_PROP_FPS))

        # Update width if using native
        if self.use_native_width:
            self.width = native_w if native_w > 0 else -1

        # Update height if using native
        if self.use_native_height:
            self.height = native_h if native_h > 0 else -1

        # Update FPS if using native
        if self.use_native_fps:
            if native_fps and native_fps > 0 and native_fps == native_fps:  # Check for NaN
                self.fps = int(round(native_fps))
            else:
                self.fps = -1

        self._recompute_frame_time()

        logger.info(
            "StreamReader",
            f"[params] width={self.width} ({'native' if self.use_native_width else 'user'}), "
            f"height={self.height} ({'native' if self.use_native_height else 'user'}), "
            f"fps={self.fps} ({'native' if self.use_native_fps else 'user' if self.fps > 0 else 'unlimited'})"
        )

    def __enter__(self):
        """Context manager entry."""
        if not self._is_opened:
            self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False