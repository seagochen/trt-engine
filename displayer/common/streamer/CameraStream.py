#!/usr/bin/env python
# coding: utf-8

import cv2
import time


class CameraStream:

    def __init__(self, url, width, height, fps, max_retries=5, delay=2):
        self.url = url
        self.cap = self.open_camera_stream(url)
        self.max_retries = max_retries
        self.delay = delay
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_time = 1.0 / fps
        self.last_frame_time = time.time()

    @staticmethod
    def open_camera_stream(url):
        """Open a camera stream and check if it's successful."""
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            return cap
        else:
            return None

    def read_frame(self):
        """Read a frame from the camera stream."""
        current_time = time.time()
        if current_time - self.last_frame_time >= self.frame_time:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (self.width, self.height))
                self.last_frame_time = current_time
                return frame
            else:
                # If the frame read fails, try to reconnect to the camera stream
                self.cap = self.reconnect_camera(self.url, self.max_retries, self.delay)

                # If the reconnection is successful, read a frame
                if self.cap:
                    ret, frame = self.cap.read()
                    if ret:
                        frame = cv2.resize(frame, (self.width, self.height))
                        self.last_frame_time = current_time
                        return frame

                # If the reconnection fails, return None
                return None
        else:
            # If the time elapsed since the last frame is less than the frame time, return None
            return None

    def reconnect_camera(self, url, max_retries=5, delay=2):
        """Attempt to reconnect to the camera stream with retries."""
        attempts = 0
        while attempts < max_retries:
            cap = self.open_camera_stream(url)
            if cap:
                print("Successfully reconnected to the camera.")
                return cap
            else:
                print(f"Reconnection attempt {attempts + 1}/{max_retries} failed. Retrying in {delay} seconds...")
                attempts += 1
                time.sleep(delay)

        print("Max retries reached. Could not reconnect to the camera.")
        return None

    def close_camera_stream(self):
        """Close the camera stream."""
        # Release the camera stream
        self.cap.release()
