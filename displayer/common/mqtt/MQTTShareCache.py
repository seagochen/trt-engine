import cv2
import numpy as np
from queue import PriorityQueue

from common.utils.FPSCountingProcess import FPSCountingProcess
from common.yaml.YamlConfig import YamlConfig
from common.yolo.YoloInferenceResult import YoloInferenceResults
from protobufs import video_frame_pb2, inference_result_pb2


class MQTTShareCache:
    def __init__(self, config: YamlConfig, stakeholder: str):
        self.config = config
        self.stakeholder = stakeholder
        self.cache = PriorityQueue()
        self.infer_config = config.get_inference_config()
        self.mqtt_config = config.get_mqtt_config()

        # Initialize the FPS counting process
        self.fps_process = FPSCountingProcess()
        self.fps_process.start()

    def has_id(self, frame_id):
        return any(frame_id == item[1].frame_number for item in self.cache.queue)

    def count(self):
        return self.cache.qsize()

    def get(self, frame_id):
        return next((item[1] for item in self.cache.queue if item[1].frame_number == frame_id), None)

    def pop_when_ready(self):
        if self.cache.empty():
            return None

        item = self.cache.get()
        if item[1].is_ready():
            return item[1]
        else:
            self.cache.put(item)
            return None

    def put(self, topic, payload):
        if topic == self.mqtt_config["infer_before_topic"]:
            self._handle_video_frame(payload)
        elif topic == self.mqtt_config["infer_result_topic"]:
            self._handle_inference_result(payload)

    def _handle_video_frame(self, payload):
        video_frame = video_frame_pb2.VideoFrame()
        video_frame.ParseFromString(payload)

        if video_frame.publish_by != self.infer_config['inference_stakeholder']:
            return

        # Parse the video frame
        cv_image = self._parse_video_frame(video_frame)

        # Increment the FPS counter
        self.fps_process.increment_frame_count()

        # Get the current FPS
        fps = self.fps_process.get_fps()

        # Draw the FPS on the frame
        cv2.putText(cv_image, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Update the cache
        self._update_cache(video_frame.frame_number, frame=cv_image)

    def _handle_inference_result(self, payload):
        inference_result = inference_result_pb2.InferenceResult()
        inference_result.ParseFromString(payload)

        if inference_result.publish_by != self.infer_config['inference_stakeholder']:
            return

        self._update_cache(inference_result.frame_number, results=inference_result.results)

    def _parse_video_frame(self, video_frame):
        np_array = np.frombuffer(video_frame.frame_raw_data, dtype=np.uint8)
        cv_image = np_array.reshape((video_frame.frame_height, video_frame.frame_width, 3))
        return cv2.resize(cv_image, (self.infer_config["inference_width"], self.infer_config["inference_height"]))

    def _update_cache(self, frame_number, frame=None, results=None):
        item = self.get(frame_number)
        if item is None:
            item = YoloInferenceResults()
            item.frame_number = frame_number

        if frame is not None:
            item.frame_data = frame
            item.frame_ready = True

        if results is not None:
            item.results = results
            item.results_ready = True

        self.cache.put((frame_number, item))