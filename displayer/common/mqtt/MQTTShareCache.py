#!/usr/bin/env python
# coding: utf-8

from common.yaml.YamlConfig import YamlConfig
from queue import PriorityQueue

from protobufs import video_frame_pb2, inference_result_pb2


class MQTTShareCache:

    def __init__(self, config: YamlConfig, stakeholder: str):
        self.config = config
        self.stakeholder = stakeholder
        self.cache = PriorityQueue()


    def put(self, topic, payload):
        # Get inference stakeholder
        infer_config = self.config.get_inference_config()
        mqtt_config = self.config.get_mqtt_config()

        if topic == mqtt_config["infer_before_topic"]:
            video_frame = video_frame_pb2.VideoFrame()
            video_frame.ParseFromString(payload)
            if video_frame.publish_by == infer_config['inference_stakeholder']:


                print(f"Received video frame: {video_frame.frame_number}")

        elif topic == mqtt_config["infer_result_topic"]:
            inference_result = inference_result_pb2.InferenceResult()
            inference_result.ParseFromString(payload)
            if inference_result.publish_by == infer_config['inference_stakeholder']:
                print(f"Received inference result: {inference_result.frame_number}")