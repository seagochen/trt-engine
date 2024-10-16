import argparse
import os
import sys
import time

import cv2
import numpy as np

from common.cache.cache_ver1 import MqttCache
from common.network.mqtt_client import MQTTClient
from common.painter.asc_text import draw_text_with_opposite_color
from common.utils.fps_process import FPSCountingProcess
from common.utils.load_labels import read_labels_from_file
from common.utils.logger import Logger
from common.yaml.yaml_reader import load_config
from common.yolo.simple_structs import YoloPose
from common.yolo.visualization import draw_skeletons, draw_boxes_with_labels
from protobufs.inference_result_pb2 import InferenceResult
from protobufs.video_frame_pb2 import VideoFrame

# Logger
logger = Logger()

# MQTT Cache
mqtt_cache = MqttCache(1024)

# Global variables
infer_topic = None
stream_topic = None
publish_by = None
model_name = None
labels = []


def parse_arguments():
    parser = argparse.ArgumentParser(description="MQTT Client for inference data processing")
    parser.add_argument("-c", "--config", default="configs/app.yaml", help="Path to the configuration file")
    return parser.parse_args()


def load_yaml(config_file):
    if not os.path.exists(config_file):
        logger.error("LoadYAML", f"Configuration file '{config_file}' does not exist")
        sys.exit(1)
    return load_config(config_file)


def handle_inference_message(payload):
    global model_name
    inference_result = InferenceResult()
    inference_result.ParseFromString(payload)

    if publish_by != inference_result.publish_by:
        return

    index = inference_result.frame_number
    model_name = inference_result.model_name
    results = inference_result.results

    mqtt_cache.add_inference(results, index)


def handle_stream_message(payload):
    video_frame = VideoFrame()
    video_frame.ParseFromString(payload)

    if publish_by != video_frame.publish_by:
        return

    index = video_frame.frame_number
    width = video_frame.frame_width
    height = video_frame.frame_height
    channels = video_frame.frame_channels
    data = video_frame.frame_raw_data
    format = video_frame.frame_format

    image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, channels)
    if format == 1:  # Convert RGB to BGR
        image = image[:, :, ::-1]

    mqtt_cache.add_image(image, index)


def message_handler(topic, payload, payload_len):
    if topic == infer_topic:
        handle_inference_message(payload)
    elif topic == stream_topic:
        handle_stream_message(payload)


def adjust_wait_time(cache_size, min_wait=1, max_wait=100, threshold_high=20, threshold_low=5):
    if cache_size > threshold_high:
        return min_wait
    elif cache_size < threshold_low:
        return max_wait
    else:
        return int(max_wait - (cache_size - threshold_low) / (threshold_high - threshold_low) * (max_wait - min_wait))



def process_inference(image, inference, config):
    """Process inference results and return the updated image."""
    if not config.display.show_inference_info:
        return image

    if model_name == "yolov8n":
        return draw_boxes_with_labels(image, inference, labels)
    elif model_name == "yolov8n_pose":
        return draw_skeletons(image, inference, YoloPose, labels)
    return image



def main_loop(config, fps_process):
    """Main loop that processes the cache and displays images."""
    window_name = f"Display: {model_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create a resizable window
    cv2.resizeWindow(window_name, config.display.width, config.display.height)  # Set an initial window size

    while True:
        cache_size = mqtt_cache.count()
        wait_time = adjust_wait_time(cache_size)

        if cache_size < 10:
            time.sleep(0.01)
            continue

        index = mqtt_cache.get_min_index()

        if mqtt_cache.complete(index):
            image = mqtt_cache.get(index).image
            inference = mqtt_cache.get(index).inference

            # Resize to inference size
            image = cv2.resize(image, (config.inference.width, config.inference.height))

            # Process inference results
            image = process_inference(image, inference, config)

            # FPS processing and display
            fps_process.increment_frame_count()
            if config.display.show_fps:
                fps = fps_process.get_fps()
                image = draw_text_with_opposite_color(image, f"FPS: {fps:.2f}", (10, 30), 1, 1)

            # Display the local time
            if config.display.show_local_time:
                local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                image = draw_text_with_opposite_color(image, local_time, (10, 60), 1, 1)

            # If allow autoresize
            if config.display.auto_resize:
                # Get the current window size and resize the image to fit the window
                window_width, window_height = cv2.getWindowImageRect(window_name)[2:4]

                # Ensure the window size is not too small, otherwise default to a reasonable size
                if window_width < config.display.width or window_height < config.display.height:
                    window_width, window_height = config.display.width, config.display.height

                resized_image = cv2.resize(image, (window_width, window_height))

                # Display the resized image
                cv2.imshow(window_name, resized_image)
            else:
                # Not allow resized
                cv2.imshow(window_name, image)

            key = cv2.waitKey(wait_time) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q' to exit
                break

            mqtt_cache.remove(index)
        else:
            mqtt_cache.remove(index)
            logger.debug("Inference", f"Frame {index} dropped")



def main():
    # Load the configuration file
    args = parse_arguments()
    config = load_yaml(args.config)

    # Setup the MQTTClient
    mqtt_client = MQTTClient(config.mqtt.host, config.mqtt.port, config.client.id)
    mqtt_client.set_message_callback(message_handler)
    if not mqtt_client.connect():
        logger.error("MAIN", "Failed to connect to MQTT broker")
        sys.exit(1)

    # Load labels
    global labels
    labels = read_labels_from_file(config.inference.label_path)

    # Subscribe to topics
    mqtt_client.subscribe(config.topics.stream)
    mqtt_client.subscribe(config.topics.inference)

    # Update global variables
    global infer_topic, stream_topic, publish_by, model_name
    infer_topic = config.topics.inference
    stream_topic = config.topics.stream
    publish_by = config.topics.publish_by

    # Start FPS counting process
    fps_process = FPSCountingProcess()
    fps_process.start()

    logger.info("MAIN", "MQTT client is running. Press ESC or 'q' to exit.")

    try:
        main_loop(config, fps_process)
    except KeyboardInterrupt:
        pass
    finally:
        mqtt_client.disconnect()
        fps_process.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
