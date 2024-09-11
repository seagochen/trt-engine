import argparse
import logging
import os
import sys
import time

from common.image.DisplayHandler import DisplayHandler
from common.mqtt.MQTTClient import MQTTClient
from common.mqtt.MQTTShareCache import MQTTShareCache
from common.yaml.YamlConfig import YamlConfig
from common.yolo.YoloInferenceResult import parse_yolo_str


def setup_logging():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def mqtt_on_message(topic, payload, cache: MQTTShareCache):
    try:
        cache.put(topic=topic, payload=payload)
    except Exception as e:
        logging.error(f"Error processing message: {e}", exc_info=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description="MQTT Client for inference data processing")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to the configuration file")
    return parser.parse_args()

def load_config(config_file):
    if not os.path.exists(config_file):
        logging.error(f"Configuration file '{config_file}' does not exist")
        sys.exit(1)
    return YamlConfig(config_file)

def setup_mqtt_client(config):
    mqtt_config = config.get_mqtt_config()
    return MQTTClient(
        mqtt_config["host"],
        mqtt_config["port"],
        mqtt_config["client_id"],
        mqtt_config.get("username", ""),
        mqtt_config.get("password", "")
    )

def main():
    setup_logging()
    args = parse_arguments()
    config = load_config(args.config)

    mqtt_client = setup_mqtt_client(config)
    shared_cache = MQTTShareCache(config, stakeholder=config.get_inference_config()["inference_stakeholder"])
    mqtt_client.set_message_callback(mqtt_on_message, shared_cache)

    if not mqtt_client.connect():
        logging.error("Failed to connect to MQTT broker")
        sys.exit(1)

    # Subscribe to the topics
    mqtt_config = config.get_mqtt_config()
    mqtt_client.subscribe(mqtt_config["infer_before_topic"])
    mqtt_client.subscribe(mqtt_config["infer_result_topic"])

    # Initialize the display handler
    infer_config = config.get_inference_config()
    display_handler = DisplayHandler(config)

    # Main loop
    logging.info("MQTT client is running. Press ESC to exit.")
    try:
        while True:
            try:
                if not mqtt_client.listen():
                    logging.warning("Error in MQTT message loop. Reconnecting...")
                    if not mqtt_client.connect():
                        logging.error("Failed to reconnect. Exiting.")
                        break

                # Check if there are any ready items in the cache
                if shared_cache.count() > 10:
                    ready_item = shared_cache.pop_when_ready()
                    if ready_item:
                        bounding_boxes = parse_yolo_str(ready_item.results)
                        display_handler.display_frame(ready_item.frame_data, bounding_boxes)

                if display_handler.check_for_exit():
                    logging.info("ESC key pressed. Exiting...")
                    break

            except Exception as e:
                logging.error(f"Unexpected error in MQTT loop: {e}", exc_info=True)
    except KeyboardInterrupt:
        logging.info("Interrupt received. Shutting down...")
    finally:
        mqtt_client.disconnect()
        display_handler.close()

if __name__ == "__main__":
    main()