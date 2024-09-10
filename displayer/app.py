#!/usr/bin/env python
# coding: utf-8

import os
import sys
from common.yaml.YamlConfig import YamlConfig
from common.mqtt.MQTTClient import MQTTClient
from common.mqtt.MQTTShareCache import MQTTShareCache



def mqtt_on_message(topic, payload, cache: MQTTShareCache):

    # Handle incoming frames
    cache.put(topic=topic, payload=payload)

def main():
    # Use command-line argument for config file if provided, otherwise use default
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config.yaml"

    # Check if the configuration file exists
    if not os.path.exists(config_file):
        print(f"Configuration file '{config_file}' does not exist")
        sys.exit(1)

    # Load the configuration file
    config = YamlConfig(config_file)

    # Set the MQTT client
    mqtt_config = config.get_mqtt_config()
    mqtt_client = MQTTClient(
        mqtt_config["host"],
        mqtt_config["port"],
        mqtt_config["client_id"],
        mqtt_config.get("username", ""),
        mqtt_config.get("password", "")
    )

    # Set the MQTT message callback and shared cache
    shared_cache = MQTTShareCache(config,
                                  stakeholder=config.get_inference_config()["inference_stakeholder"])
    mqtt_client.set_message_callback(mqtt_on_message, shared_cache)

    if not mqtt_client.connect():
        print("Failed to connect to MQTT broker")
        sys.exit(1)

    mqtt_client.subscribe(mqtt_config["infer_before_topic"])
    mqtt_client.subscribe(mqtt_config["infer_result_topic"])

    print("MQTT client is running. Press Ctrl+C to exit.")
    try:
        while True:
            if not mqtt_client.listen():
                print("Error in MQTT message loop. Reconnecting...")
                if not mqtt_client.connect():
                    print("Failed to reconnect. Exiting.")
                    break
    except KeyboardInterrupt:
        print("Interrupt received. Shutting down...")
    finally:
        mqtt_client.disconnect()


if __name__ == "__main__":
    main()