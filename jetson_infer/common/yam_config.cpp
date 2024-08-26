//
// Created by ubuntu on 9/9/24.
//

#include "common/yaml_config.h"
#include <yaml-cpp/yaml.h>


YamlConfig loadYamlConfig(const std::string &config_file) {
    YAML::Node config = YAML::LoadFile(config_file);

    YamlConfig frame_config;

    // Set the MQTT configuration
    frame_config.broker.broker_host = config["mqtt"]["broker"]["host"].as<std::string>();
    frame_config.broker.broker_port = config["mqtt"]["broker"]["port"].as<int>();
    frame_config.broker.client_id = config["mqtt"]["broker"]["client_id"].as<std::string>();
    frame_config.broker.inference_topic = config["mqtt"]["broker"]["inference_topic"].as<std::string>();
    frame_config.broker.publish_topic = config["mqtt"]["broker"]["publish_topic"].as<std::string>();

    // Set the model configuration
    frame_config.model.model_path = config["model"]["model_path"].as<std::string>();
    frame_config.model.model_type = config["model"]["model_type"].as<std::string>();
    frame_config.model.label_path = config["model"]["labels_path"].as<std::string>();
    frame_config.model.input_width = config["model"]["input_width"].as<int>();
    frame_config.model.input_height = config["model"]["input_height"].as<int>();
    frame_config.model.input_channels = config["model"]["input_channels"].as<int>();
    frame_config.model.confidence = config["model"]["confidence"].as<float>();

    // Set debug mode
    frame_config.debug = config["debug_mode"].as<bool>();

    return frame_config;
}