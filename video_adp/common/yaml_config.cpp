//
// Created by ubuntu on 9/9/24.
//

#include "yaml_config.h"
#include <yaml-cpp/yaml.h>

YamlConfig loadYamlConfig(const std::string &config_file) {
    YAML::Node config = YAML::LoadFile(config_file);

    YamlConfig frame_config;

    // Set MQTT broker configuration
    frame_config.broker.broker_host = config["mqtt"]["broker"]["host"].as<std::string>();
    frame_config.broker.broker_port = config["mqtt"]["broker"]["port"].as<int>();
    frame_config.broker.client_id = config["mqtt"]["broker"]["client_id"].as<std::string>();
    frame_config.broker.infer_before_topic = config["mqtt"]["broker"]["infer_before_topic"].as<std::string>();

    // Set camera configuration
    frame_config.camera.source_url = config["camera"]["source"]["url"].as<std::string>();
    frame_config.camera.source_width = config["camera"]["source"]["width"].as<int>();
    frame_config.camera.source_height = config["camera"]["source"]["height"].as<int>();
    frame_config.camera.source_fps = config["camera"]["source"]["fps"].as<int>();

    // Set record configuration
    frame_config.record.record_enable = config["camera"]["record"]["enable"].as<bool>();
    frame_config.record.filename = config["camera"]["record"]["filename"].as<std::string>();

    // Set preprocess configuration
    frame_config.preprocess.preprocess_enable = config["camera"]["preprocess"]["enable"].as<bool>();
    frame_config.preprocess.gamma = config["camera"]["preprocess"]["gamma"].as<double>();
    frame_config.preprocess.contrast = config["camera"]["preprocess"]["contrast"].as<double>();
    frame_config.preprocess.brightness = config["camera"]["preprocess"]["brightness"].as<double>();
    frame_config.preprocess.color_space = config["camera"]["preprocess"]["colorspace"].as<std::string>();

    // Set debug mode
    frame_config.debug = config["debug_mode"].as<bool>();

    return frame_config;
}