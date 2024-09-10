//
// Created by ubuntu on 9/9/24.
//

#ifndef JETSON_INFER_YAML_CONFIG_H
#define JETSON_INFER_YAML_CONFIG_H

#include <string>
#include <vector>

struct BrokerConfig {
    std::string broker_host;
    int broker_port;
    std::string client_id;
    std::string infer_before_topic;
    std::string infer_result_topic;
};

struct ModelConfig {
    std::string model_path;
    std::string model_type;
    int input_width;
    int input_height;
    int input_channels;
    float confidence;
};

struct YamlConfig {
    // MQTT Broker Config
    BrokerConfig broker;

    // Model Config
    ModelConfig model;

    // Enable/Disable debug mode
    bool debug{};
};

// Load YAML configuration file
YamlConfig loadYamlConfig(const std::string &config_file);

#endif //JETSON_INFER_YAML_CONFIG_H
