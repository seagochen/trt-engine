//
// Created by ubuntu on 9/9/24.
//

#ifndef VIDEO_ADP_YAMLCONFIG_H
#define VIDEO_ADP_YAMLCONFIG_H

#include <string>

struct BrokerConfig {
    std::string broker_host;
    int broker_port;
    std::string client_id;
    std::string infer_before_topic;
};

struct CameraConfig {
    std::string source_url;
    int source_width;
    int source_height;
    int source_fps;
};

struct RecordConfig {
    bool record_enable;
    std::string filename;
};

struct PreprocessConfig {
    bool preprocess_enable;
    double gamma;
    double contrast;
    double brightness;
    std::string color_space;
};

struct YamlConfig {
    // MQTT broker configuration
    BrokerConfig broker;

    // Camera configuration
    CameraConfig camera;

    // Record configuration
    RecordConfig record;

    // Preprocess configuration
    PreprocessConfig preprocess;

    // Enable/Disable debug mode
    bool debug;
};


// Load configuration from yaml file
YamlConfig loadYamlConfig(const std::string &config_file);

#endif //VIDEO_ADP_YAMLCONFIG_H
