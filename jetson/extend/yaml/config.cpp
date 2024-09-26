//
// Created by ubuntu on 9/9/24.
//

#include "config.h"
#include <yaml-cpp/yaml.h>

// Load MQTT configuration
MQTTConfig loadMQTTConfig(const std::string &yaml) {
    YAML::Node config = YAML::LoadFile(yaml);

    MQTTConfig mqtt_config;

    // Parse broker settings
    mqtt_config.broker.host = config["mqtt"]["broker"]["host"].as<std::string>();
    mqtt_config.broker.port = config["mqtt"]["broker"]["port"].as<int>();

    // Parse clients
    mqtt_config.hidden.id = config["mqtt"]["client_yolo"]["client_id"].as<std::string>();
    mqtt_config.hidden.in_topic = config["mqtt"]["client_yolo"]["in_topic"].as<std::string>();
    mqtt_config.hidden.out_topic = config["mqtt"]["client_yolo"]["out_topic"].as<std::string>();

    mqtt_config.input.id = config["mqtt"]["client_stream"]["client_id"].as<std::string>();
    mqtt_config.input.out_topic = config["mqtt"]["client_stream"]["out_topic"].as<std::string>();

    mqtt_config.output.id = config["mqtt"]["client_display"]["client_id"].as<std::string>();
    mqtt_config.output.in_topic = config["mqtt"]["client_display"]["in_topic"].as<std::string>();

    return mqtt_config;
}

// Load model configuration
ModelConfig loadModelConfig(const std::string &yaml) {
    YAML::Node config = YAML::LoadFile(yaml);

    ModelConfig model_config;

    // Set default values
    model_config.type = YOLOv8n;

    // Parse model type
    auto model_type_str = config["model"]["model_type"].as<std::string>();
    if (model_type_str == "yolov8n") {
        model_config.type = YOLOv8n;
    } else if (model_type_str == "yolov8n_pose") {
        model_config.type = YOLOv8n_POSE;
    }

    // Parse model path and tensor information
    model_config.path = config["model"]["model_path"].as<std::string>();
    model_config.input.name = config["model"]["intput_name"].as<std::string>();
    model_config.input.dims = config["model"]["input_shape"].as<std::vector<int>>();
    model_config.output.name = config["model"]["output_name"].as<std::string>();
    model_config.output.dims = config["model"]["output_shape"].as<std::vector<int>>();

    // Parse thresholds
    model_config.max_det = config["model"]["max_detections"].as<int>();
    model_config.cls_threshold = config["model"]["cls_threshold"].as<float>();
    model_config.nms_threshold = config["model"]["nms_threshold"].as<float>();

    // Parse debug mode
    model_config.enable_debug = config["model"]["debug_mode"].as<bool>();

    return model_config;
}

// Load streamer configuration
StreamConfig loadStreamConfig(const std::string &yaml) {
    YAML::Node config = YAML::LoadFile(yaml);

    StreamConfig stream_config;

    // Parse stream source settings
    stream_config.url = config["stream"]["source"]["url"].as<std::string>();
    stream_config.width = config["stream"]["source"]["width"].as<int>();
    stream_config.height = config["stream"]["source"]["height"].as<int>();
    stream_config.fps = config["stream"]["source"]["fps"].as<int>();
    stream_config.format = config["stream"]["source"]["format"].as<int>();

    // Parse recording settings
    stream_config.record.enable = config["stream"]["record"]["enable"].as<bool>();
    stream_config.record.filename = config["stream"]["record"]["filename"].as<std::string>();

    // Parse debug mode
    stream_config.enable_debug = config["stream"]["debug_mode"].as<bool>();

    return stream_config;
}

// Load display configuration
DisplayConfig loadDisplayConfig(const std::string &yaml) {
    YAML::Node config = YAML::LoadFile(yaml);

    DisplayConfig display_config;

    // Parse resolution
    display_config.width = config["display"]["resolution"]["width"].as<int>();
    display_config.height = config["display"]["resolution"]["height"].as<int>();

    // Parse window settings
    display_config.enable_fullscreen = config["display"]["window"]["enable_fullscreen"].as<bool>();
    display_config.enable_inference_info = config["display"]["window"]["enable_inference_info"].as<bool>();
    display_config.enable_fps_info = config["display"]["window"]["enable_fps_info"].as<bool>();
    display_config.enable_local_time = config["display"]["window"]["enable_local_time"].as<bool>();

    // Parse recording settings
    display_config.record.enable = config["display"]["record"]["enable"].as<bool>();
    display_config.record.filename = config["display"]["record"]["filename"].as<std::string>();

    // Parse debug mode
    display_config.enable_debug = config["display"]["debug_mode"].as<bool>();

    return display_config;
}

