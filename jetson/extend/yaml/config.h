//
// Created by ubuntu on 9/9/24.
//

#ifndef INFER_CONFIG_H
#define INFER_CONFIG_H

#include <string>
#include <vector>

struct Broker {
    std::string host;
    int port;
};

struct HiddenClient {
    std::string id;
    std::string in_topic;
    std::string out_topic;
};

struct InputClient {
    std::string id;
    std::string out_topic;
};

struct OutputClient {
    std::string id;
    std::string in_topic;
};

struct MQTTConfig {
    Broker broker;
    HiddenClient hidden;
    InputClient input;
    OutputClient output;
};

////////////////////////////////////////////////////////////////////////////////////

enum ModelType {
    YOLOv8n = 1,
    YOLOv8n_POSE = 2,
};

struct TensorInfo {
    std::string name;
    std::vector<int> dims;
};

struct ModelConfig {
    std::string path;
    ModelType type;
    TensorInfo input;
    TensorInfo output;
    int max_det;
    float cls_threshold;
    float nms_threshold;
    bool enable_debug;
};

////////////////////////////////////////////////////////////////////////////////////

struct RecordConfig {
    bool enable;
    std::string filename;
};

struct StreamConfig {
    std::string url;
    int width;
    int height;
    int fps;
    int format;
    bool enable_debug;
    RecordConfig record;
};

struct DisplayConfig {
    int width;
    int height;
    int fps;
    bool enable_fullscreen;
    bool enable_inference_info;
    bool enable_fps_info;
    bool enable_local_time;
    bool enable_debug;
    std::string label_path;
    RecordConfig record;
};

////////////////////////////////////////////////////////////////////////////////////

// Load MQTT configuration
MQTTConfig loadMQTTConfig(const std::string &yaml);

// Load model configuration
ModelConfig loadModelConfig(const std::string &yaml);

// Load streamer configuration
StreamConfig loadStreamConfig(const std::string &yaml);

// Load display configuration
DisplayConfig loadDisplayConfig(const std::string &yaml);

#endif //INFER_CONFIG_H
