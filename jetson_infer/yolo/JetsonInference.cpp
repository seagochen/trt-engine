//
// Created by ubuntu on 9/10/24.
//

#include "yolo/JetsonInference.h"
#include "common/infer/yolov8.h"

#include <iostream>
#include <csignal>

// 静的メンバーの定義
VideoFrame JetsonInference::proto_frame;
InferenceResult JetsonInference::proto_result;
cv::Mat JetsonInference::resized_frame;

JetsonInference::JetsonInference(const std::string& config_path):
        config(loadYamlConfig(config_path)),
        engine(loadEngineFromFile(config.model.model_path)),
        context(createExecutionContext(engine)),
        gpu_tensors(loadTensorsFromModel(engine)),
        mqtt(config.broker.broker_host, config.broker.broker_port, config.broker.client_id) {

    if (!engine) {
        throw std::runtime_error("Failed to load TensorRT engine.");
    }

    initCudaTemporaryBuffer(config.model.input_width, config.model.input_height, config.model.input_channels);

    if (!mqtt.connect()) {
        throw std::runtime_error("Failed to connect to the MQTT broker");
    }

    mqtt.subscribe(config.broker.infer_before_topic);
    mqtt.setMessageCallback([this](const std::string& topic, const void* payload, size_t len) {
        this->processMessage(payload, len);
    });
}

void JetsonInference::run() {
    while (is_running) {
        if (!mqtt.listen(10)) {
            std::cerr << "Error in MQTT listen." << std::endl;
            break;
        }
    }
}

void JetsonInference::signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    is_running = false;
}

void JetsonInference::processMessage(const void* payload, size_t len) {
    if (!proto_frame.ParseFromArray(payload, static_cast<int>(len))) {
        std::cerr << "Failed to parse VideoFrame protobuf message" << std::endl;
        return;
    }

    receiveDataFrame(proto_frame, resized_frame);
    preprocess(resized_frame, gpu_tensors["images"]);
    inference(context, gpu_tensors["images"], gpu_tensors["output0"]);
    postprocess_(results);
    sendResults(results);

    if (config.debug) {
        displayResults(resized_frame, results);
    }
}

void JetsonInference::receiveDataFrame(const VideoFrame& frame, cv::Mat &resized) {
    cv::Mat frame_mat = cv::Mat(frame.frame_height(), frame.frame_width(),
                                (frame.frame_channels() == 3 ? CV_8UC3 : CV_8UC1),
                                const_cast<char*>(frame.frame_raw_data().data()));
    if (!frame.frame_bgr_color() && frame.frame_channels() == 3) {
        cv::cvtColor(frame_mat, frame_mat, cv::COLOR_RGB2BGR);
    }
    cv::resize(frame_mat, resized, cv::Size(config.model.input_width, config.model.input_height));
}

void JetsonInference::postprocess_(std::vector<YoloResult>& _results) {

    postprocess(gpu_tensors["output0"], _results, config.model.confidence, config.model.model_type);

//    if (config.model.model_type == "yolov8") {
//        obj_postprocess(, config.model.confidence, _results);
//    } else if (config.model.model_type == "yolov8-pose") {
//        pose_postprocess(gpu_tensors["output0"], config.model.confidence, _results);
//    } else {
//        throw std::runtime_error("Unsupported model type: " + config.model.model_type);
//    }
}

void JetsonInference::displayResults(cv::Mat &mat, std::vector<YoloResult> _results) {
    for (const auto& result : _results) {
        cv::rectangle(mat, cv::Point(result.lx, result.ly), cv::Point(result.rx, result.ry), cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("Jetson Inference Debug Mode", mat);
    cv::waitKey(1);
}

void JetsonInference::sendResults(const std::vector<YoloResult> &_results) {
    std::string result_str = to_json(_results);

    // Assign the result to the proto_result
    proto_result.set_frame_number(proto_frame.frame_number());
    proto_result.set_publish_by(proto_frame.publish_by());
    proto_result.set_model_name(config.model.model_type);
    proto_result.set_results(result_str);

    // Serialize the proto_result
    std::string serialized_result;
    if (!proto_result.SerializeToString(&serialized_result)) {
        std::cerr << "Failed to serialize InferenceResult protobuf message" << std::endl;
        return;
    }

    // Publish the serialized result to the MQTT broker
    if (!mqtt.publish(config.broker.infer_result_topic, serialized_result.c_str(), serialized_result.size())) {
        std::cerr << "Failed to publish the inference result" << std::endl;
    }
}