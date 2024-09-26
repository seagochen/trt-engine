#include <opencv2/opencv.hpp>
#include <csignal>
#include <iostream>
#include <filesystem>
#include <map>
#include <string>
#include <memory>
#include <unistd.h>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "common/engine/infer_wrapper.h"
#include "common/utils/fps_counter.hpp"
#include "common/yolo/yolo_def.h"
#include "common/yolo/yolo_visualization.h"
#include "common/yolo/load_labels.h"
#include "common/args/parse_args.hpp"
#include "common/args/sys_signal.hpp"
#include "common/text/text_painter.h"

#include "extend/yaml/config.h"
#include "extend/network/mqtt_client.h"
#include "extend/jsonconvt/yolo_json.h"
#include "extend/protobufs/video_frame.pb.h"
#include "extend/protobufs/inference_result.pb.h"
#include "extend/lockfree/lock_free_queue.hpp"

// Struct to store inference data
struct InferenceData {
    uint64 frame_number;
    std::string publish_by;
    std::shared_ptr<cv::Mat> processed_frame;
    std::vector<Yolo> results;
};

// Global variables
nvinfer1::Dims4 g_input_dims;
nvinfer1::Dims3 g_output_dims;
std::mutex queue_mutex;
std::condition_variable data_ready;
bool stop_processing = false;

// Inference queue
LockFreeQueue<InferenceData> g_inference_queue(1024);

/**
 * @brief Load the model configuration from the YAML file.
 * @param infer The inference wrapper object.
 * @param config_path The path to the YAML file.
 * @return The model configuration.
 */
void initializeModel(InferWrapper& infer, const std::string& config_path) {
    auto model_config = loadModelConfig(config_path);
    std::map<std::string, std::string> tensor_names = {
        {"input", model_config.input.name},
        {"output", model_config.output.name}
    };

    g_input_dims = {model_config.input.dims[0], model_config.input.dims[1], model_config.input.dims[2], model_config.input.dims[3]};
    g_output_dims = {model_config.output.dims[0], model_config.output.dims[1], model_config.output.dims[2]};

    infer.update(model_config.path, tensor_names, g_input_dims, g_output_dims, model_config.max_det);
}

/**
 * @brief Decode the raw output, and return the results in a vector of Yolo objects.
 * @param raw The raw output from the model.
 * @param boxes The number of boxes.
 */
std::vector<Yolo> decode_yolo(float* raw, int boxes, int features) {
    std::vector<Yolo> results;

    for (int i = 0; i < boxes; i++) {
        if (raw[i * features + 4] > 0.0) {
            Yolo result;
            result.lx = int(raw[i * features + 0]);
            result.ly = int(raw[i * features + 1]);
            result.rx = int(raw[i * features + 2]);
            result.ry = int(raw[i * features + 3]);
            result.conf = raw[i * features + 4];
            result.cls = int(raw[i * features + 5]);

            results.push_back(result);
        }
    }
    return results;
}

/**
 * @brief Decode the raw output, and return the results in a vector of YoloPose objects.
 * @param raw The raw output from the model.
 * @param boxes The number of boxes.
 */
std::vector<YoloPose> decode_pose(float* raw, int boxes, int features) {
    std::vector<YoloPose> results;

    for (int i = 0; i < boxes; i++) {
        if (raw[i * features + 4] > 0.0) {
            YoloPose result;
            result.lx = int(raw[i * features + 0]);
            result.ly = int(raw[i * features + 1]);
            result.rx = int(raw[i * features + 2]);
            result.ry = int(raw[i * features + 3]);
            result.conf = raw[i * features + 4];

            for (int j = 0; j < 17; j++) {
                YoloPoint pt;
                pt.x = int(raw[i * features + 5 + j * 3]);
                pt.y = int(raw[i * features + 5 + j * 3 + 1]);
                pt.conf = raw[i * features + 5 + j * 3 + 2];
                result.pts.push_back(pt);
            }

            results.push_back(result);
        }
    }
    return results;
}

/**
 * @brief Process the received buffer from MQTT
 * @param topic The topic the message was received on
 * @param buf The buffer containing the message
 * @param size The size of the buffer
 */
void processReceivedBuffer(const std::string& topic, const void* buf, const size_t size) {
    VideoFrame proto_frame;
    if (!proto_frame.ParseFromArray(buf, size)) {
        LOG_ERROR("MQTT", "Failed to parse the received buffer");
        return;
    }

    // 将protobuf帧转换为OpenCV图像
    cv::Mat recv_frame;
    if (proto_frame.frame_format() == 1) {  // frame_formatが1の場合は、もらった画像はRGB形式になる
        recv_frame = cv::Mat(proto_frame.frame_height(), 
                            proto_frame.frame_width(), 
                            CV_8UC3, 
                            (void*)proto_frame.frame_raw_data().c_str());
    } else {  // もしframe_formatが1以外の場合は、もらった画像は大体BGR形式になる
        recv_frame = cv::Mat(proto_frame.frame_height(), 
                            proto_frame.frame_width(), CV_8UC3, 
                            (void*)proto_frame.frame_raw_data().c_str());

        // BGR形式をRGB形式に変換
        cv::cvtColor(recv_frame, recv_frame, cv::COLOR_RGB2BGR);
    }

    // 画像をリサイズ
    cv::Mat resized_frame;
    cv::resize(recv_frame, resized_frame, cv::Size(g_input_dims.d[3], g_input_dims.d[2]));

    // 推論用データをキューに追加
    InferenceData data;
    data.frame_number = proto_frame.frame_number();
    data.publish_by = proto_frame.publish_by();
    data.processed_frame = std::make_shared<cv::Mat>(resized_frame);
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        g_inference_queue.enqueue(data);
    }
    data_ready.notify_one();  // 推論スレッドに通知
}



/**
 * @brief Function to publish a frame via MQTT
 * @param mqtt The MQTT client object
 * @param mqtt_config The MQTT configuration object
 * @param proto_frame The inference result in protobuf format
 */
bool publishFrameToMQTT(MQTTClient& mqtt, const MQTTConfig& mqtt_config, InferenceResult& proto_frame) {

    // Serialize the frame
    int byte_size = proto_frame.ByteSizeLong();
    std::vector<char> buffer(byte_size);
    if (!proto_frame.SerializeToArray(buffer.data(), byte_size)) {
        LOG_ERROR("JetsonAdapter", "Failed to serialize the frame");
        return false;
    }

    // Publish the frame
    if (!mqtt.publish(mqtt_config.hidden.out_topic, buffer.data(), byte_size)) {
        LOG_ERROR("JetsonAdapter", "Failed to publish the frame to the MQTT broker");
        return false;
    }

    return true;
}

/**
 * @brief Process the inference queue
 * @param infer The inference wrapper object
 * @param model_config The model configuration
 * @param mqtt The MQTT client object for publishing results
 * @param mqtt_config The MQTT configuration for publishing results
 */
void processInferenceQueue(InferWrapper& infer, const ModelConfig& model_config, MQTTClient& mqtt, const MQTTConfig& mqtt_config) {
    while (!stop_processing) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        data_ready.wait(lock, [] { return g_inference_queue.size() > 0 || stop_processing; });

        if (stop_processing) {
            break;
        }

        // 批量处理队列中的数据
        std::vector<InferenceData> results;
        while (g_inference_queue.size() > 0 && results.size() < 4) {
            InferenceData data;
            if (g_inference_queue.dequeue(data)) {
                infer.addImage(*data.processed_frame, true);
                results.push_back(data);
            }
        }

        // 执行推理
        infer.inferObjectDetection(model_config.cls_threshold, model_config.nms_threshold);

        // 处理推理结果
        for (size_t i = 0; i < results.size(); ++i) {

            // もらった結果をJSON文字列に変换
            std::string json_str;
            if (model_config.type == YOLOv8n) {
                auto inference_output = infer.getResults(i, decode_yolo);
                json_str = yolo2json(inference_output);
            } else if (model_config.type == YOLOv8n_POSE) {
                auto inference_output = infer.getResults(i, decode_pose);
                json_str = yoloPose2json(inference_output);
            } else {
                LOG_ERROR("Inference", "Unknown model type");
                stop_processing = true;
            }

            // 結果をInferenceResultにセット
            InferenceResult result;
            result.set_frame_number(results[i].frame_number);
            result.set_publish_by(results[i].publish_by);
            if (model_config.type == YOLOv8n) {
                result.set_model_name("yolov8n");
            } else if (model_config.type == YOLOv8n_POSE) {
                result.set_model_name("yolov8n_pose");
            }
            result.set_results(json_str);

            // 結果をMQTTで送信
            if (!publishFrameToMQTT(mqtt, mqtt_config, result)) {
                stop_processing = true;
            }
        }
    }
}

/**
 * Main function
 */
int main(int argc, char* argv[]) {
    try {
        GOOGLE_PROTOBUF_VERIFY_VERSION;
        auto args = parse_args_v3(argc, argv);
        registerSIGINT();

        // 加载 YAML 配置
        auto yaml_path = args["config"];
        auto mqtt_config = loadMQTTConfig(yaml_path);

        // 定义模型配置
        ModelConfig model_config = loadModelConfig(yaml_path);

        // 创建推理包装器对象
        InferWrapper infer;
        initializeModel(infer, yaml_path);  // 传递 YAML 配置路径

        MQTTClient mqtt(mqtt_config.broker.host, mqtt_config.broker.port, mqtt_config.hidden.id);
        if (!mqtt.connect()) {
            LOG_ERROR("MQTT", "Failed to connect to the broker");
            return EXIT_FAILURE;
        }

        mqtt.setMessageCallback(processReceivedBuffer);
        if (!mqtt.subscribe(mqtt_config.hidden.in_topic)) {
            LOG_ERROR("MQTT", "Failed to subscribe to the input topic");
            return EXIT_FAILURE;
        }

        // 启动推理处理线程，传递 infer 和 model_config
        std::thread inference_thread(processInferenceQueue, std::ref(infer), std::ref(model_config), std::ref(mqtt), std::ref(mqtt_config));

        // 主线程等待 Ctrl+C 信号
        while (getSigStatus() != SIGINT) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // 通知推理线程停止
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stop_processing = true;
        }
        data_ready.notify_all();
        inference_thread.join();

        mqtt.disconnect();
    }
    catch (const std::exception& e) {
        LOG_ERROR("Main", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
