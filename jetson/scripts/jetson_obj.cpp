//
// Created by orlando on 9/26/24.
//

#include <iostream>
#include <string>
#include <csignal>
#include <vector>
#include <chrono>
#include <thread>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "common/args/sys_signal.hpp"
#include "common/args/parse_args.hpp"
#include "common/utils/logger.h"

#include "extend/yaml/config.h"
#include "extend/network/mqtt_client.h"

#include "extend/protobufs/video_frame.pb.h"
#include "extend/protobufs/inference_result.pb.h"


#define CONFIG_PATH "./res/app.config.yaml"


// Function to handle the main streaming and processing loop
void processFrames(StreamReader& reader,
    MQTTClient& mqtt,
    const MQTTConfig& mqtt_config,
    const StreamConfig& stream_config,
    VideoMaker* maker) {

    VideoFrame proto_frame;  // Protobuf object
    uint64 frame_no = 0;
    std::vector<uint8_t> buffer;  // Pre-allocated buffer

    while (getSigStatus() != SIGINT) {

        // Get the next frame
        cv::Mat frame = reader.readFrame();
        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Update protobuf with frame data
        proto_frame.set_frame_number(frame_no++);
        proto_frame.set_frame_width(stream_config.width);
        proto_frame.set_frame_height(stream_config.height);
        proto_frame.set_frame_format(stream_config.format);
        proto_frame.set_frame_channels(frame.channels());
        proto_frame.set_publish_by(mqtt_config.input.id);
        proto_frame.set_frame_raw_data(frame.data, frame.total() * frame.elemSize());

        // Serialize protobuf to buffer
        int byte_size = proto_frame.ByteSizeLong();
        if (buffer.size() < static_cast<size_t>(byte_size)) {
            buffer.resize(byte_size);
        }

        if (!proto_frame.SerializeToArray(buffer.data(), byte_size)) {
            LOG_WARNING("JetsonAdapter", "Failed to serialize the protobuf frame.");
            continue;
        }

        // Publish frame to MQTT broker
        if (!mqtt.publish(mqtt_config.input.out_topic, buffer.data(), byte_size)) {
            throw std::runtime_error("Failed to publish the frame to the MQTT broker");
        }

        // Display the frame if debug mode is enabled
        if (stream_config.enable_debug) {
            cv::imshow("Debug Mode", frame);
            if (cv::waitKey(1) == 27 || cv::waitKey(1) == 'q') { // Break on ESC or 'q'
                break;
            }
        }

        // Write frame to video if enabled
        if (stream_config.record.enable) {
            maker->addFrame(frame);
        }
    }

    LOG_VERBOSE("JetsonAdapter", "Quit the loop by sending Ctrl+C...");
}


int main(int argc, char** argv) {

    // Check the input arguments
    auto args = parse_args_v3(argc, argv);

    // Initialize protobuf
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    //　Register Ctrl+C signal handler
    registerSIGINT();

    // Load the YAML file
    auto yaml_path = args.find("config") != args.end() ? args["config"] : CONFIG_PATH;
    auto mqtt_config = loadMQTTConfig(yaml_path);
    auto stream_config = loadStreamConfig(yaml_path);

    try {

        // Create MQTT client and connect
        MQTTClient mqtt(mqtt_config.broker.host, mqtt_config.broker.port, mqtt_config.input.id);
        if (!mqtt.connect()) {
            LOG_ERROR("JetsonAdapter", "Failed to connect to the MQTT broker");
            return EXIT_FAILURE;
        }

        // Initialize streamer
        StreamReader streamer(stream_config.url, stream_config.width, stream_config.height, stream_config.fps);

        // Create VideoMaker if recording is enabled
        std::unique_ptr<VideoMaker> maker = nullptr;
        if (stream_config.record.enable) {
            maker = std::make_unique<VideoMaker>(
                streamer.getCapture(),
                stream_config.record.filename,
                stream_config.fps,
                stream_config.width,
                stream_config.height);
        }

        // Start processing frames
        processFrames(streamer, mqtt, mqtt_config, stream_config,maker.get());

        // Clean up
        if (maker) maker->release();
        streamer.closeStream();
        mqtt.disconnect();
        cv::destroyAllWindows();
    }
    catch (const std::exception& e) {
        LOG_ERROR("JetsonAdapter", e.what());
        return EXIT_FAILURE;
    }

    // Shutdown protobuf
    return EXIT_SUCCESS;
}
