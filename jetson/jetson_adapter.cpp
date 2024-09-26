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

#include "common/stream/video_maker.h"
#include "common/stream/stream_reader.h"
#include "common/args/sys_signal.hpp"
#include "common/args/parse_args.hpp"
#include "common/utils/logger.h"

#include "extend/yaml/config.h"
#include "extend/network/mqtt_client.h"

#include "extend/protobufs/video_frame.pb.h"
#include "extend/protobufs/inference_result.pb.h"

#define CONFIG_PATH "./res/app.config.yaml"

// Function to handle protobuf frame creation
VideoFrame createProtoFrame(const cv::Mat& frame, uint64 frame_no, const StreamConfig& stream_config, const MQTTConfig& mqtt_config) {
    VideoFrame proto_frame;
    proto_frame.set_frame_number(frame_no);
    proto_frame.set_frame_width(stream_config.width);
    proto_frame.set_frame_height(stream_config.height);
    proto_frame.set_frame_format(stream_config.format);
    proto_frame.set_frame_channels(frame.channels());
    proto_frame.set_publish_by(mqtt_config.input.id);
    proto_frame.set_frame_raw_data(frame.data, frame.total() * frame.elemSize());

    return proto_frame;
}

// Function to publish a frame via MQTT
bool publishFrameToMQTT(MQTTClient& mqtt, const MQTTConfig& mqtt_config, VideoFrame& proto_frame, std::vector<uint8_t>& buffer) {
    int byte_size = proto_frame.ByteSizeLong();
    
    if (buffer.size() < static_cast<size_t>(byte_size)) {
        buffer.resize(byte_size);
    }

    if (!proto_frame.SerializeToArray(buffer.data(), byte_size)) {
        LOG_WARNING("JetsonAdapter", "Failed to serialize the protobuf frame.");
        return false;
    }

    if (!mqtt.publish(mqtt_config.input.out_topic, buffer.data(), byte_size)) {
        LOG_ERROR("JetsonAdapter", "Failed to publish the frame to the MQTT broker");
        return false;
    }

    return true;
}


// Function to handle the main streaming and processing loop
void processFrames(StreamReader& reader, MQTTClient& mqtt, const MQTTConfig& mqtt_config, const StreamConfig& stream_config, VideoMaker* maker) {
    uint64 frame_no = 0;
    std::vector<uint8_t> buffer;

    while (getSigStatus() != SIGINT) {
        try {
            // Get the next frame
            cv::Mat frame = reader.readFrame();
            if (frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // Create protobuf frame
            VideoFrame proto_frame = createProtoFrame(frame, frame_no++, stream_config, mqtt_config);

            // Publish the frame via MQTT
            if (!publishFrameToMQTT(mqtt, mqtt_config, proto_frame, buffer)) {
                continue;
            }

            // Display the frame if debug mode is enabled
            if (stream_config.enable_debug) {
                cv::imshow("Debug Mode", frame);
                if (cv::waitKey(1) == 27 || cv::waitKey(1) == 'q') { // Break on ESC or 'q'
                    LOG_WARNING("JetsonAdapter", "Debug mode interrupted by user");
                    break;
                }
            }

            // Write frame to video if enabled
            if (stream_config.record.enable) {
                maker->addFrame(frame);
            }
        }
        catch (const std::exception& e) {
            LOG_ERROR("FrameProcessing", e.what());
        }
    }

    LOG_VERBOSE("JetsonAdapter", "Quit the loop...");
}

int main(int argc, char** argv) {
    try {
        // Initialize protobuf
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        // Check the input arguments
        auto args = parse_args_v3(argc, argv);

        // Register Ctrl+C signal handler
        registerSIGINT();

        // Load the YAML configuration
        auto yaml_path = args.find("config") != args.end() ? args["config"] : CONFIG_PATH;
        auto mqtt_config = loadMQTTConfig(yaml_path);
        auto stream_config = loadStreamConfig(yaml_path);

        // Initialize MQTT client and connect
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
                stream_config.height
            );
        }

        // Start frame processing
        processFrames(streamer, mqtt, mqtt_config, stream_config, maker.get());

        // Clean up
        if (maker) maker->release();
        streamer.closeStream();
        mqtt.disconnect();
        cv::destroyAllWindows();
    }
    catch (const std::exception& e) {
        LOG_ERROR("Main", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
