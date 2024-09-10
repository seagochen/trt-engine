#include <iostream>
#include <string>
#include <csignal>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>  // OpenCV for camera and video processing
#include "common/CameraStreamer.h"
#include "common/MQTTClient.h"
#include "common/VideoMaker.h"
#include "common/yaml_config.h"
#include "protobufs/video_frame.pb.h"  // For the VideoFrame protobuf

namespace fs = std::filesystem;

// Global flag to indicate whether the program is running
volatile bool is_running = true;

// Signal handler function for SIGINT (Ctrl+C)
void handleSigint(int sig) {
    std::cout << "Caught signal " << sig << ", exiting gracefully..." << std::endl;
    is_running = false;
}

// Function to validate command line arguments
std::string validateArguments(int argc, char** argv) {
    if (argc != 2) {
        throw std::invalid_argument("Usage: " + std::string(argv[0]) + " <config.yaml>");
    }

    std::string config_path = argv[1];
    if (!fs::exists(config_path) || !fs::is_regular_file(config_path)) {
        throw std::invalid_argument("Error: The file " + config_path + " does not exist or is not a regular file.");
    }

    return config_path;
}

// Function to handle the main streaming and processing loop
void processFrames(CameraStream& camera, MQTTClient& mqtt, const YamlConfig& config, VideoMaker* maker) {
    VideoFrame proto_frame;  // Protobuf object
    uint64 frame_no = 0;
    std::vector<uint8_t> buffer;  // Pre-allocated buffer

    while (is_running) {
        // Handle MQTT events
        if (!mqtt.listen(10)) {  // 每次循环监听10ms
            std::cerr << "Error in MQTT listen." << std::endl;
            break;
        }

        // Get the next frame
        cv::Mat frame = camera.readFrame();
        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Update protobuf with frame data
        proto_frame.set_frame_number(frame_no++);
        proto_frame.set_frame_width(config.camera.source_width);
        proto_frame.set_frame_height(config.camera.source_height);
        proto_frame.set_frame_channels(frame.channels());
        proto_frame.set_frame_bgr_color(true);
        proto_frame.set_publish_by(config.broker.client_id);
        proto_frame.set_frame_raw_data(frame.data, frame.total() * frame.elemSize());

        // Serialize protobuf to buffer
        int byte_size = proto_frame.ByteSizeLong();
        if (buffer.size() < static_cast<size_t>(byte_size)) {
            buffer.resize(byte_size);
        }

        if (!proto_frame.SerializeToArray(buffer.data(), byte_size)) {
            std::cerr << "Failed to serialize the protobuf frame." << std::endl;
            continue;
        }

        // Publish frame to MQTT broker
        if (!mqtt.publish(config.broker.infer_before_topic, buffer.data(), byte_size)) {
            throw std::runtime_error("Failed to publish the frame to the MQTT broker");
        }

        // Display the frame if debug mode is enabled
        if (config.debug) {
            cv::imshow("Debug Mode", frame);
            cv::waitKey(1);  // Allows OpenCV to process GUI events
        }

        // Write frame to video if enabled
        if (config.record.record_enable && maker) {
            maker->addFrame(frame);
        }
    }
}

// Main function where everything is initialized
int main(int argc, char** argv) {
    // Register the SIGINT signal handler
    std::signal(SIGINT, handleSigint);

    try {
        // Validate and load config file
        std::string config_path = validateArguments(argc, argv);
        auto config = loadYamlConfig(config_path);

        // Initialize protobuf
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        // Create MQTT client and connect
        MQTTClient mqtt(config.broker.broker_host, config.broker.broker_port, config.broker.client_id);
        if (!mqtt.connect()) {
            throw std::runtime_error("Failed to connect to the MQTT broker");
        }

        // Initialize camera streamer
        CameraStream camera(config.camera.source_url, config.camera.source_width, config.camera.source_height, config.camera.source_fps);

        // Create VideoMaker if recording is enabled
        std::unique_ptr<VideoMaker> maker = nullptr;
        if (config.record.record_enable) {
            maker = std::make_unique<VideoMaker>(camera.getCapture(), config.record.filename, config.camera.source_fps, config.camera.source_width, config.camera.source_height);
        }

        // Start processing frames
        processFrames(camera, mqtt, config, maker.get());

        // Clean up
        if (maker) maker->release();
        camera.closeCameraStream();
        mqtt.disconnect();
        cv::destroyAllWindows();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Exiting program..." << std::endl;
    return EXIT_SUCCESS;
}
