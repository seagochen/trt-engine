#include <iostream>
#include <string>
#include <csignal>
#include <vector>
#include "common/CameraStreamer.h"
#include "common/MQTTClient.h"
#include "common/VideoMaker.h"
#include "common/yaml_config.h"
#include "protobufs/video_frame.pb.h" // For the VideoFrame protobuf

// Global flag to indicate whether the program is running
volatile bool is_running = true;

// Signal handler function for SIGINT (Ctrl+C)
void handleSigint(int sig) {
    std::cout << "Caught signal " << sig << ", exiting gracefully..." << std::endl;
    is_running = false;  // Set flag to false to exit the listen
}

int main() {
    // Register the SIGINT signal handler
    std::signal(SIGINT, handleSigint);

    // 初始化 protobuf
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    try {
        // Load configuration from yaml file
        auto config = loadYamlConfig("config.yaml");

        // 创建一个 VideoFrame 对象并设置字段
        VideoFrame proto_frame;

        // Create a MQTT client
        MQTTClient mqtt(config.broker.broker_host,
                        config.broker.broker_port,
                        config.broker.client_id);
        if (!mqtt.connect()) {
            throw std::runtime_error("Failed to connect to the MQTT broker");
        }

        // Create a camera streamer
        CameraStream camera(config.camera.source_url,
                            config.camera.source_width,
                            config.camera.source_height,
                            config.camera.source_fps);

        // If creating the video is enabled, create a VideoMaker instance
        std::unique_ptr<VideoMaker> maker = nullptr;
        if (config.record.record_enable) {
            maker = std::make_unique<VideoMaker>(camera.getCapture(),
                                                 config.record.filename,
                                                 config.camera.source_fps,
                                                 config.camera.source_width,
                                                 config.camera.source_height);
        }

        // Frame counter
        uint64 frame_no = 0;

        // 预先分配足够的缓冲区，并在后续使用中重复使用该缓冲区
        std::vector<uint8_t> buffer;

        // Main listen manually managing Mosquitto events
        while (is_running) {
            // Handle MQTT events
            if (!mqtt.listen(10)) {  // 每次循环监听10ms
                std::cerr << "Error in MQTT listen." << std::endl;
                break;
            }

            // Get the next frame
            cv::Mat frame = camera.readFrame();

            // Check if the frame is empty
            if (frame.empty()) {
                // Sleep for a short time to avoid high CPU usage
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            // Update the video frame protobuf with the frame data
            proto_frame.set_frame_number(frame_no++);
            proto_frame.set_frame_width(config.camera.source_width);
            proto_frame.set_frame_height(config.camera.source_height);
            proto_frame.set_frame_channels(frame.channels());
            proto_frame.set_frame_bgr_color(true);
            proto_frame.set_publish_by(config.broker.client_id);
            proto_frame.set_frame_raw_data(frame.data, frame.total() * frame.elemSize());

            // 获取序列化后的大小
            int byte_size = proto_frame.ByteSizeLong();

            // 如果当前缓冲区不够大，重新分配
            if (buffer.size() < static_cast<size_t>(byte_size)) {
                buffer.resize(byte_size);
            }

            // 将 protobuf 对象序列化到缓冲区中
            if (!proto_frame.SerializeToArray(buffer.data(), byte_size)) {
                std::cerr << "Failed to serialize the protobuf frame." << std::endl;
                continue;
            }

            // Publish the protobuf frame to the MQTT broker as binary data
            if (!mqtt.publish(config.broker.inference_topic, buffer.data(), byte_size)) {
                throw std::runtime_error("Failed to publish the frame to the MQTT broker");
            }

            // If debug mode is enabled, display the frame
            if (config.debug) {
                cv::imshow("Debug Mode", frame);
            }

            // If creating the video is enabled, write the frame to the video
            if (config.record.record_enable && maker) {
                maker->addFrame(frame);
            }

            // Allow OpenCV to process the GUI events
            cv::waitKey(1);
        }

        // If creating the video is enabled, release the video writer
        if (config.record.record_enable && maker) {
            maker->release();
        }

        // Release the camera and video writer
        camera.closeCameraStream();

        // Close the debug window
        cv::destroyAllWindows();

        // Disconnect from the MQTT broker
        mqtt.disconnect();
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Print a message and exit
    std::cout << "Exiting program..." << std::endl;

    return EXIT_SUCCESS;
}
