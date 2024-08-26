#include <opencv2/opencv.hpp>
#include <csignal>
#include <stdexcept>

#include "yolo/JetsonInference.h"

#include "protobufs/video_frame.pb.h"

int main() {

    try {

        // Initialize Google's logging library.
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        // Register signal handler
        std::signal(SIGINT, JetsonInference::signalHandler);

        // Run the inference
        JetsonInference inference("config.yaml");
        inference.run();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Close all OpenCV windows
    cv::destroyAllWindows();
    return 0;
}
