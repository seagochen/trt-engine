#include <opencv2/opencv.hpp>
#include <csignal>
#include <stdexcept>
#include <iostream>
#include <filesystem>

#include "yolo/JetsonInference.h"

#include "protobufs/video_frame.pb.h"

namespace fs = std::filesystem;


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


int main(int argc, char** argv) {

    try {

        // Initialize Google's logging library.
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        // Register signal handler
        std::signal(SIGINT, JetsonInference::signalHandler);

        // Validate command line arguments
        std::string config_path = validateArguments(argc, argv);

        // Run the inference
        JetsonInference inference(config_path);
        inference.run();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Close all OpenCV windows
    cv::destroyAllWindows();

    // Print a message and exit
    std::cout << "Exiting program..." << std::endl;
    return 0;
}
