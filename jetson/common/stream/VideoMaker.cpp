//
// Created by orlando on 9/25/24.
//

#include "VideoMaker.h"

VideoMaker::VideoMaker(cv::VideoCapture& cap, const std::string& name, int fps, int width, int height)
        : cap(cap) {
    if (!cap.isOpened()) {
        throw std::invalid_argument("Invalid video capture object");
    }

    // Check if the name has an extension
    size_t dot_pos = name.find_last_of('.');
    if (dot_pos != std::string::npos) {
        // Remove the extension
        this->name = name.substr(0, dot_pos);
    } else {
        this->name = name;
    }

    this->name = this->name + "_" + getCurrentTimestamp();

    // Get or set FPS
    this->fps = (fps == 0) ? static_cast<int>(cap.get(cv::CAP_PROP_FPS)) : fps;

    // Get or set width and height
    this->width = (width == 0) ? static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)) : width;
    this->height = (height == 0) ? static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)) : height;

    // Initialize VideoWriter
    initializeVideoWriter();
}

void VideoMaker::addFrame(const cv::Mat& frame) {
    if (frame.empty()) {
        std::cerr << "Empty frame detected, skipping." << std::endl;
        return;
    }

    if (frame.cols != this->width || frame.rows != this->height) {
        // Resize the frame to match the video size instead of throwing an exception
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(this->width, this->height));
        this->video.write(resized_frame);
    } else {
        // Write frame to the video
        this->video.write(frame);
    }
}

void VideoMaker::release() {
    std::cout << "Closing video file: " << this->name + ".mp4" << std::endl;
    this->video.release();
}

std::string VideoMaker::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now = *std::localtime(&now_time);

    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

void VideoMaker::initializeVideoWriter() {
    // Try to open video writer with MP4 codec
    std::string video_name = this->name + ".mp4";
    bool initialized = video.open(video_name, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), this->fps, cv::Size(this->width, this->height));

    // If MP4 fails, fallback to AVI codec
    if (!initialized) {
        std::cerr << "Failed to initialize MP4, falling back to AVI" << std::endl;
        video_name = this->name + ".avi";
        initialized = video.open(video_name, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), this->fps, cv::Size(this->width, this->height));
    }

    // If both codecs fail, throw an exception
    if (!initialized) {
        throw std::runtime_error("Failed to initialize video writer with both MP4 and AVI codecs");
    }

    std::cout << "Video recording started: " << video_name << std::endl;
}