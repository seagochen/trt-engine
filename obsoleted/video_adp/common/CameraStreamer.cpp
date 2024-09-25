//
// Created by ubuntu on 9/9/24.
//

#include "CameraStreamer.h"


CameraStream::CameraStream(const std::string& url, int width, int height, int fps, int max_retries, int delay)
        : url(url), width(width), height(height), max_retries(max_retries), delay(delay) {
    openCameraStream(url);
    frame_time = std::chrono::duration<double>(1.0 / fps);  // 以秒为单位设置帧时间间隔
    base_time = std::chrono::steady_clock::now();  // 设置基准时间
}

cv::Mat CameraStream::readFrame() {
    // 获取当前时间
    auto current_time = std::chrono::steady_clock::now();
    // 计算自基准时间以来经过的时间
    std::chrono::duration<double> elapsed = current_time - base_time;

    // 检查是否到了读取下一帧的时间
    if (elapsed >= frame_time) {
        // Read frame from camera
        cv::Mat frame;
        bool ret = cap.read(frame);

        // 更新基准时间，确保精确的帧时间控制
        base_time = current_time;

        if (ret) {
            cv::resize(frame, frame, cv::Size(width, height));
            return frame;
        } else {
            reconnectCamera(url);

            if (!cap.isOpened()) { // 如果重新连接失败，抛出异常
                throw std::runtime_error("Could not reconnect to the camera.");
            }
        }
    }

    auto empty_frame = cv::Mat();  // 返回空帧
    return empty_frame;
}


void CameraStream::closeCameraStream() {
    if (cap.isOpened()) {
        cap.release();
    }
    std::cout << "Camera stream closed." << std::endl;
}

cv::VideoCapture& CameraStream::getCapture() {
    return cap;
}

bool CameraStream::openCameraStream(const std::string& url_link) {
    // If the capture is already opened, release it
    if (cap.isOpened()) {
        cap.release();
    }

    // Attempt to open the camera stream
    cv::VideoCapture temp_cap(url_link);  // Use a temporary object

    if (!temp_cap.isOpened()) {
        std::cerr << "Error: Could not open camera stream." << std::endl;
        return false;  // Return false if unable to open stream
    }

    // If successful, assign the temporary capture object to the class member
    cap = std::move(temp_cap);  // Move temporary to member variable
    std::cout << "Camera stream opened successfully." << std::endl;
    return true;
}

void CameraStream::reconnectCamera(const std::string& url_link) {
    int attempts = 0;
    while (attempts < max_retries) {
        std::cout << "Attempting to reconnect (" << (attempts + 1) << "/" << max_retries << ")..." << std::endl;

        if (openCameraStream(url_link)) {
            std::cout << "Successfully reconnected to the camera." << std::endl;
            return;  // If reconnection is successful, exit the function
        } else {
            std::cout << "Reconnection attempt " << (attempts + 1) << " failed. Retrying in " << delay << " seconds..." << std::endl;
        }

        // Wait for the specified delay before trying again
        std::this_thread::sleep_for(std::chrono::seconds(delay));
        attempts++;
    }

    std::cout << "Max retries reached. Could not reconnect to the camera." << std::endl;
}
