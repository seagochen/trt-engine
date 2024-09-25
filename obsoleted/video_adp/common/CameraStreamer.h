//
// Created by ubuntu on 9/9/24.
//

#ifndef VIDEO_ADP_CAMERASTREAMER_H
#define VIDEO_ADP_CAMERASTREAMER_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>

class CameraStream {

public:
    CameraStream(const std::string& url, int width, int height, int fps, int max_retries = 5, int delay = 2);

    cv::Mat readFrame();

    void closeCameraStream();

    cv::VideoCapture& getCapture();

private:
    bool openCameraStream(const std::string& url);

    void reconnectCamera(const std::string& url);

    std::string url;
    int width;
    int height;
    int max_retries;
    int delay;
    cv::VideoCapture cap;

    // 修改 frame_time 为 std::chrono::duration<double>
    std::chrono::duration<double> frame_time{};  // 单帧的时间间隔
    std::chrono::steady_clock::time_point base_time;  // 基准时间
    std::chrono::steady_clock::time_point last_frame_time;  // 上一帧的时间
};


#endif //VIDEO_ADP_CAMERASTREAMER_H
