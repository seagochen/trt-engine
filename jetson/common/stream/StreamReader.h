//
// Created by orlando on 9/25/24.
//

#ifndef STREAMREADER_H
#define STREAMREADER_H

#include <opencv2/opencv.hpp>
#include <chrono>

class StreamReader {

public:
    /**
     * @brief Construct a new Stream Reader object
     * 
     * @param url
     * @param width
     * @param height
     * @param fps
     * @param max_retries
     * @param delay
     */
    StreamReader(const std::string& url, int width, int height, int fps, int max_retries = 5, int delay = 2);

    /**
     * @brief Load a frame from the camera stream or video file, and return it as a cv::Mat object
     * 
     * @return
     */
    cv::Mat readFrame();

    /**
     * @brief Close the camera stream
     */
    void closeStream();

    /**
     * @brief Get the capture object
     * 
     * @return
     */
    cv::VideoCapture& getCapture();

private:

    /**
     * @brief Open the stream from the camera or video file
     * 
     * @param url
     * @return
     */
    bool openStream(const std::string& url);

    /**
     * @brief Reconnect the camera stream
     * 
     * @param url
     */
    void reconnect(const std::string& url);

    std::string url;
    int width;
    int height;
    int max_retries;
    int delay;
    cv::VideoCapture cap;

    std::chrono::duration<double> frame_time{};  // 单帧的时间间隔
    std::chrono::steady_clock::time_point base_time;  // 基准时间
    std::chrono::steady_clock::time_point last_frame_time;  // 上一帧的时间
};

#endif //STREAMREADER_H
