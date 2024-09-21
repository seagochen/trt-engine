//
// Created by orlando on 9/25/24.
//

#ifndef VIDEOMAKER_H
#define VIDEOMAKER_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <ctime>
#include <iomanip>

class VideoMaker {
public:
    explicit VideoMaker(cv::VideoCapture& cap, const std::string& name = "", int width = 0, int height = 0, int fps = 0);

    void initializeVideoWriter();

    void addFrame(const cv::Mat& frame);

    void release();

private:
    std::string getCurrentTimestamp();

    cv::VideoCapture& cap;
    cv::VideoWriter video;
    std::string name;
    int fps;
    int width;
    int height;
};

#endif //VIDEOMAKER_H
