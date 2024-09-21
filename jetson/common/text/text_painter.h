//
// Created by orlando on 9/25/24.
//

#ifndef TEXT_PAINTER_H
#define TEXT_PAINTER_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>


void drawTextWithBackground(cv::Mat &frame,
                            const std::string &text,
                            const cv::Point &left_top, 
                            double font_scale = 1.0,
                            int thickness = 2, 
                            const cv::Scalar &background_color = cv::Scalar(0, 0, 0), 
                            double background_alpha = 0.5, 
                            int background_padding = 5);


void drawTextWithOppositeColor(cv::Mat &frame,
                            const std::string &text, 
                            const cv::Point &left_top, 
                            double font_scale = 1.0, 
                            int thickness = 2);

#endif //TEXT_PAINTER_H