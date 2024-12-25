//
// Created by orlando on 9/25/24.
//

#ifndef IMAGE_ADJUST_HPP
#define IMAGE_ADJUST_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>
#include <cmath>

cv::Mat adjustContrastBrightness(const cv::Mat& image, double contrast = 1.0, double brightness = 0.0) {
    // Adjust contrast and brightness
    cv::Mat adjusted;
    image.convertTo(adjusted, -1, contrast, brightness);
    return adjusted;
}

cv::Mat adjustGamma(const cv::Mat& image, double gamma = 1.0) {
    // Adjust Gamma
    cv::Mat adjusted;
    double invGamma = 1.0 / gamma;

    // Build the lookup table
    cv::Mat table(1, 256, CV_8U);
    uchar* p = table.ptr();
    for (int i = 0; i < 256; ++i) {
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, invGamma) * 255.0);
    }

    // Apply the gamma correction using the lookup table
    cv::LUT(image, table, adjusted);
    return adjusted;
}

cv::Mat sharpenImage(const cv::Mat& image, double alpha = 1.5) {
    // Ensure alpha is positive
    if (alpha <= 0) {
        throw std::invalid_argument("Alpha must be positive.");
    }

    // Create a kernel for sharpening
    cv::Mat kernel = (cv::Mat_<double>(3, 3) << 0, -1, 0,
            -1, 5 + alpha, -1,
            0, -1, 0);

    // Apply the kernel to the image
    cv::Mat sharpened_image;
    cv::filter2D(image, sharpened_image, -1, kernel);

    return sharpened_image;
}

cv::Mat smoothImage(const cv::Mat& image, int ksize = 5) {
    // Ensure kernel size is odd and greater than 1
    if (ksize % 2 == 0 || ksize < 1) {
        throw std::invalid_argument("Kernel size must be an odd number greater than 1.");
    }

    // Apply Gaussian blur to the image
    cv::Mat smoothed_image;
    cv::GaussianBlur(image, smoothed_image, cv::Size(ksize, ksize), 0);

    return smoothed_image;
}

#endif //IMAGE_ADJUST_HPP
