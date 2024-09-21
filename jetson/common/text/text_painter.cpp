#include "text_painter.h"

// 计算图像中某个区域的平均颜色
cv::Scalar calculateAverageColor(const cv::Mat &image, const cv::Rect &bbox) {
    cv::Mat cropped_image = image(bbox);
    cv::Scalar avg_color = cv::mean(cropped_image);
    return avg_color;
}

// 根据背景平均颜色决定前景文字的颜色
cv::Scalar decideTextColor(const cv::Scalar &average_color) {
    double mean_intensity = (average_color[0] + average_color[1] + average_color[2]) / 3.0;
    return (mean_intensity > 127) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);
}

// 在背景平均颜色对比基础上绘制与背景颜色相反的文字
void drawTextWithOppositeColor(cv::Mat &frame, 
                                const std::string &text, 
                                const cv::Point &left_top, 
                                double font_scale, int thickness) {
    // 定义字体
    int font = cv::FONT_HERSHEY_SIMPLEX;

    // 计算文本大小和背景区域的平均颜色
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, font, font_scale, thickness, &baseline);
    cv::Rect bbox(left_top.x, left_top.y - text_size.height, text_size.width, text_size.height);
    cv::Scalar avg_color = calculateAverageColor(frame, bbox);

    // 决定文字颜色
    cv::Scalar text_color = decideTextColor(avg_color);

    // 绘制文本
    cv::putText(frame, text, left_top, font, font_scale, text_color, thickness, cv::LINE_AA);
}

// 在带有背景色的矩形区域中绘制文字
void drawTextWithBackground(cv::Mat &frame, 
                            const std::string &text, 
                            const cv::Point &left_top, 
                            double font_scale, 
                            int thickness, 
                            const cv::Scalar &background_color, 
                            double background_alpha, 
                            int background_padding) {
    // 定义字体
    int font = cv::FONT_HERSHEY_SIMPLEX;

    // 计算文本大小
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(text, font, font_scale, thickness, &baseline);

    // 定义背景矩形
    cv::Rect background_rect(left_top.x - background_padding, 
                             left_top.y - text_size.height - baseline - background_padding, 
                             text_size.width + 2 * background_padding, 
                             text_size.height + 2 * background_padding);

    // 绘制背景矩形
    cv::Mat overlay;
    frame.copyTo(overlay);
    cv::rectangle(overlay, background_rect, background_color, -1);

    // 添加透明度效果
    cv::addWeighted(overlay, background_alpha, frame, 1 - background_alpha, 0, frame);

    // 绘制文字
    cv::putText(frame, text, cv::Point(left_top.x, left_top.y - baseline), font, font_scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
}

// int main() {
//     // 读取图像
//     cv::Mat frame = cv::imread("images/spectrum.jpg");
//     if (frame.empty()) {
//         std::cerr << "Error: Image file not found or unable to load." << std::endl;
//         return -1;
//     }

//     // 绘制具有与背景颜色对比的文字
//     drawTextWithOppositeColor(frame, "Hello, World!", cv::Point(50, 100), 1.0, 2);

//     // 绘制具有背景色的文字
//     drawTextWithBackground(frame, "Hello, OpenCV!", cv::Point(50, 200), 1.0, 2, cv::Scalar(0, 0, 0), 0.5);

//     // 调整图像大小
//     cv::resize(frame, frame, cv::Size(600, 600));

//     // 显示图像
//     cv::imshow("frame", frame);
//     cv::waitKey(0);

//     // 释放资源
//     cv::destroyAllWindows();

//     return 0;
// }
