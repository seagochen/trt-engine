#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

#include "common/engine/infer_yolov8.h"
#include "common/utils/fps_counter.hpp"
#include "common/yolo/yolo_visualization.h"
#include "common/args/parse_args.hpp"
#include "common/yolo/load_labels.h"

#define MODEL_PATH "/opt/models/yolo/yolov8n.dynamic.engine"
#define VIDEO_PATH "/opt/videos/pedestrain_01.mp4"
#define LABEL_FILE "/opt/labels/coco_labels.txt"


int main() {

    InferYoloObjectv8 yolo_infer(
        MODEL_PATH,
        {{"input", "images"}, {"output", "output0"}},
        {1, 3, 640, 640},   // Input dimensions  (N, C, H, W)
        {1, 84, 8400});     // Output dimensions (N, attributes, boxes)

    // Open video
    // cv::VideoCapture cap(VIDEO_PATH);
    // if (!cap.isOpened()) {
    //     std::cerr << "Error: Cannot open video file" << std::endl;
    //     return 1;
    // }

    // Load labels
    auto labels = readLabelsFromFile(LABEL_FILE);




    return 0;
}

