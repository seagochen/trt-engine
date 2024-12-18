#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

#include "common/engine/infer_yolo_v8.h"
#include "common/utils/fps_counter.hpp"
#include "common/yolo/yolo_visualization.h"
#include "common/args/parse_args.hpp"


#include "common/yolo/load_labels.h"

#define MODEL_PATH "/opt/models/yolo/yolov8n.dynamic.engine"
#define VIDEO_PATH "/opt/videos/pedestrain_01.mp4"
#define LABEL_FILE "/opt/labels/coco_labels.txt"

// Function to decode the raw output from the model
std::vector<Yolo> decode(float* raw, int boxes, int features) {

    std::vector<Yolo> results;

    for (int i = 0; i < boxes; i++) {
        if (raw[i * features + 4] > 0.0) {
            Yolo result;

            result.lx = int(raw[i * features + 0]);
            result.ly = int(raw[i * features + 1]);
            result.rx = int(raw[i * features + 2]);
            result.ry = int(raw[i * features + 3]);
            result.conf = raw[i * features + 4];
            result.cls = int(raw[i * features + 5]);

            results.push_back(result);
        }
    }

    // Print the number of detected objects
    std::cout << "Detected objects: " << results.size() << std::endl;

    return results;
}


int main() {

    InferWrapper yolo_infer(MODEL_PATH,
        {
            {"input", "images"},
            {"output", "output0"},
        },
        {4, 3, 640, 640},
        {4, 84, 8400});

    // Load model
    // yolo_infer.loadEngine(MODEL_PATH,
    //     {{"input", "images"}, {"output", "output0"}},
    //     {4, 3, 640, 640},
    //     {4, 84, 8400});

    // Open video
    cv::VideoCapture cap(VIDEO_PATH);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file" << std::endl;
        return 1;
    }

    // Load labels
    auto labels = readLabelsFromFile(LABEL_FILE);

    FPSCounter fps_counter;
    cv::Mat frame;
    while (true) {
        for (int i = 0; i < 4; ++i) {
            cap >> frame;
            if (frame.empty()) {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Restart video
                continue;
            }

            // Resize frame to 640x640 and add to batch
            cv::resize(frame, frame, cv::Size(640, 640));
            yolo_infer.addImage(frame);
        }

        // Run inference
        yolo_infer.inferObjectDetection(0.4, 0.1);

        // Postprocess results and visualize
        for (int i = 0; i < 4; ++i) {
            auto results = yolo_infer.getResults(i, decode);
            drawBoxes(frame, results, labels);
        }

        // Display FPS
        fps_counter.countFrames();

        // Draw FPS on the image
        cv::putText(frame, "FPS: " + std::to_string(fps_counter.getFPS()), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        // Display the image
        cv::imshow("YoloV8", frame);
        if (cv::waitKey(1) == 27) break; // ESC key to exit
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

