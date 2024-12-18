#include <opencv2/opencv.hpp>
#include <csignal>
#include <iostream>
#include <filesystem>
#include <map>
#include <string>
#include <unistd.h>

#include "common/engine/infer_yolo_v8.h"
#include "common/utils/fps_counter.hpp"
#include "common/yolo/yolo_def.h"
#include "common/yolo/yolo_visualization.h"
#include "common/yolo/load_labels.h"
#include "common/args/parse_args.hpp"
#include "common/args/sys_signal.hpp"
#include "common/text/text_painter.h"


#define MODEL_PATH "/opt/models/yolo/yolov8n.dynamic.engine"
#define VIDEO_APTH "/opt/videos/pedestrian_normal_01.mp4"
#define LABEL_FILE "./res/coco_labels.txt"


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

    return results;
}


int main(int argc, char *argv[]) {

    // Parse command line arguments
    auto args = parse_args_v1(argc, argv);

    //　Register Ctrl+C signal handler
    registerSIGINT();

    // Load the TensorRT engine from the serialized engine file
    auto model_path = args.find("model") != args.end() ? args["model"] : MODEL_PATH;
    InferYoloWrapper infer(model_path,
    {
        {"input", "images"},
        {"output", "output0"}
    },
    {4, 3, 640, 640},
    {4, 84, 8400});

    // Load 4 video sources
    auto video_path = args.find("video") != args.end() ? args["video"] : VIDEO_APTH;
    cv::VideoCapture cap1(video_path);
    cv::VideoCapture cap2(video_path);
    cv::VideoCapture cap3(video_path);
    cv::VideoCapture cap4(video_path);

    // Load the labels
    auto labels = args.find("label") != args.end() ? readLabelsFromFile(args["label"]) : readLabelsFromFile(LABEL_FILE);

    // Check if all videos are opened
    if (!cap1.isOpened() || !cap2.isOpened() || !cap3.isOpened() || !cap4.isOpened()) {
        std::cerr << "Error: Cannot open one or more video files" << std::endl;
        return 1;
    }

    // Create the FPS counter
    FPSCounter counter;

    // Cv Frames
    cv::Mat frame1, frame2, frame3, frame4;
    while (true) {
        // Read frames from each video
        cap1 >> frame1;
        cap2 >> frame2;
        cap3 >> frame3;
        cap4 >> frame4;

        // Continue from the beginning
        if (frame1.empty()) {
            if (cap1.isOpened()) {
                cap1.release();
            }

            cap1.open(video_path); 
            continue;
        }

        if (frame2.empty()) {
            if (cap2.isOpened()) {
                cap2.release();
            }

            cap2.open(video_path); 
            continue;
        }

        if (frame3.empty()) {
            if (cap3.isOpened()) {
                cap3.release();
            }

            cap3.open(video_path); 
            continue;
        }

        if (frame4.empty()) {
            if (cap4.isOpened()) {
                cap4.release();
            }

            cap4.open(video_path);
            continue;
        }

        // Resize frames to 640x640 for inference
        cv::resize(frame1, frame1, cv::Size(640, 640));
        cv::resize(frame2, frame2, cv::Size(640, 640));
        cv::resize(frame3, frame3, cv::Size(640, 640));
        cv::resize(frame4, frame4, cv::Size(640, 640));

        // Preprocess all frames before inference
        infer.addImage(frame1);
        infer.addImage(frame2);
        infer.addImage(frame3);
        infer.addImage(frame4);

        // Perform inference once for all preprocessed frames
        infer.inferObjectDetection(0.4, 0.1);

        // Retrieve results and draw boxes on each frame
        std::vector<cv::Mat> frames = {frame1, frame2, frame3, frame4};
        for (int i = 0; i < 4; i++) {
            // Get the results for each frame
            auto results = infer.getResults(i, decode);

            // Draw boxes on the corresponding frame
            drawBoxes(frames[i], results, labels);
        }

        // Resize each frame to 320x320 for displaying
        cv::resize(frame1, frame1, cv::Size(320, 320));
        cv::resize(frame2, frame2, cv::Size(320, 320));
        cv::resize(frame3, frame3, cv::Size(320, 320));
        cv::resize(frame4, frame4, cv::Size(320, 320));

        // Create a 640x640 canvas to place the 4 frames
        cv::Mat canvas(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));

        // Copy each resized frame to the respective quadrant
        frame1.copyTo(canvas(cv::Rect(0, 0, 320, 320)));
        frame2.copyTo(canvas(cv::Rect(320, 0, 320, 320)));
        frame3.copyTo(canvas(cv::Rect(0, 320, 320, 320)));
        frame4.copyTo(canvas(cv::Rect(320, 320, 320, 320)));

        // Calculate FPS
        counter.countFrames();

        // Resize the output canvas to 800x800 for display
        cv::resize(canvas, canvas, cv::Size(800, 800));

        // Display FPS on the top-right corner of the canvas
        std::string fps_text = "FPS: " + std::to_string(int(counter.getFPS()));
        drawTextWithBackground(canvas, 
                    fps_text, // Text to display
                    cv::Point(650, 50), // Left-top corner of the text
                    0.8, // Font scale
                    2, // Thickness
                    cv::Scalar(0, 0, 0)); // Background color

        // Show the combined canvas
        cv::imshow("Combined Video", canvas);

        // Break on ESC key press
        if (cv::waitKey(1) == 27) {
            signal_received = SIGINT;
        }
    }

    // Release the video captures
    cap1.release();
    cap2.release();
    cap3.release();
    cap4.release();

    // Destroy all windows
    cv::destroyAllWindows();

    return 0;
}
