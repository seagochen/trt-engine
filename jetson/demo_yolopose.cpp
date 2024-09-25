#include <opencv2/opencv.hpp>
#include <csignal>
#include <iostream>
#include <filesystem>
#include <chrono>

#include "common/engine/infer_wrapper.h"
#include "common/utils/fps_counter.h"
#include "common/yolo/yolo_def.h"
#include "common/yolo/yolo_visualization.h"
#include "common/stream/stream_reader.h"
#include "common/args/parse_args.hpp"
#include "common/args/sys_signal.hpp"
#include "common/text/text_painter.h"


#define MODEL_PATH "/opt/models/yolov8n-pose.dynamic.engine"
#define VIDEO_APTH "/opt/videos/football_01.mp4"


std::vector<YoloPose> decode(float* raw, int boxes, int features) {

    std::vector<YoloPose> results;

    for (int i = 0; i < boxes; i++) {
        if (raw[i * features + 4] > 0.0) {
            YoloPose result;
            result.lx = raw[i * features + 0];
            result.ly = raw[i * features + 1];
            result.rx = raw[i * features + 2];
            result.ry = raw[i * features + 3];
            result.conf = raw[i * features + 4];

            for (int j = 0; j < 17; j++) { // YOLO-POSE can detect 17 pts
                YoloPoint pt;
                
                pt.x = raw[i * features + 5 + j * 3 + 0];
                pt.y = raw[i * features + 5 + j * 3 + 1];
                pt.conf = raw[i * features + 5 + j * 3 + 2];

                result.pts.push_back(pt);
            }
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
    InferWrapper infer(model_path,
    {
        {"input", "images"},
        {"output", "output0"}
    },
    {1, 3, 640, 640},
    {1, 56, 8400});

    // Load the video file
    auto video_path = args.find("video") != args.end() ? args["video"] : VIDEO_APTH;
    StreamReader reader(video_path, 640, 640, 25);

    // Create the FPS counter
    FPSCounter counter;

    cv::Mat frame;
    while (signal_received != SIGINT) {

        // Read frames from each video
        frame = reader.readFrame();
        if (frame.empty()) {
            continue;   // Skip the empty frames
        }

        // Preprocess all frames before inference
        infer.addImage(frame);

        // Perform inference once for all preprocessed frames
        infer.inferPoseEstimation(0.4, 0.2);

        // Get the results for
        auto results = infer.getResults(0, decode);

        // Draw boxes on the corresponding frame
        drawBoxes(frame, results);
        drawSkeletons(frame, results, true, false);

        // Draw skeletons on the corresponding frame
        drawSkeletons(frame, results);

        // Calculate FPS
        counter.countFrames();

        // Calculate FPS
        std::string fps_text = "FPS: " + std::to_string(int(counter.getFPS()));
        cv::putText(frame, fps_text, cv::Point(500, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 191, 255), 2);

        // Show the combined canvas
        cv::imshow("Combined Video", frame);

        // Break on ESC key press
        if (cv::waitKey(10) == 27) {
            signal_received = SIGINT;
        }
    }

    // Release the video captures
    reader.closeStream();

    // Destroy all windows
    cv::destroyAllWindows();

    return 0;
}
