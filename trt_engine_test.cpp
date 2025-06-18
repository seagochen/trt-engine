//
// Created by user on 6/13/25.
//

#include <opencv2/opencv.hpp>
#include "include/serverlet/models/inference/infer_yolo_v8.hpp"
#include "serverlet/models/common/yolo_drawer.h"

// infer_yolo_v8.hpp
//
// This file consolidates all YOLOv8 inference related functionalities,
// including the templated InferYoloV8 class and the sct_yolo_post_proc
// utility function, into a single header-only file.
//
//
// Define simple lambdas or free functions for conversion
// auto obj_converter = [](const std::vector<float>& input, std::vector<Yolo>& output, int features, int results) {
//     host_xywh_to_xyxy_yolo(input, output, features, results);
// };
//
// auto pose_converter = [](const std::vector<float>& input, std::vector<YoloPose>& output, int features, int results) {
//     host_xywh_to_xyxy_pose(input, output, features, results);
// };
//
// // Instantiate for object detection
// InferYoloV8<Yolo, decltype(obj_converter)> yolo_obj_detector(
//     "path/to/obj_engine.trt",
//     1, // maximum_batch
//     100, // maximum_items
//     84, // infer_features
//     {{"output0", {1, 84, 8400}}}, // output_tensor_defs
//     obj_converter
// );
//
// // Instantiate for pose estimation
// InferYoloV8<YoloPose, decltype(pose_converter)> yolo_pose_estimator(
//     "path/to/pose_engine.trt",
//     1, // maximum_batch
//     100, // maximum_items
//     56, // infer_features
//     {{"output0", {1, 56, 8400}}}, // output_tensor_defs
//     pose_converter
// );

int main()
{

    // Load the YOLO model
    // auto model = InferYoloV8Pose("/opt/models/yolov8s-pose.engine", 4);
    //
    // // Load an image
    // cv::Mat image3 = cv::imread("/opt/images/human_and_pets.png");
    //
    //
    // // Perform inference on the images
    // model.preprocess(image3, 0);
    //
    // // Run inference
    // model.inference();
    //
    // // Get the results
    // auto res1 = model.postprocess(0, 0.3, 0.2);
    //
    // // Print out the results
    // for (const auto& item : res1) {
    //     std::cout << "Left: (" << item.lx << ", " << item.ly << "), "
    //               << "Right: (" << item.rx << ", " << item.ry << "), "
    //               << "Confidence: " << item.conf << ", "
    //               << "Class: " << item.cls << std::endl;
    // }

    return 0;

}