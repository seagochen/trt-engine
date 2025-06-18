//
// Created by user on 3/21/25.
//

#ifndef INFER_YOLO_DEF_H
#define INFER_YOLO_DEF_H

#include <vector>

/**
 * @struct Yolo
 * @brief Represents a detection result from a YOLO model.
 *
 * This structure holds the bounding box coordinates, confidence score,
 * and class index for a detected object. The virtual destructor ensures
 * proper cleanup when deleting derived objects via a base class pointer.
 *
 * Members:
 *   lx   - Left x-coordinate of the bounding box.
 *   ly   - Top y-coordinate of the bounding box.
 *   rx   - Right x-coordinate of the bounding box.
 *   ry   - Bottom y-coordinate of the bounding box.
 *   conf - Confidence score of the detection.
 *   cls  - Class index of the detected object.
 */
struct Yolo {
    int lx, ly, rx, ry;
    float conf;
    int cls;

    // A virtual destructor is crucial for proper cleanup when
    // deleting derived class objects through a base class pointer.
    virtual ~Yolo() = default;
};

/**
 * @struct YoloPoint
 * @brief Represents a point in the YOLO pose estimation context.
 *
 * This structure holds the x and y coordinates of a point, along with its confidence score.
 * It is used to represent keypoints in pose estimation tasks.
 *
 * Members:
 *   x    - X-coordinate of the point.
 *   y    - Y-coordinate of the point.
 *   conf  - Confidence score of the point.
 */
struct YoloPoint {
    int x, y;
    float conf;
};

/**
 * @brief Represents a YOLO detection result with additional pose keypoints.
 *
 * Inherits basic detection fields (lx, ly, rx, ry, conf, cls) from the Yolo base struct,
 * and adds a vector of keypoints for pose estimation.
 *
 * @see Yolo
 *
 * @struct YoloPose
 * @extends Yolo
 *
 * @var std::vector<YoloPoint> pts
 *   List of keypoints associated with the detected object, used for pose estimation.
 */
struct YoloPose final : Yolo {
    // lx, ly, rx, ry, conf, and cls are inherited from Yolo
    std::vector<YoloPoint> pts;
};

// YoloPose publicly inherits from Yolo
// 禁止其他类继承 YoloPose
// 这样可以确保 YoloPose 只用于 YOLO Pose 任务，避免其他类错误地继承它。
// 该写法在 C++17 中是合法的，且符合 C++ 的继承和多态原则。

#endif //INFER_YOLO_DEF_H
