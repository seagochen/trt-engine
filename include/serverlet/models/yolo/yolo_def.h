//
// Created by user on 3/21/25.
//

#ifndef COMBINEDPROJECT_YOLO_DEF_H
#define COMBINEDPROJECT_YOLO_DEF_H

#include <vector>

struct Yolo {
    int lx, ly, rx, ry;
    float conf;
    int cls;
};

struct YoloPoint {
    int x, y;
    float conf;
};

struct YoloPose {
    int lx, ly, rx, ry;
    float conf;
    std::vector<YoloPoint> pts;
};

#endif //COMBINEDPROJECT_YOLO_DEF_H
