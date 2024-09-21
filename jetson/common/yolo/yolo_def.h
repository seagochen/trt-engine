//
// Created by orlando on 9/24/24.
//

#ifndef YOLO_DEF_H
#define YOLO_DEF_H

#include <vector>

struct Yolo {
    int lx, ly, rx, ry, cls;
    float conf;
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

#endif //YOLO_DEF_H
