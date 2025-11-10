#ifndef COMBINEDPROJECT_TRTENGINE_V2_C_STRUCTS_H
#define COMBINEDPROJECT_TRTENGINE_V2_C_STRUCTS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float x;
    float y;
    float conf;
} C_KeyPoint;

typedef struct {
    int lx, ly, rx, ry;     // Bounding box coordinates
    int cls;                // Class index (e.g., person)
    float conf;             // Overall confidence score for the pose detection
} C_YoloDetect;

// C_YoloPose "包含"一个 C_YoloDetect，并添加了额外信息
typedef struct {
    C_YoloDetect detection; // <-- 在这里“组合”了基础结构
    C_KeyPoint pts[17];     // 姿态估计的17个关键点
} C_YoloPose;

#ifdef __cplusplus
};
#endif


#endif // COMBINEDPROJECT_TRTENGINE_V2_C_STRUCTS_H