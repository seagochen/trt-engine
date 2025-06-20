#ifndef C_DSTRUCT_H
#define C_DSTRUCT_H

#ifdef __cplusplus
extern "C" {
#endif

    // 定义C风格的Point结构体
    typedef struct C_Point {
        float x;
        float y;
        float score; // 关键点置信度
    } C_Point;

    // 定义C风格的边界框结构体 (x1, y1, x2, y2)
    typedef struct C_Rect {
        float x1;
        float y1;
        float x2;
        float y2;
    } C_Rect;

    // 对应C++的YoloPose，但增加了分类信息
    typedef struct C_Extended_Pose_Feats {

        // YOLO 检测结果
        C_Rect box;         // 边界框
        float confidence;   // 检测置信度
        float class_id;     // EfficientNet分类结果 (0 or 1 for your case)
        C_Point pts[17];    // YOLOv8 Pose通常有17个关键点

        // 追加的人物特征
        float features[256]; // 如果需要返回特征向量

    } C_Extended_Person_Feats;

    // 每张图片的最终处理结果，封装了C_Extended_Pose_Feats
    typedef struct C_InferenceResult {
        int num_detected;                       // 检测到的物体数量, 如果检测失败则为-1，只有当
                                                // num_detected > 0 时，detections 才有效
        C_Extended_Person_Feats* detections;    // 检测结果数组，处理结束后需要由调用方释放内存
    } C_InferenceResult;

#ifdef __cplusplus
};
#endif

#endif // C_DSTRUCT_H