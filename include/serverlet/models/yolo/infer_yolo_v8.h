#ifndef INFER_YOLO_V8_H
#define INFER_YOLO_V8_H

#include <vector>
#include "serverlet/models/infer_model_multi.h"
#include "yolo_def.h"


class InferYoloV8Obj final: public InferModelBaseMulti {
public:
    // Constructor and destructor
    // engine_path: TensorRT 引擎文件路径
    // maximum_batch: 最大 1<= batch <=8
    explicit InferYoloV8Obj(const std::string& engine_path, int maximum_batch = 1);

    // Preprocess the image
    void preprocess(const cv::Mat& image, int batchIdx) override;

    // Postprocess the output
    std::vector<Yolo> postprocess(int batchIdx=0, float cls=0.4);

private:
    int g_int_maximumBatch;     // Maximum number of batch
    int g_int_inputWidth;       // Input width
    int g_int_inputHeight;      // Input height
    int g_int_inputChannels;    // Input channels
    int g_int_outputFeatures;   // Number of output features
};


class InferYoloV8Pose final: public InferModelBaseMulti {
public:
    // Constructor and destructor
    // engine_path: TensorRT 引擎文件路径
    // maximum_batch: 最大 1<= batch <=8
    explicit InferYoloV8Pose(const std::string& engine_path, int maximum_batch = 1);

    // Preprocess the image
    void preprocess(const cv::Mat& image, int batchIdx) override;

    // Postprocess the output
    std::vector<YoloPose> postprocess(int batchIdx=0, float cls=0.4);

private:
    int g_int_maximumBatch;     // Maximum number of batch
    int g_int_inputWidth;       // Input width
    int g_int_inputHeight;      // Input height
    int g_int_inputChannels;    // Input channels
    int g_int_outputFeatures;   // Number of output features
};


/**
 * @brief 处理 Yolov8 模型的输出结果
 * 
 * @param cudaOutput CUDA的输出结果指针
 * @param results    存储处理后的结果
 * @param batchIdx   批次索引
 * @param cls        分类阈值
 * @param detectionMode 是否为检测模式 * 
 */
void postprocessYoloV8(const float* cudaOutput, std::vector<std::vector<float>>& results, int batchIdx,
    float cls, bool detectionMode);

#endif //INFER_YOLO_V8_H
