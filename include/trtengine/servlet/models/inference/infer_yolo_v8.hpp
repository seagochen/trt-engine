#ifndef COMBINEDPROJECT_INFER_YOLO_V8_HPP
#define COMBINEDPROJECT_INFER_YOLO_V8_HPP

#include <vector>           // For std::vector
#include <type_traits>      // For std::is_same_v
#include <string>           // For std::string

// CUDA Runtime includes for device memory management
#include <cuda_runtime.h>

// Project-specific includes
#include "trtengine/servlet/models/infer_model_multi.h"
#include "trtengine/servlet/models/cuda_tensor_processor.h"
#include "trtengine/servlet/models/common/yolo_dstruct.h"
#include "trtengine/servlet/models/common/yolo_nms.hpp"
#include "trtengine/servlet/models/common/yolo_postprocess.h"
#include "trtengine/utils/logger.h"


// --- Templated InferYoloV8 Class ---
template<typename YoloResultType, typename ConvertFunc>
class InferYoloV8 final : public InferModelBaseMulti {
public:
    /**
     * @brief Constructor for InferYoloV8.
     * @param engine_path Path to the TensorRT engine file.
     * @param maximum_batch Maximum batch size (must be between 1 and 8).
     * @param maximum_items Maximum number of items to process (default is 100).
     * @param infer_features Number of output features from the model.
     * @param output_tensor_defs Vector of TensorDefinition for model outputs.
     * @param converter A function object to convert raw float output to YoloResultType.
     */
    explicit InferYoloV8(const std::string& engine_path,
                         int maximum_batch,
                         int maximum_items,
                         int infer_features,
                         const std::vector<TensorDefinition>& output_tensor_defs,
                         ConvertFunc converter);

    /**
     * @brief Destructor for InferYoloV8.
     */
    ~InferYoloV8() override;

    /**
     * @brief Preprocess the input image for inference.
     * @param image Input image in cv::Mat format.
     * @param batchIdx Index of the batch to which this image belongs.
     */
    void preprocess(const cv::Mat& image, int batchIdx) override;

    /**
     * @brief Postprocess the output from the model.
     * @param batchIdx Index of the batch to process (default is 0).
     * @param args Additional arguments for postprocessing (e.g., classification threshold, IoU threshold).
     * @param results_out Output container for the processed results.
     */
    void postprocess(int batchIdx, const std::map<std::string, std::any>& args, std::any& results_out) override;

private:
    int maximum_batch;      // Maximum number of batch
    int maximum_items;      // Maximum number of items to process
    int image_width;        // Input width (fixed to 640 for YOLOv8)
    int image_height;       // Input height (fixed to 640 for YOLOv8)
    int image_channels;     // Input channels (fixed to 3 for YOLOv8)
    int infer_features;     // Number of output features per sample (e.g., 84 for Obj, 56 for Pose)
    int infer_samples;      // Number of output samples (fixed to 8400 for YOLOv8)

    std::vector<float>      vec_output;         // Output buffer for postprocessing
    ConvertFunc             converter_func;     // Function object for converting raw output to structured Yolo objects
    CudaTensorProcessor     tensor_processor;   // CudaTensorProcessor for image transformation
};

// --- Template Class Implementation ---

template<typename YoloResultType, typename ConvertFunc>
InferYoloV8<YoloResultType, ConvertFunc>::InferYoloV8(
    const std::string& engine_path,
    int maximum_batch,
    int maximum_items,
    int infer_features,
    const std::vector<TensorDefinition>& output_tensor_defs,
    ConvertFunc converter)
    : InferModelBaseMulti(engine_path,
                          {{"images", {maximum_batch, 3, 640, 640}}}, 
                          output_tensor_defs), 
      maximum_batch(maximum_batch),
      maximum_items(maximum_items),
      image_width(640),                 // YOLOv8 模型输入固定为640x640，BGR格式
      image_height(640),                // YOLOv8 模型输入固定为640x640，BGR格式
      image_channels(3),                // YOLOv8 模型输入固定为640x640，BGR格式
      infer_features(infer_features),   // YOLOv8 对于分类任务通常为84，对于姿态估计任务通常为56
      infer_samples(8400),              // YOLOv8 预训练模型默认输出8400个样本
      converter_func(converter),        // 用户需要提供转换函数
      tensor_processor(image_height, image_width, image_channels)
{
    
    // 初始化输出缓冲区，只保留必要的空间
    vec_output.resize(infer_features * maximum_items, 0.0f);
    LOG_VERBOSE_TOPIC("InferYoloV8", "constructor", "InferYoloV8 instance created successfully.");
}

template<typename YoloResultType, typename ConvertFunc>
InferYoloV8<YoloResultType, ConvertFunc>::~InferYoloV8() {
    vec_output.clear();
    LOG_VERBOSE_TOPIC("InferYoloV8", "deconstructor", "Local buffer released successfully.");
}

template<typename YoloResultType, typename ConvertFunc>
void InferYoloV8<YoloResultType, ConvertFunc>::preprocess(const cv::Mat& image, const int batchIdx) {
    if (batchIdx >= maximum_batch) {
        LOG_ERROR("InferYoloV8", "Preprocess: batchIdx (" + std::to_string(batchIdx) + ") >= maximumBatch (" + std::to_string(maximum_batch) + ").");
        return;
    }

    auto cuda_device_ptr = const_cast<float*>(accessCudaBufByBatchIdx("images", batchIdx));
    if (cuda_device_ptr == nullptr) {
        LOG_ERROR("InferYoloV8", "Preprocess: Failed to access CUDA buffer for input at batchIdx " + std::to_string(batchIdx));
        return;
    }

    // 将图像转换为CUDA张量。YOLOv8通常期望输入为BGR格式且无需归一化。
    // 传递 CUDA 流以支持异步传输
    tensor_processor.transformImage(
        image,                  // 输入图像（cv::Mat）
        cuda_device_ptr,        // CUDA设备输出指针
        false,                  // 不进行BGR到RGB的转换（保持BGR格式）
        {0.0f, 0.0f, 0.0f},  // YOLOv8不需要归一化，因此传递零均值
        {1.0f, 1.0f, 1.0f}   // YOLOv8不需要归一化，因此传递单位标准差
    );
}

template<typename YoloResultType, typename ConvertFunc>
void InferYoloV8<YoloResultType, ConvertFunc>::postprocess(int batchIdx, const std::map<std::string, std::any>& args, std::any& results_out) {

    // Initialize results_out with an empty vector of the correct type
    // This ensures std::any_cast will succeed even if no results are found.
    results_out = std::vector<YoloResultType>();

    // --- MODIFICATION START ---
    // Explicitly check for the existence of "cls" and "iou" keys.
    // This is more robust than relying on a try-catch block for logic flow.
    if (args.find("cls") == args.end()) {
        LOG_ERROR("InferYoloV8", "Postprocess: Critical parameter 'cls' not found in arguments map. Aborting.");
        return;
    }
    if (args.find("iou") == args.end()) {
        LOG_ERROR("InferYoloV8", "Postprocess: Critical parameter 'iou' not found in arguments map. Aborting.");
        return;
    }

    // Directly access the values, now that we know they exist.
    // The `std::any_cast` will throw if the type is wrong, which is appropriate behavior.
    float cls = std::any_cast<float>(args.at("cls"));
    float iou = std::any_cast<float>(args.at("iou"));

    // --- MODIFICATION END ---

    if (batchIdx >= maximum_batch) {
        LOG_ERROR("InferYoloV8", "Postprocess: batchIdx out of bounds. Max batch: " + std::to_string(maximum_batch) + ", Current batchIdx: " + std::to_string(batchIdx));
        return; // results_out is already an empty vector
    }

    const float* cuda_device_ptr = this->accessCudaBufByBatchIdx("output0", batchIdx); // Use this-> for clarity
    if (cuda_device_ptr == nullptr) {
        LOG_ERROR("InferYoloV8", "Postprocess: Failed to access CUDA buffer for output at batchIdx " + std::to_string(batchIdx));
        return; // results_out is already an empty vector
    }

    // 根据 YoloResultType 判断是否需要进行姿态估计后处理
    const bool use_pose = std::is_same_v<YoloResultType, YoloPose>;

    // 在 CUDA 设备上执行初步的 YOLO 后处理
    int results_count = inferPostProcForYolo(
        cuda_device_ptr,        // 指向CUDA设备上输出数据的指针
        vec_output,             // 输出的结果向量，包含处理后的边界框、类别等信息
        infer_features,         // 每个样本的特征数量
        infer_samples,          // 样本总数
        cls,                    // 分类阈值，通常用于过滤低置信度的检测结果
        maximum_items,          // 最大边界框数量，通常是8400或其他预设值
        use_pose                // 是否使用姿态估计，如果为false，将计算分类索引和置信度
    );

    // 如果返回的结果数量小于等于0，表示没有有效的检测结果
    if (results_count <= 0) {
        // LOG_VERBOSE_TOPIC("InferYoloV8", "postprocess", "No valid results found after sct_yolo_post_proc.");
        // results_out is already an empty vector, so nothing more to do here.
        return;
    }

    // 准备转换后的结果容器
    std::vector<YoloResultType> yolo_results;

    // 使用提供的转换函数将原始 float 数据转换为结构化的 Yolo 对象
    converter_func(vec_output, yolo_results, infer_features, results_count);

    // 有条件地应用非极大值抑制（NMS）
    // 'if constexpr' 保证只有在明确处理的类型下才会编译 NMS 调用。
    if constexpr (std::is_same_v<YoloResultType, YoloPose> || std::is_same_v<YoloResultType, Yolo>) {
        yolo_results = nms(yolo_results, iou);
    }

    // 将处理后的结果存储到 results_out 中
    results_out = yolo_results; // 将结果包装到 std::any
}

#endif // COMBINEDPROJECT_INFER_YOLO_V8_HPP