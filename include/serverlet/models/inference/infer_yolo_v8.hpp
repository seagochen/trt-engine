#ifndef INFER_YOLO_V8_HPP
#define INFER_YOLO_V8_HPP

#include <vector>           // For std::vector
#include <functional>       // For std::function
#include <type_traits>      // For std::is_same_v
#include <string>           // For std::string

// CUDA Runtime includes for device memory management
#include <cuda_runtime.h>

// Project-specific includes
#include "serverlet/models/infer_model_multi.h"
#include "serverlet/models/common/generic_image_to_tensor.h"
#include "serverlet/models/common/yolo_dstruct.h"
#include "serverlet/models/common/yolo_nms.hpp"
#include "serverlet/models/common/yolo_postprocess.h"
#include "serverlet/utils/logger.h"


// --- Templated InferYoloV8 Class ---
template<typename YoloResultType, typename ConvertFunc>
class InferYoloV8 final : public InferModelBaseMulti {
public:
    /**
     * @brief Constructor for InferYoloV8.
     * @param engine_path Path to the TensorRT engine file.
     * @param maximum_batch Maximum batch size (must be between 1 and 8).
     * @param maximum_items Maximum number of items to process (default is 100).
     * @param infer_features_val Number of output features from the model.
     * @param output_tensor_defs Vector of TensorDefinition for model outputs.
     * @param converter A function object to convert raw float output to YoloResultType.
     */
    explicit InferYoloV8(const std::string& engine_path,
                         int maximum_batch,
                         int maximum_items,
                         int infer_features_val,
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
     * @param cls Confidence threshold for class detection (default is 0.4).
     * @param iou IoU threshold for non-maximum suppression (default is 0.5).
     * @return A vector of YoloResultType objects containing detected bounding boxes/keypoints.
     */
    [[nodiscard]] std::vector<YoloResultType> postprocess(int batchIdx=0, float cls=0.4, float iou=0.5);

private:
    int maximum_batch;      // Maximum number of batch
    int maximum_items;      // Maximum number of items to process
    int image_width;        // Input width (fixed to 640 for YOLOv8)
    int image_height;       // Input height (fixed to 640 for YOLOv8)
    int image_channels;     // Input channels (fixed to 3 for YOLOv8)
    int infer_features;     // Number of output features per sample (e.g., 84 for Obj, 56 for Pose)
    int infer_samples;      // Number of output samples (fixed to 8400 for YOLOv8)

    std::vector<float> g_vec_output; // Output buffer for postprocessing
    ConvertFunc m_converter;         // Function object for converting raw output to structured Yolo objects
};

// --- Template Class Implementation ---

template<typename YoloResultType, typename ConvertFunc>
InferYoloV8<YoloResultType, ConvertFunc>::InferYoloV8(
    const std::string& engine_path,
    int maximum_batch,
    int maximum_items,
    int infer_features_val,
    const std::vector<TensorDefinition>& output_tensor_defs,
    ConvertFunc converter)
    : InferModelBaseMulti(engine_path,
                          {{"images", {maximum_batch, 3, 640, 640}}}, // Fixed input tensor definition
                          output_tensor_defs), // Model-specific output tensor definition
      maximum_batch(maximum_batch),
      maximum_items(maximum_items),
      image_width(640),
      image_height(640),
      image_channels(3),
      infer_features(infer_features_val), // Initialized from constructor parameter
      infer_samples(8400),                // Fixed for YOLOv8
      m_converter(converter)
{
    g_vec_output.resize(infer_features * infer_samples, 0.0f);
    LOG_VERBOSE_TOPIC("InferYoloV8", "constructor", "InferYoloV8 instance created successfully.");
}

template<typename YoloResultType, typename ConvertFunc>
InferYoloV8<YoloResultType, ConvertFunc>::~InferYoloV8() {
    g_vec_output.clear();
    LOG_VERBOSE_TOPIC("InferYoloV8", "deconstructor", "Local buffer released successfully.");
}

template<typename YoloResultType, typename ConvertFunc>
void InferYoloV8<YoloResultType, ConvertFunc>::preprocess(const cv::Mat& image, const int batchIdx) {
    if (batchIdx >= maximum_batch) {
        LOG_ERROR("InferYoloV8", "Preprocess: batchIdx out of bounds. Max batch: " + std::to_string(maximum_batch) + ", Current batchIdx: " + std::to_string(batchIdx));
        return;
    }

    const float* cuda_device_ptr = accessCudaBufByBatchIdx("images", batchIdx);
    auto cuda_buffer_float = const_cast<float*>(cuda_device_ptr);
    if (cuda_buffer_float == nullptr) {
        LOG_ERROR("InferYoloV8", "Preprocess: Failed to access CUDA buffer for input at batchIdx " + std::to_string(batchIdx));
        return;
    }

    // Convert image to CUDA tensor. YOLOv8 typically expects BGR and no normalization.
    imageToCudaTensor(
        image,               // Input image (cv::Mat)
        cuda_buffer_float,   // CUDA device pointer for output
        image_height,        // Target height
        image_width,         // Target width
        image_channels,      // Target channels
        false                // Do not convert BGR to RGB (keep as BGR)
    );
}

template<typename YoloResultType, typename ConvertFunc>
std::vector<YoloResultType> InferYoloV8<YoloResultType, ConvertFunc>::postprocess(const int batchIdx, const float cls, const float iou) {
    if (batchIdx >= maximum_batch) {
        LOG_ERROR("InferYoloV8", "Postprocess: batchIdx out of bounds. Max batch: " + std::to_string(maximum_batch) + ", Current batchIdx: " + std::to_string(batchIdx));
        return {};
    }

    const float* cuda_device_ptr = accessCudaBufByBatchIdx("output0", batchIdx);
    if (cuda_device_ptr == nullptr) {
        LOG_ERROR("InferYoloV8", "Postprocess: Failed to access CUDA buffer for output at batchIdx " + std::to_string(batchIdx));
        return {};
    }

    // Determine if pose processing is needed based on the YoloResultType
    bool use_pose = std::is_same_v<YoloResultType, YoloPose>;

    // Perform the initial YOLO post-processing on the CUDA device
    int results_count = inferPostProcForYolo(cuda_device_ptr, g_vec_output, infer_features, infer_samples, cls, use_pose);
    if (results_count <= 0) { // sct_yolo_post_proc returns -1 on error, 0 for no results
        LOG_VERBOSE_TOPIC("InferYoloV8", "postprocess", "No valid results found after sct_yolo_post_proc.");
        return {};
    }

    std::vector<YoloResultType> yolo_results;
    // Resize vector to hold expected number of results, prevent reallocations during conversion
    yolo_results.reserve(results_count);
    // Use the provided converter function to transform raw float data into structured Yolo objects
    m_converter(g_vec_output, yolo_results, infer_features, results_count);

    // Apply Non-Maximum Suppression (NMS) conditionally
    // 'if constexpr' ensures that the NMS call is only compiled for types where it's explicitly handled.
    if constexpr (std::is_same_v<YoloResultType, YoloPose> || std::is_same_v<YoloResultType, Yolo>) {
        yolo_results = nms(yolo_results, iou);
    }
    // You can add more specific NMS logic for other types if needed with additional 'else if constexpr'

    return yolo_results;
}

#endif // INFER_YOLO_V8_HPP