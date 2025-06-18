// infer_efficientnet.hpp
#ifndef INFER_EFFICIENTNET_HPP
#define INFER_EFFICIENTNET_HPP

#include <vector>
#include <string>
#include <opencv2/opencv.hpp> // Assuming OpenCV is installed and linked

// Project-specific headers
#include "serverlet/models/infer_model_multi.h"
#include "serverlet/models/common/generic_image_to_tensor.h"
#include "serverlet/utils/logger.h"


class EfficientNetForFeatAndClassification final : public InferModelBaseMulti {
public:
    /**
     * @brief Constructor for EfficientNetForFeatAndClassification.
     * @param engine_path Path to the TensorRT engine file.
     * @param maximum_batch Maximum batch size (default is 1), must be between 1 and 8.
     */
    explicit EfficientNetForFeatAndClassification(const std::string& engine_path, int maximum_batch = 1);

    /**
     * @brief Preprocesses a single OpenCV image and uploads it to the GPU.
     * @param image Input image in cv::Mat format.
     * @param batchIdx Index of the batch to which this image belongs.
     */
    void preprocess(const cv::Mat& image, int batchIdx) override;

    /**
     * @brief Postprocesses the model's output, reading features and logits, and returning them combined.
     * The returned vector contains [max_class_index, feat0, feat1, ..., featN].
     * @param batchIdx Index of the batch to process.
     * @return A vector of floats containing the maximum class index followed by the feature vector.
     */
    std::vector<float> postprocess(int batchIdx);

private:
    // Using 'm_' prefix for member variables for clarity, common in C++ style guides
    int m_maximumBatch;
    int m_inputWidth;
    int m_inputHeight;
    int m_inputChannels;
};


// --- Inline Implementation ---

// Constructor implementation: Use member initializer list for efficiency and safety.
inline EfficientNetForFeatAndClassification::EfficientNetForFeatAndClassification(
    const std::string& engine_path,
    int maximum_batch)
    : InferModelBaseMulti(
        engine_path,
        std::vector<TensorDefinition>{{"input", {maximum_batch, 3, 224, 224}}},
        std::vector<TensorDefinition>{
            {"logits", {maximum_batch, 2}},
            {"feat",   {maximum_batch, 256}}
        }),
      m_maximumBatch(maximum_batch),
      m_inputWidth(224),
      m_inputHeight(224),
      m_inputChannels(3)
{
    // Constructor body can be empty if all initialization is done in the initializer list
    LOG_VERBOSE_TOPIC("EfficientNetFeatCls", "constructor", "EfficientNetForFeatAndClassification instance created.");
}


// Preprocess method implementation
inline void EfficientNetForFeatAndClassification::preprocess(const cv::Mat& image, int batchIdx) {
    if (batchIdx >= m_maximumBatch) {
        LOG_ERROR("EfficientNetFeatCls", "Preprocess: batchIdx (" + std::to_string(batchIdx) + ") >= maximumBatch (" + std::to_string(m_maximumBatch) + ")");
        return;
    }

    // Standardization parameters for EfficientNet
    // These could be static const members if always the same, to avoid recreation per call
    const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    const std::vector<float> stdv = {0.229f, 0.224f, 0.225f};

    // Access the CUDA buffer for the current batch index
    const float* cuda_buffer = accessCudaBufByBatchIdx("input", batchIdx);
    float* cuda_buffer_float = const_cast<float*>(cuda_buffer); // Safe as we're writing to it

    if (cuda_buffer_float == nullptr) {
        LOG_ERROR("EfficientNetFeatCls", "Preprocess: Failed to access CUDA buffer for input at batchIdx " + std::to_string(batchIdx));
        return;
    }

    // Convert and copy image to CUDA device with BGR to RGB conversion and normalization
    imageToCudaTensor(
        image,               // Input image
        cuda_buffer_float,   // CUDA device pointer
        m_inputHeight,       // Target height
        m_inputWidth,        // Target width
        m_inputChannels,     // Target channels
        true,                // Perform BGR to RGB conversion
        mean,                // Mean for normalization
        stdv                 // Standard deviation for normalization
    );
}


// Postprocess method implementation
inline std::vector<float> EfficientNetForFeatAndClassification::postprocess(int batchIdx) {
    if (batchIdx >= m_maximumBatch) {
        LOG_ERROR("EfficientNetFeatCls", "Postprocess: batchIdx (" + std::to_string(batchIdx) + ") >= maximumBatch (" + std::to_string(m_maximumBatch) + ")");
        return {};
    }

    std::vector<float> feats;
    std::vector<float> types;

    // Copy data from CUDA output buffers to CPU vectors
    // These functions should handle memory allocation for feats and types automatically
    copyCpuDataFromOutputBuffer("feat", feats, batchIdx);
    copyCpuDataFromOutputBuffer("logits", types, batchIdx);

    // Ensure types vector has at least 2 elements for safe access
    if (types.size() < 2) {
        LOG_ERROR("EfficientNetFeatCls", "Postprocess: Logits vector has fewer than 2 elements for batchIdx " + std::to_string(batchIdx));
        return {}; // Return empty vector if not enough logits
    }

    // Determine the class with the maximum logit score
    int maxIndex = (types[0] > types[1]) ? 0 : 1;

    // Construct the final result vector: [maxIndex, feat0, feat1, ..., featN]
    std::vector<float> result;
    result.reserve(1 + feats.size()); // Reserve space to avoid reallocations
    result.push_back(static_cast<float>(maxIndex)); // First element is the predicted class index
    result.insert(result.end(),
                  feats.begin(),
                  feats.end()); // Append all feature elements

    return result;
}

#endif //INFER_EFFICIENTNET_HPP