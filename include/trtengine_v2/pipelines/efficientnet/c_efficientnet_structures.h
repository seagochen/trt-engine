/**
 * @file c_efficientnet_structures.h
 * @brief EfficientNet specific data structures
 *
 * This file defines structures for EfficientNet classification and feature extraction models.
 * EfficientNet is used for image classification and generating feature embeddings.
 *
 * @author TrtEngineToolkits
 * @date 2025-11-10
 */

#ifndef TRTENGINE_V2_PIPELINES_EFFICIENTNET_C_EFFICIENTNET_STRUCTURES_H
#define TRTENGINE_V2_PIPELINES_EFFICIENTNET_C_EFFICIENTNET_STRUCTURES_H

#include <stddef.h>
#include "trtengine_v2/common/c_structures.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Default feature vector size for EfficientNet B0
 */
#define EFFICIENTNET_DEFAULT_FEAT_SIZE 256

/**
 * @brief Default number of classes
 */
#define EFFICIENTNET_DEFAULT_NUM_CLASSES 2

/**
 * @brief Default input image size for EfficientNet B0 (224x224)
 */
#define EFFICIENTNET_DEFAULT_IMAGE_SIZE 224

/**
 * @brief EfficientNet classification result with feature vector
 *
 * This structure contains both the classification result and the feature embedding
 * extracted from the image.
 */
typedef struct {
    int class_id;                  ///< Predicted class ID (argmax of logits)
    float confidence;              ///< Confidence score (max logit value)
    float* logits;                 ///< Raw logits for all classes (size: num_classes)
    size_t num_classes;            ///< Number of classes
    float* features;               ///< Feature embedding vector (size: feature_size)
    size_t feature_size;           ///< Size of feature vector
} C_EfficientNetResult;

/**
 * @brief Batch of EfficientNet results
 */
typedef struct {
    C_EfficientNetResult* results; ///< Array of results (one per image)
    size_t count;                  ///< Number of results in the batch
} C_EfficientNetBatchResult;

// Note: C_ImageInput and C_ImageBatch are defined in trtengine_v2/common/c_structures.h

/**
 * @brief Free a single EfficientNet result
 *
 * @param result Pointer to the result to free
 */
void c_efficientnet_result_free(C_EfficientNetResult* result);

/**
 * @brief Free a batch of EfficientNet results
 *
 * @param batch_result Pointer to the batch result to free
 */
void c_efficientnet_batch_result_free(C_EfficientNetBatchResult* batch_result);

#ifdef __cplusplus
}
#endif

#endif // TRTENGINE_V2_PIPELINES_EFFICIENTNET_C_EFFICIENTNET_STRUCTURES_H
