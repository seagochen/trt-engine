/**
 * @file c_efficientnet_pipeline.h
 * @brief EfficientNet inference pipeline (Pure C implementation)
 *
 * This pipeline provides a complete EfficientNet classification and feature
 * extraction implementation in pure C, designed for the v2 architecture.
 *
 * Key features:
 * - Pure C implementation (no C++/OpenCV dependencies in interface)
 * - Uses TrtEngineMultiTs from v2/core
 * - Supports batch inference
 * - Outputs both classification and feature embeddings
 *
 * @author TrtEngineToolkits
 * @date 2025-11-10
 */

#ifndef TRTENGINE_V2_PIPELINES_EFFICIENTNET_C_EFFICIENTNET_PIPELINE_H
#define TRTENGINE_V2_PIPELINES_EFFICIENTNET_C_EFFICIENTNET_PIPELINE_H

#include "trtengine_v2/pipelines/efficientnet/c_efficientnet_structures.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
//                         Pipeline Configuration
// ============================================================================

/**
 * @brief EfficientNet inference pipeline configuration
 */
typedef struct {
    // Model configuration
    const char* engine_path;        ///< TensorRT engine file path
    int input_width;                ///< Model input width (default: 224)
    int input_height;               ///< Model input height (default: 224)
    int max_batch_size;             ///< Maximum batch size

    // Model output configuration
    int num_classes;                ///< Number of classification classes
    int feature_size;               ///< Size of feature embedding vector

    // Preprocessing configuration
    float mean[3];                  ///< Normalization mean (R, G, B)
    float stddev[3];                ///< Normalization standard deviation (R, G, B)
} C_EfficientNetPipelineConfig;

/**
 * @brief EfficientNet pipeline context (opaque handle)
 */
typedef struct C_EfficientNetPipelineContext C_EfficientNetPipelineContext;

// ============================================================================
//                         Pipeline Lifecycle
// ============================================================================

/**
 * @brief Get default pipeline configuration
 *
 * Returns a configuration with reasonable defaults:
 * - Input size: 224x224
 * - Batch size: 1
 * - Classes: 2
 * - Feature size: 256
 * - ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 *
 * @return Default configuration
 */
C_EfficientNetPipelineConfig c_efficientnet_pipeline_get_default_config(void);

/**
 * @brief Validate pipeline configuration
 *
 * @param config Configuration to validate
 * @return true if valid, false otherwise
 */
bool c_efficientnet_pipeline_validate_config(const C_EfficientNetPipelineConfig* config);

/**
 * @brief Create and initialize the pipeline
 *
 * @param config Pipeline configuration
 * @return Pipeline context handle, or NULL on failure
 */
C_EfficientNetPipelineContext* c_efficientnet_pipeline_create(
    const C_EfficientNetPipelineConfig* config
);

/**
 * @brief Destroy the pipeline and free all resources
 *
 * @param context Pipeline context to destroy
 */
void c_efficientnet_pipeline_destroy(C_EfficientNetPipelineContext* context);

// ============================================================================
//                         Inference Functions
// ============================================================================

/**
 * @brief Run inference on a single image
 *
 * @param context Pipeline context
 * @param image Input image (RGB format)
 * @param result Output result (caller must free with c_efficientnet_result_free)
 * @return true on success, false on failure
 */
bool c_efficientnet_infer_single(
    C_EfficientNetPipelineContext* context,
    const C_ImageInput* image,
    C_EfficientNetResult* result
);

/**
 * @brief Run inference on a batch of images
 *
 * @param context Pipeline context
 * @param batch Input image batch
 * @param results Output batch results (caller must free with c_efficientnet_batch_result_free)
 * @return true on success, false on failure
 */
bool c_efficientnet_infer_batch(
    C_EfficientNetPipelineContext* context,
    const C_ImageBatch* batch,
    C_EfficientNetBatchResult* results
);

// ============================================================================
//                         Utility Functions
// ============================================================================

/**
 * @brief Get last error message
 *
 * @param context Pipeline context
 * @return Error message string, or NULL if no error
 *
 * @note The returned string is owned by the context and should not be freed
 */
const char* c_efficientnet_pipeline_get_last_error(
    C_EfficientNetPipelineContext* context
);

#ifdef __cplusplus
}
#endif

#endif // TRTENGINE_V2_PIPELINES_EFFICIENTNET_C_EFFICIENTNET_PIPELINE_H
