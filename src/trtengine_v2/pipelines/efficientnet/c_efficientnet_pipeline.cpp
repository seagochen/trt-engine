/**
 * @file c_efficientnet_pipeline.c
 * @brief Pure C implementation of EfficientNet inference pipeline
 *
 * This file implements a complete EfficientNet classification and feature extraction
 * pipeline in pure C, integrating with TensorRT engine and providing image preprocessing,
 * inference, and postprocessing functionality.
 *
 * Author: TrtEngineToolkits
 * Date: 2025-11-10
 */

#include "trtengine_v2/pipelines/efficientnet/c_efficientnet_pipeline.h"
#include "trtengine_v2/pipelines/efficientnet/c_efficientnet_structures.h"
#include "trtengine_v2/core/trt_engine_multi.h"
#include "trtengine_v2/utils/logger.h"

#include <simple_cuda_toolkits/tensor.hpp>

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// ============================================================================
//                         Internal Structures
// ============================================================================

/**
 * @brief Internal context structure for EfficientNet pipeline
 */
struct C_EfficientNetPipelineContext {
    // Configuration
    C_EfficientNetPipelineConfig config;

    // TensorRT engine
    TrtEngineMultiTs* trt_engine;

    // Input/Output tensors (CUDA device memory)
    Tensor<float>* input_tensor;
    Tensor<float>* logits_tensor;
    Tensor<float>* features_tensor;

    // Host memory buffers for data transfer
    float* host_input_buffer;
    float* host_logits_buffer;
    float* host_features_buffer;

    // Error message
    char error_msg[256];
};

// ============================================================================
//                         Constants and Macros
// ============================================================================

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// ============================================================================
//                         Image Preprocessing
// ============================================================================

/**
 * @brief Resize and normalize image for EfficientNet input
 *
 * This function performs bilinear resize and ImageNet normalization.
 */
static bool preprocess_image(
    const unsigned char* input_data,
    int input_width,
    int input_height,
    int input_channels,
    float* output_buffer,
    int output_width,
    int output_height,
    const float* mean,
    const float* stddev
) {
    if (!input_data || !output_buffer || input_channels != 3) {
        return false;
    }

    // Calculate scaling factors
    float scale_x = (float)input_width / output_width;
    float scale_y = (float)input_height / output_height;

    // Resize and normalize image
    // Format: CHW (Channel-Height-Width) for TensorRT
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < output_height; y++) {
            for (int x = 0; x < output_width; x++) {
                // Source coordinates in original image
                float src_x = (x + 0.5f) * scale_x - 0.5f;
                float src_y = (y + 0.5f) * scale_y - 0.5f;

                src_x = MAX(0.0f, MIN(src_x, input_width - 1.0f));
                src_y = MAX(0.0f, MIN(src_y, input_height - 1.0f));

                int x0 = (int)src_x;
                int y0 = (int)src_y;
                int x1 = MIN(x0 + 1, input_width - 1);
                int y1 = MIN(y0 + 1, input_height - 1);

                float dx = src_x - x0;
                float dy = src_y - y0;

                // Bilinear interpolation
                int src_idx_00 = (y0 * input_width + x0) * input_channels + c;
                int src_idx_01 = (y0 * input_width + x1) * input_channels + c;
                int src_idx_10 = (y1 * input_width + x0) * input_channels + c;
                int src_idx_11 = (y1 * input_width + x1) * input_channels + c;

                float val00 = input_data[src_idx_00] / 255.0f;
                float val01 = input_data[src_idx_01] / 255.0f;
                float val10 = input_data[src_idx_10] / 255.0f;
                float val11 = input_data[src_idx_11] / 255.0f;

                float val = val00 * (1 - dx) * (1 - dy) +
                           val01 * dx * (1 - dy) +
                           val10 * (1 - dx) * dy +
                           val11 * dx * dy;

                // Normalize with ImageNet statistics
                val = (val - mean[c]) / stddev[c];

                // Output: CHW format
                int dst_idx = c * output_height * output_width + y * output_width + x;
                output_buffer[dst_idx] = val;
            }
        }
    }

    return true;
}

// ============================================================================
//                         Pipeline Lifecycle
// ============================================================================

C_EfficientNetPipelineConfig c_efficientnet_pipeline_get_default_config(void) {
    C_EfficientNetPipelineConfig config = {
        .engine_path = NULL,
        .input_width = EFFICIENTNET_DEFAULT_IMAGE_SIZE,
        .input_height = EFFICIENTNET_DEFAULT_IMAGE_SIZE,
        .max_batch_size = 1,
        .num_classes = EFFICIENTNET_DEFAULT_NUM_CLASSES,
        .feature_size = EFFICIENTNET_DEFAULT_FEAT_SIZE,
        .mean = {0.485f, 0.456f, 0.406f},     // ImageNet mean
        .stddev = {0.229f, 0.224f, 0.225f}    // ImageNet std
    };
    return config;
}

bool c_efficientnet_pipeline_validate_config(const C_EfficientNetPipelineConfig* config) {
    if (!config) return false;
    if (!config->engine_path || strlen(config->engine_path) == 0) return false;
    if (config->input_width <= 0 || config->input_height <= 0) return false;
    if (config->max_batch_size <= 0) return false;
    if (config->num_classes <= 0) return false;
    if (config->feature_size <= 0) return false;
    return true;
}

C_EfficientNetPipelineContext* c_efficientnet_pipeline_create(
    const C_EfficientNetPipelineConfig* config
) {
    if (!c_efficientnet_pipeline_validate_config(config)) {
        LOG_ERROR("EfficientNetPipeline", "Invalid configuration");
        return NULL;
    }

    // Allocate context
    C_EfficientNetPipelineContext* ctx = (C_EfficientNetPipelineContext*)calloc(
        1, sizeof(C_EfficientNetPipelineContext)
    );
    if (!ctx) {
        LOG_ERROR("EfficientNetPipeline", "Failed to allocate context");
        return NULL;
    }

    // Copy configuration
    ctx->config = *config;
    ctx->config.engine_path = strdup(config->engine_path);

    // Create TensorRT engine
    ctx->trt_engine = new TrtEngineMultiTs();
    if (!ctx->trt_engine) {
        LOG_ERROR("EfficientNetPipeline", "Failed to create TRT engine");
        free((void*)ctx->config.engine_path);
        free(ctx);
        return NULL;
    }

    // Load engine from file
    if (!ctx->trt_engine->loadFromFile(ctx->config.engine_path)) {
        LOG_ERROR("EfficientNetPipeline", "Failed to load TRT engine");
        delete ctx->trt_engine;
        free((void*)ctx->config.engine_path);
        free(ctx);
        return NULL;
    }

    // Create input tensor (NCHW format)
    std::vector<int> input_dims = {
        ctx->config.max_batch_size,
        3,  // RGB
        ctx->config.input_height,
        ctx->config.input_width
    };
    ctx->input_tensor = new Tensor<float>(TensorType::FLOAT32, input_dims);

    // Create logits tensor
    std::vector<int> logits_dims = {
        ctx->config.max_batch_size,
        ctx->config.num_classes
    };
    ctx->logits_tensor = new Tensor<float>(TensorType::FLOAT32, logits_dims);

    // Create features tensor
    std::vector<int> features_dims = {
        ctx->config.max_batch_size,
        ctx->config.feature_size
    };
    ctx->features_tensor = new Tensor<float>(TensorType::FLOAT32, features_dims);

    // Allocate host buffers
    int input_size = ctx->config.max_batch_size * 3 *
                     ctx->config.input_height * ctx->config.input_width;
    int logits_size = ctx->config.max_batch_size * ctx->config.num_classes;
    int features_size = ctx->config.max_batch_size * ctx->config.feature_size;

    ctx->host_input_buffer = (float*)malloc(input_size * sizeof(float));
    ctx->host_logits_buffer = (float*)malloc(logits_size * sizeof(float));
    ctx->host_features_buffer = (float*)malloc(features_size * sizeof(float));

    if (!ctx->host_input_buffer || !ctx->host_logits_buffer || !ctx->host_features_buffer) {
        LOG_ERROR("EfficientNetPipeline", "Failed to allocate host buffers");
        c_efficientnet_pipeline_destroy(ctx);
        return NULL;
    }

    // Create TensorRT context with input/output shapes
    std::vector<std::string> input_names = {"input"};
    std::vector<nvinfer1::Dims4> input_shapes = {
        {ctx->config.max_batch_size, 3, ctx->config.input_height, ctx->config.input_width}
    };
    std::vector<std::string> output_names = {"logits", "feat"};

    if (!ctx->trt_engine->createContext(input_names, input_shapes, output_names)) {
        LOG_ERROR("EfficientNetPipeline", "Failed to create TRT context");
        c_efficientnet_pipeline_destroy(ctx);
        return NULL;
    }

    LOG_INFO("EfficientNetPipeline", "Pipeline created successfully");
    return ctx;
}

void c_efficientnet_pipeline_destroy(C_EfficientNetPipelineContext* context) {
    if (!context) return;

    if (context->trt_engine) {
        delete context->trt_engine;
    }
    if (context->input_tensor) {
        delete context->input_tensor;
    }
    if (context->logits_tensor) {
        delete context->logits_tensor;
    }
    if (context->features_tensor) {
        delete context->features_tensor;
    }
    if (context->host_input_buffer) {
        free(context->host_input_buffer);
    }
    if (context->host_logits_buffer) {
        free(context->host_logits_buffer);
    }
    if (context->host_features_buffer) {
        free(context->host_features_buffer);
    }
    if (context->config.engine_path) {
        free((void*)context->config.engine_path);
    }

    free(context);
    LOG_INFO("EfficientNetPipeline", "Pipeline destroyed");
}

// ============================================================================
//                         Inference Functions
// ============================================================================

bool c_efficientnet_infer_single(
    C_EfficientNetPipelineContext* context,
    const C_ImageInput* image,
    C_EfficientNetResult* result
) {
    if (!context || !image || !result) {
        return false;
    }

    // Validate input
    if (!image->data || image->width <= 0 || image->height <= 0 ||
        image->channels != 3) {
        snprintf(context->error_msg, sizeof(context->error_msg),
                 "Invalid input image");
        return false;
    }

    // Preprocess image to host buffer
    if (!preprocess_image(
        image->data, image->width, image->height, image->channels,
        context->host_input_buffer,
        context->config.input_width, context->config.input_height,
        context->config.mean, context->config.stddev
    )) {
        snprintf(context->error_msg, sizeof(context->error_msg),
                 "Image preprocessing failed");
        return false;
    }

    // Copy to GPU
    int input_size = context->config.max_batch_size * 3 *
                     context->config.input_height * context->config.input_width;
    std::vector<float> input_vec(context->host_input_buffer,
                                  context->host_input_buffer + input_size);
    context->input_tensor->copyFromVector(input_vec);

    // Run inference
    std::vector<Tensor<float>*> inputs = {context->input_tensor};
    std::vector<Tensor<float>*> outputs = {context->logits_tensor, context->features_tensor};

    if (!context->trt_engine->infer(inputs, outputs)) {
        snprintf(context->error_msg, sizeof(context->error_msg),
                 "TensorRT inference failed");
        return false;
    }

    // Copy outputs back to host
    std::vector<float> logits_vec;
    std::vector<float> features_vec;
    context->logits_tensor->copyToVector(logits_vec);
    context->features_tensor->copyToVector(features_vec);

    memcpy(context->host_logits_buffer, logits_vec.data(),
           logits_vec.size() * sizeof(float));
    memcpy(context->host_features_buffer, features_vec.data(),
           features_vec.size() * sizeof(float));

    // Process results
    result->num_classes = context->config.num_classes;
    result->feature_size = context->config.feature_size;

    // Allocate output arrays
    result->logits = (float*)malloc(result->num_classes * sizeof(float));
    result->features = (float*)malloc(result->feature_size * sizeof(float));

    if (!result->logits || !result->features) {
        snprintf(context->error_msg, sizeof(context->error_msg),
                 "Memory allocation failed");
        if (result->logits) free(result->logits);
        if (result->features) free(result->features);
        return false;
    }

    // Copy logits and find max
    memcpy(result->logits, context->host_logits_buffer,
           result->num_classes * sizeof(float));

    float max_logit = result->logits[0];
    int max_idx = 0;
    for (size_t i = 1; i < result->num_classes; i++) {
        if (result->logits[i] > max_logit) {
            max_logit = result->logits[i];
            max_idx = i;
        }
    }
    result->class_id = max_idx;
    result->confidence = max_logit;

    // Copy features
    memcpy(result->features, context->host_features_buffer,
           result->feature_size * sizeof(float));

    return true;
}

bool c_efficientnet_infer_batch(
    C_EfficientNetPipelineContext* context,
    const C_ImageBatch* batch,
    C_EfficientNetBatchResult* results
) {
    if (!context || !batch || !results || batch->count == 0) {
        return false;
    }

    // For now, process images one by one
    // TODO: Implement true batching
    results->count = batch->count;
    results->results = (C_EfficientNetResult*)calloc(
        batch->count, sizeof(C_EfficientNetResult)
    );

    if (!results->results) {
        return false;
    }

    for (size_t i = 0; i < batch->count; i++) {
        if (!c_efficientnet_infer_single(context, &batch->images[i],
                                          &results->results[i])) {
            // Clean up on failure
            for (size_t j = 0; j < i; j++) {
                c_efficientnet_result_free(&results->results[j]);
            }
            free(results->results);
            results->results = NULL;
            results->count = 0;
            return false;
        }
    }

    return true;
}

// ============================================================================
//                         Memory Management
// ============================================================================

void c_efficientnet_result_free(C_EfficientNetResult* result) {
    if (!result) return;

    if (result->logits) {
        free(result->logits);
        result->logits = NULL;
    }
    if (result->features) {
        free(result->features);
        result->features = NULL;
    }
    result->num_classes = 0;
    result->feature_size = 0;
}

void c_efficientnet_batch_result_free(C_EfficientNetBatchResult* batch_result) {
    if (!batch_result) return;

    if (batch_result->results) {
        for (size_t i = 0; i < batch_result->count; i++) {
            c_efficientnet_result_free(&batch_result->results[i]);
        }
        free(batch_result->results);
        batch_result->results = NULL;
    }
    batch_result->count = 0;
}

// ============================================================================
//                         Utility Functions
// ============================================================================

const char* c_efficientnet_pipeline_get_last_error(
    C_EfficientNetPipelineContext* context
) {
    if (!context) return NULL;
    return context->error_msg[0] != '\0' ? context->error_msg : NULL;
}
