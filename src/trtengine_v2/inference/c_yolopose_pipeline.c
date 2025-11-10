/**
 * @file c_yolopose_pipeline.c
 * @brief Pure C implementation of YOLO Pose inference pipeline
 *
 * This file implements a complete YOLO Pose detection pipeline in pure C,
 * integrating with TensorRT engine and providing image preprocessing,
 * inference, and postprocessing functionality.
 *
 * Author: TrtEngineToolkits
 * Date: 2025-11-10
 */

#include "trtengine_v2/inference/c_yolopose_pipeline.h"
#include "trtengine_v2/core/trt_engine_multi.h"
#include "trtengine_v2/model/c_common_ops.h"
#include "trtengine_v2/utils/logger.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// ============================================================================
//                         Internal Structures
// ============================================================================

/**
 * @brief Internal context structure for YOLO Pose pipeline
 */
struct C_YoloPosePipelineContext {
    // Configuration
    C_YoloPosePipelineConfig config;

    // TensorRT engine
    TrtEngineMultiTs* trt_engine;

    // Input/Output tensors (CUDA device memory)
    Tensor<float>* input_tensor;
    Tensor<float>* output_tensor;

    // Host memory buffers for data transfer
    float* host_input_buffer;
    float* host_output_buffer;

    // Tensor dimensions
    int output_features;     // Number of features per detection (e.g., 56)
    int output_samples;      // Number of detection samples (e.g., 8400)

    // Error message
    char error_msg[256];
};

// ============================================================================
//                         Constants and Macros
// ============================================================================

#define YOLO_POSE_NUM_KEYPOINTS 17
#define YOLO_POSE_BOX_FEATURES  4   // x, y, w, h
#define YOLO_POSE_CONF_FEATURES 1   // objectness
#define YOLO_POSE_CLS_FEATURES  1   // class (usually just "person")
#define YOLO_POSE_KPT_FEATURES  (YOLO_POSE_NUM_KEYPOINTS * 3)  // x, y, conf for each

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// ============================================================================
//                         Image Preprocessing
// ============================================================================

/**
 * @brief Resize and normalize image for YOLO input
 *
 * This function performs letterbox resize to maintain aspect ratio,
 * then normalizes pixel values to [0, 1].
 */
static bool preprocess_image(
    const unsigned char* input_data,
    int input_width,
    int input_height,
    int input_channels,
    float* output_buffer,
    int output_width,
    int output_height,
    float* scale_x,
    float* scale_y,
    int* pad_x,
    int* pad_y
) {
    if (!input_data || !output_buffer || input_channels != 3) {
        return false;
    }

    // Calculate letterbox parameters
    float scale = MIN((float)output_width / input_width,
                      (float)output_height / input_height);

    int scaled_width = (int)(input_width * scale);
    int scaled_height = (int)(input_height * scale);

    *pad_x = (output_width - scaled_width) / 2;
    *pad_y = (output_height - scaled_height) / 2;
    *scale_x = scale;
    *scale_y = scale;

    // Initialize output buffer to 114 (gray padding)
    for (int i = 0; i < output_width * output_height * 3; i++) {
        output_buffer[i] = 114.0f / 255.0f;
    }

    // Resize and copy image with bilinear interpolation
    // Format: CHW (Channel-Height-Width) for TensorRT
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < scaled_height; y++) {
            for (int x = 0; x < scaled_width; x++) {
                // Source coordinates in original image
                float src_x = x / scale;
                float src_y = y / scale;

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

                // Output: CHW format
                int dst_y = y + *pad_y;
                int dst_x = x + *pad_x;
                int dst_idx = c * output_height * output_width +
                              dst_y * output_width + dst_x;

                output_buffer[dst_idx] = val;
            }
        }
    }

    return true;
}

// ============================================================================
//                         Output Postprocessing
// ============================================================================

/**
 * @brief Decode YOLO pose output into detections
 *
 * YOLOv8-pose output format:
 * [batch, features, samples] where features = 56 (4 box + 1 obj + 1 cls + 51 keypoints)
 */
static size_t decode_yolo_output(
    const float* output,
    int num_samples,
    int num_features,
    float conf_threshold,
    float scale_x,
    float scale_y,
    int pad_x,
    int pad_y,
    int original_width,
    int original_height,
    C_YoloPose* detections,
    size_t max_detections
) {
    size_t num_detected = 0;

    // YOLOv8 format: [cx, cy, w, h, objectness, class_conf, kpt_x1, kpt_y1, kpt_conf1, ...]
    for (int i = 0; i < num_samples && num_detected < max_detections; i++) {
        // Get objectness score
        float objectness = output[4 * num_samples + i];

        // Get class confidence (usually just one class for pose: person)
        float class_conf = output[5 * num_samples + i];

        // Combined confidence
        float confidence = objectness * class_conf;

        if (confidence < conf_threshold) {
            continue;
        }

        // Decode bounding box
        float cx = output[0 * num_samples + i];
        float cy = output[1 * num_samples + i];
        float w = output[2 * num_samples + i];
        float h = output[3 * num_samples + i];

        // Convert from model coordinates to original image coordinates
        // Remove padding and scale back
        cx = (cx - pad_x) / scale_x;
        cy = (cy - pad_y) / scale_y;
        w = w / scale_x;
        h = h / scale_y;

        // Convert to lx, ly, rx, ry format
        int lx = (int)((cx - w / 2.0f));
        int ly = (int)((cy - h / 2.0f));
        int rx = (int)((cx + w / 2.0f));
        int ry = (int)((cy + h / 2.0f));

        // Clip to image bounds
        lx = MAX(0, MIN(lx, original_width - 1));
        ly = MAX(0, MIN(ly, original_height - 1));
        rx = MAX(0, MIN(rx, original_width - 1));
        ry = MAX(0, MIN(ry, original_height - 1));

        // Create detection
        C_YoloPose* det = &detections[num_detected];
        det->detection.lx = lx;
        det->detection.ly = ly;
        det->detection.rx = rx;
        det->detection.ry = ry;
        det->detection.cls = 0;  // Person class
        det->detection.conf = confidence;

        // Decode keypoints (17 keypoints, each with x, y, conf)
        for (int k = 0; k < YOLO_POSE_NUM_KEYPOINTS; k++) {
            int kpt_offset = 6 + k * 3;  // Skip box(4) + obj(1) + cls(1)

            float kpt_x = output[(kpt_offset + 0) * num_samples + i];
            float kpt_y = output[(kpt_offset + 1) * num_samples + i];
            float kpt_conf = output[(kpt_offset + 2) * num_samples + i];

            // Transform keypoint coordinates
            kpt_x = (kpt_x - pad_x) / scale_x;
            kpt_y = (kpt_y - pad_y) / scale_y;

            det->pts[k].x = kpt_x;
            det->pts[k].y = kpt_y;
            det->pts[k].conf = kpt_conf;
        }

        num_detected++;
    }

    return num_detected;
}

// ============================================================================
//                         Pipeline Lifecycle
// ============================================================================

C_YoloPosePipelineConfig c_yolopose_pipeline_get_default_config(void) {
    C_YoloPosePipelineConfig config = {
        .engine_path = NULL,
        .input_width = 640,
        .input_height = 640,
        .max_batch_size = 1,
        .conf_threshold = 0.25f,
        .iou_threshold = 0.45f,
        .num_keypoints = YOLO_POSE_NUM_KEYPOINTS,
        .num_classes = 1  // Usually just "person" for pose
    };
    return config;
}

bool c_yolopose_pipeline_validate_config(const C_YoloPosePipelineConfig* config) {
    if (!config) return false;
    if (!config->engine_path || strlen(config->engine_path) == 0) return false;
    if (config->input_width <= 0 || config->input_height <= 0) return false;
    if (config->max_batch_size <= 0) return false;
    if (config->conf_threshold < 0.0f || config->conf_threshold > 1.0f) return false;
    if (config->iou_threshold < 0.0f || config->iou_threshold > 1.0f) return false;
    return true;
}

C_YoloPosePipelineContext* c_yolopose_pipeline_create(
    const C_YoloPosePipelineConfig* config
) {
    if (!c_yolopose_pipeline_validate_config(config)) {
        LOG_ERROR("YoloPosePipeline", "Invalid configuration");
        return NULL;
    }

    // Allocate context
    C_YoloPosePipelineContext* ctx = (C_YoloPosePipelineContext*)calloc(
        1, sizeof(C_YoloPosePipelineContext)
    );
    if (!ctx) {
        LOG_ERROR("YoloPosePipeline", "Failed to allocate context");
        return NULL;
    }

    // Copy configuration
    ctx->config = *config;
    ctx->config.engine_path = strdup(config->engine_path);

    // Create TensorRT engine
    ctx->trt_engine = new TrtEngineMultiTs();
    if (!ctx->trt_engine) {
        LOG_ERROR("YoloPosePipeline", "Failed to create TRT engine");
        free((void*)ctx->config.engine_path);
        free(ctx);
        return NULL;
    }

    // Load engine from file
    if (!ctx->trt_engine->loadFromFile(ctx->config.engine_path)) {
        LOG_ERROR("YoloPosePipeline", "Failed to load TRT engine");
        delete ctx->trt_engine;
        free((void*)ctx->config.engine_path);
        free(ctx);
        return NULL;
    }

    // Calculate output dimensions
    // YOLOv8-pose: features = 56 (4 box + 1 obj + 1 cls + 17*3 keypoints)
    ctx->output_features = 56;
    ctx->output_samples = 8400;  // Default for 640x640 input

    // Create input tensor (NCHW format)
    std::vector<int> input_dims = {
        ctx->config.max_batch_size,
        3,  // RGB
        ctx->config.input_height,
        ctx->config.input_width
    };
    ctx->input_tensor = new Tensor<float>(input_dims, TensorType::Device);

    // Create output tensor
    std::vector<int> output_dims = {
        ctx->config.max_batch_size,
        ctx->output_features,
        ctx->output_samples
    };
    ctx->output_tensor = new Tensor<float>(output_dims, TensorType::Device);

    // Allocate host buffers
    int input_size = ctx->config.max_batch_size * 3 *
                     ctx->config.input_height * ctx->config.input_width;
    int output_size = ctx->config.max_batch_size *
                      ctx->output_features * ctx->output_samples;

    ctx->host_input_buffer = (float*)malloc(input_size * sizeof(float));
    ctx->host_output_buffer = (float*)malloc(output_size * sizeof(float));

    if (!ctx->host_input_buffer || !ctx->host_output_buffer) {
        LOG_ERROR("YoloPosePipeline", "Failed to allocate host buffers");
        c_yolopose_pipeline_destroy(ctx);
        return NULL;
    }

    // Create TensorRT context with input/output shapes
    std::vector<std::string> input_names = {"images"};
    std::vector<nvinfer1::Dims4> input_shapes = {
        {ctx->config.max_batch_size, 3, ctx->config.input_height, ctx->config.input_width}
    };
    std::vector<std::string> output_names = {"output0"};

    if (!ctx->trt_engine->createContext(input_names, input_shapes, output_names)) {
        LOG_ERROR("YoloPosePipeline", "Failed to create TRT context");
        c_yolopose_pipeline_destroy(ctx);
        return NULL;
    }

    LOG_INFO("YoloPosePipeline", "Pipeline created successfully");
    return ctx;
}

void c_yolopose_pipeline_destroy(C_YoloPosePipelineContext* context) {
    if (!context) return;

    if (context->trt_engine) {
        delete context->trt_engine;
    }
    if (context->input_tensor) {
        delete context->input_tensor;
    }
    if (context->output_tensor) {
        delete context->output_tensor;
    }
    if (context->host_input_buffer) {
        free(context->host_input_buffer);
    }
    if (context->host_output_buffer) {
        free(context->host_output_buffer);
    }
    if (context->config.engine_path) {
        free((void*)context->config.engine_path);
    }

    free(context);
    LOG_INFO("YoloPosePipeline", "Pipeline destroyed");
}

// ============================================================================
//                         Inference Functions
// ============================================================================

bool c_yolopose_infer_single(
    C_YoloPosePipelineContext* context,
    const C_ImageInput* image,
    C_YoloPoseImageResult* result
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

    // Preprocessing parameters
    float scale_x, scale_y;
    int pad_x, pad_y;

    // Preprocess image to host buffer
    if (!preprocess_image(
        image->data, image->width, image->height, image->channels,
        context->host_input_buffer,
        context->config.input_width, context->config.input_height,
        &scale_x, &scale_y, &pad_x, &pad_y
    )) {
        snprintf(context->error_msg, sizeof(context->error_msg),
                 "Image preprocessing failed");
        return false;
    }

    // Copy to GPU
    context->input_tensor->copyFromHost(context->host_input_buffer);

    // Run inference
    std::vector<Tensor<float>*> inputs = {context->input_tensor};
    std::vector<Tensor<float>*> outputs = {context->output_tensor};

    if (!context->trt_engine->infer(inputs, outputs)) {
        snprintf(context->error_msg, sizeof(context->error_msg),
                 "TensorRT inference failed");
        return false;
    }

    // Copy output back to host
    context->output_tensor->copyToHost(context->host_output_buffer);

    // Decode detections (max 300 detections)
    const size_t max_detections = 300;
    C_YoloPose* temp_detections = (C_YoloPose*)malloc(
        max_detections * sizeof(C_YoloPose)
    );
    if (!temp_detections) {
        snprintf(context->error_msg, sizeof(context->error_msg),
                 "Memory allocation failed");
        return false;
    }

    size_t num_decoded = decode_yolo_output(
        context->host_output_buffer,
        context->output_samples,
        context->output_features,
        context->config.conf_threshold,
        scale_x, scale_y, pad_x, pad_y,
        image->width, image->height,
        temp_detections,
        max_detections
    );

    // Apply NMS
    C_YoloPose* nms_result = (C_YoloPose*)malloc(
        num_decoded * sizeof(C_YoloPose)
    );
    if (!nms_result) {
        free(temp_detections);
        snprintf(context->error_msg, sizeof(context->error_msg),
                 "Memory allocation failed");
        return false;
    }

    size_t num_after_nms = 0;
    c_nms_pose(
        temp_detections, num_decoded,
        context->config.iou_threshold,
        nms_result, &num_after_nms
    );

    // Fill result
    result->image_index = 0;
    result->num_poses = num_after_nms;
    result->poses = (C_YoloPose*)malloc(num_after_nms * sizeof(C_YoloPose));

    if (result->poses) {
        memcpy(result->poses, nms_result, num_after_nms * sizeof(C_YoloPose));
    }

    free(temp_detections);
    free(nms_result);

    return result->poses != NULL;
}

bool c_yolopose_infer_batch(
    C_YoloPosePipelineContext* context,
    const C_ImageBatch* batch,
    C_YoloPoseBatchResult* result
) {
    if (!context || !batch || !result || batch->count == 0) {
        return false;
    }

    // For now, process images one by one
    // TODO: Implement true batching
    result->num_images = batch->count;
    result->results = (C_YoloPoseImageResult*)calloc(
        batch->count, sizeof(C_YoloPoseImageResult)
    );

    if (!result->results) {
        return false;
    }

    for (size_t i = 0; i < batch->count; i++) {
        if (!c_yolopose_infer_single(context, &batch->images[i],
                                      &result->results[i])) {
            // Clean up on failure
            for (size_t j = 0; j < i; j++) {
                c_yolopose_image_result_free(&result->results[j]);
            }
            free(result->results);
            result->results = NULL;
            result->num_images = 0;
            return false;
        }
        result->results[i].image_index = i;
    }

    return true;
}

// ============================================================================
//                         Memory Management
// ============================================================================

void c_yolopose_image_result_free(C_YoloPoseImageResult* result) {
    if (!result) return;

    if (result->poses) {
        free(result->poses);
        result->poses = NULL;
    }
    result->num_poses = 0;
}

void c_yolopose_batch_result_free(C_YoloPoseBatchResult* result) {
    if (!result) return;

    if (result->results) {
        for (size_t i = 0; i < result->num_images; i++) {
            c_yolopose_image_result_free(&result->results[i]);
        }
        free(result->results);
        result->results = NULL;
    }
    result->num_images = 0;
}

// ============================================================================
//                         Utility Functions
// ============================================================================

const char* c_yolopose_pipeline_get_last_error(
    C_YoloPosePipelineContext* context
) {
    if (!context) return NULL;
    return context->error_msg[0] != '\0' ? context->error_msg : NULL;
}
