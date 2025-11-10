/**
 * @file yolopose_pipeline_example.c
 * @brief Example usage of the pure C YOLO Pose inference pipeline
 *
 * This example demonstrates how to use the v2 YOLO Pose pipeline
 * to perform pose detection on images.
 */

#include "trtengine_v2/inference/c_yolopose_pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
//                         Helper Functions
// ============================================================================

/**
 * @brief Load a dummy RGB image for testing
 *
 * In a real application, you would load actual image data from a file.
 * This function creates a simple gradient image for demonstration.
 */
static unsigned char* create_dummy_image(int width, int height, int channels) {
    size_t size = width * height * channels;
    unsigned char* data = (unsigned char*)malloc(size);

    if (!data) return NULL;

    // Create a simple gradient pattern
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            data[idx + 0] = (x * 255) / width;        // R
            data[idx + 1] = (y * 255) / height;       // G
            data[idx + 2] = ((x + y) * 255) / (width + height); // B
        }
    }

    return data;
}

/**
 * @brief Print detection results
 */
static void print_results(const C_YoloPoseImageResult* result) {
    printf("\n=== Detection Results for Image %d ===\n", result->image_index);
    printf("Total poses detected: %zu\n\n", result->num_poses);

    for (size_t i = 0; i < result->num_poses; i++) {
        const C_YoloPose* pose = &result->poses[i];

        printf("Pose #%zu:\n", i + 1);
        printf("  Bounding Box: [%d, %d, %d, %d]\n",
               pose->detection.lx, pose->detection.ly,
               pose->detection.rx, pose->detection.ry);
        printf("  Confidence: %.3f\n", pose->detection.conf);
        printf("  Class: %d\n", pose->detection.cls);

        printf("  Keypoints:\n");
        const char* keypoint_names[] = {
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        };

        for (int k = 0; k < 17; k++) {
            const C_KeyPoint* kpt = &pose->pts[k];
            printf("    %2d. %-15s: (%.1f, %.1f) conf=%.2f\n",
                   k, keypoint_names[k], kpt->x, kpt->y, kpt->conf);
        }
        printf("\n");
    }
}

// ============================================================================
//                         Example 1: Single Image Inference
// ============================================================================

static int example_single_image(const char* engine_path) {
    printf("========================================\n");
    printf("Example 1: Single Image Inference\n");
    printf("========================================\n\n");

    // 1. Get default configuration
    C_YoloPosePipelineConfig config = c_yolopose_pipeline_get_default_config();
    config.engine_path = engine_path;
    config.input_width = 640;
    config.input_height = 640;
    config.conf_threshold = 0.25f;
    config.iou_threshold = 0.45f;

    // 2. Validate configuration
    if (!c_yolopose_pipeline_validate_config(&config)) {
        printf("ERROR: Invalid configuration\n");
        return 1;
    }
    printf("✓ Configuration validated\n");

    // 3. Create pipeline
    C_YoloPosePipelineContext* ctx = c_yolopose_pipeline_create(&config);
    if (!ctx) {
        printf("ERROR: Failed to create pipeline\n");
        return 1;
    }
    printf("✓ Pipeline created successfully\n");

    // 4. Prepare input image
    int img_width = 1920;
    int img_height = 1080;
    int img_channels = 3;

    unsigned char* image_data = create_dummy_image(img_width, img_height, img_channels);
    if (!image_data) {
        printf("ERROR: Failed to create test image\n");
        c_yolopose_pipeline_destroy(ctx);
        return 1;
    }
    printf("✓ Test image created (%dx%dx%d)\n", img_width, img_height, img_channels);

    C_ImageInput image = {
        .data = image_data,
        .width = img_width,
        .height = img_height,
        .channels = img_channels
    };

    // 5. Run inference
    C_YoloPoseImageResult result;
    printf("\nRunning inference...\n");

    if (!c_yolopose_infer_single(ctx, &image, &result)) {
        const char* error = c_yolopose_pipeline_get_last_error(ctx);
        printf("ERROR: Inference failed: %s\n", error ? error : "unknown");
        free(image_data);
        c_yolopose_pipeline_destroy(ctx);
        return 1;
    }
    printf("✓ Inference completed\n");

    // 6. Print results
    print_results(&result);

    // 7. Cleanup
    c_yolopose_image_result_free(&result);
    free(image_data);
    c_yolopose_pipeline_destroy(ctx);

    printf("✓ Resources cleaned up\n\n");
    return 0;
}

// ============================================================================
//                         Example 2: Batch Inference
// ============================================================================

static int example_batch_inference(const char* engine_path) {
    printf("========================================\n");
    printf("Example 2: Batch Inference\n");
    printf("========================================\n\n");

    // 1. Create pipeline
    C_YoloPosePipelineConfig config = c_yolopose_pipeline_get_default_config();
    config.engine_path = engine_path;
    config.max_batch_size = 4;  // Process up to 4 images at once

    C_YoloPosePipelineContext* ctx = c_yolopose_pipeline_create(&config);
    if (!ctx) {
        printf("ERROR: Failed to create pipeline\n");
        return 1;
    }
    printf("✓ Pipeline created with batch size %d\n", config.max_batch_size);

    // 2. Prepare batch of images
    const int num_images = 3;
    C_ImageInput images[num_images];
    unsigned char* image_buffers[num_images];

    for (int i = 0; i < num_images; i++) {
        int width = 1280 + i * 100;  // Different sizes
        int height = 720 + i * 50;

        image_buffers[i] = create_dummy_image(width, height, 3);
        if (!image_buffers[i]) {
            printf("ERROR: Failed to create test image %d\n", i);
            // Cleanup previously allocated images
            for (int j = 0; j < i; j++) {
                free(image_buffers[j]);
            }
            c_yolopose_pipeline_destroy(ctx);
            return 1;
        }

        images[i].data = image_buffers[i];
        images[i].width = width;
        images[i].height = height;
        images[i].channels = 3;

        printf("✓ Image %d created (%dx%d)\n", i, width, height);
    }

    C_ImageBatch batch = {
        .images = images,
        .count = num_images
    };

    // 3. Run batch inference
    printf("\nRunning batch inference...\n");
    C_YoloPoseBatchResult result;

    if (!c_yolopose_infer_batch(ctx, &batch, &result)) {
        const char* error = c_yolopose_pipeline_get_last_error(ctx);
        printf("ERROR: Batch inference failed: %s\n", error ? error : "unknown");
        for (int i = 0; i < num_images; i++) {
            free(image_buffers[i]);
        }
        c_yolopose_pipeline_destroy(ctx);
        return 1;
    }
    printf("✓ Batch inference completed\n");

    // 4. Print results for all images
    for (size_t i = 0; i < result.num_images; i++) {
        print_results(&result.results[i]);
    }

    // 5. Cleanup
    c_yolopose_batch_result_free(&result);
    for (int i = 0; i < num_images; i++) {
        free(image_buffers[i]);
    }
    c_yolopose_pipeline_destroy(ctx);

    printf("✓ Resources cleaned up\n\n");
    return 0;
}

// ============================================================================
//                         Main Function
// ============================================================================

int main(int argc, char* argv[]) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  YOLO Pose Inference Pipeline - C API Example             ║\n");
    printf("║  TrtEngineToolkits v2                                      ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    // Check command line arguments
    const char* engine_path = "yolov8n-pose.engine";  // Default

    if (argc >= 2) {
        engine_path = argv[1];
    } else {
        printf("Usage: %s [engine_path]\n", argv[0]);
        printf("Using default engine path: %s\n\n", engine_path);
        printf("Note: This example uses dummy images for demonstration.\n");
        printf("      In production, load real image data from files.\n\n");
    }

    // Run examples
    int ret = 0;

    // Example 1: Single image
    ret = example_single_image(engine_path);
    if (ret != 0) {
        printf("Example 1 failed with code %d\n", ret);
        return ret;
    }

    // Example 2: Batch inference
    ret = example_batch_inference(engine_path);
    if (ret != 0) {
        printf("Example 2 failed with code %d\n", ret);
        return ret;
    }

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  All examples completed successfully!                      ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    return 0;
}
