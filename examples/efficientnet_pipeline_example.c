/**
 * @file efficientnet_pipeline_example.c
 * @brief Example usage of EfficientNet inference pipeline
 *
 * This example demonstrates how to use the EfficientNet pipeline for:
 * 1. Single image inference
 * 2. Batch image inference
 * 3. Extracting both classification results and feature embeddings
 *
 * @author TrtEngineToolkits
 * @date 2025-11-10
 */

#include "trtengine_v2/pipelines/efficientnet/c_efficientnet_pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * @brief Load a dummy image for testing
 */
static bool load_dummy_image(C_ImageInput* image, int width, int height) {
    image->width = width;
    image->height = height;
    image->channels = 3;

    size_t size = width * height * 3;
    image->data = (unsigned char*)malloc(size);

    if (!image->data) {
        return false;
    }

    // Fill with dummy gradient data
    for (int i = 0; i < size; i++) {
        image->data[i] = (unsigned char)(i % 256);
    }

    return true;
}

/**
 * @brief Example 1: Single image inference
 */
static void example_single_inference(const char* engine_path) {
    printf("\n=== Example 1: Single Image Inference ===\n");

    // 1. Get default configuration
    C_EfficientNetPipelineConfig config = c_efficientnet_pipeline_get_default_config();
    config.engine_path = engine_path;

    // 2. Create pipeline
    printf("Creating EfficientNet pipeline...\n");
    C_EfficientNetPipelineContext* pipeline = c_efficientnet_pipeline_create(&config);
    if (!pipeline) {
        printf("Failed to create pipeline!\n");
        return;
    }
    printf("Pipeline created successfully.\n");

    // 3. Load input image
    C_ImageInput image;
    if (!load_dummy_image(&image, 640, 480)) {
        printf("Failed to load image!\n");
        c_efficientnet_pipeline_destroy(pipeline);
        return;
    }
    printf("Loaded image: %dx%d\n", image.width, image.height);

    // 4. Run inference
    C_EfficientNetResult result = {0};
    printf("Running inference...\n");
    if (!c_efficientnet_infer_single(pipeline, &image, &result)) {
        const char* error = c_efficientnet_pipeline_get_last_error(pipeline);
        printf("Inference failed: %s\n", error ? error : "Unknown error");
        free(image.data);
        c_efficientnet_pipeline_destroy(pipeline);
        return;
    }

    // 5. Display results
    printf("\n--- Classification Result ---\n");
    printf("Predicted Class: %d\n", result.class_id);
    printf("Confidence: %.4f\n", result.confidence);

    printf("\nLogits (%zu classes):\n", result.num_classes);
    for (size_t i = 0; i < result.num_classes; i++) {
        printf("  Class %zu: %.4f\n", i, result.logits[i]);
    }

    printf("\nFeature Vector (first 10 of %zu dimensions):\n", result.feature_size);
    for (size_t i = 0; i < 10 && i < result.feature_size; i++) {
        printf("  [%zu]: %.4f\n", i, result.features[i]);
    }

    // 6. Cleanup
    c_efficientnet_result_free(&result);
    free(image.data);
    c_efficientnet_pipeline_destroy(pipeline);

    printf("\nExample 1 completed successfully.\n");
}

/**
 * @brief Example 2: Batch inference
 */
static void example_batch_inference(const char* engine_path) {
    printf("\n=== Example 2: Batch Image Inference ===\n");

    // 1. Configuration with larger batch size
    C_EfficientNetPipelineConfig config = c_efficientnet_pipeline_get_default_config();
    config.engine_path = engine_path;
    config.max_batch_size = 4;  // Process 4 images at once

    // 2. Create pipeline
    printf("Creating EfficientNet pipeline (batch size: %d)...\n", config.max_batch_size);
    C_EfficientNetPipelineContext* pipeline = c_efficientnet_pipeline_create(&config);
    if (!pipeline) {
        printf("Failed to create pipeline!\n");
        return;
    }

    // 3. Prepare batch of images
    const int batch_size = 3;
    C_ImageBatch batch;
    batch.count = batch_size;
    batch.images = (C_ImageInput*)calloc(batch_size, sizeof(C_ImageInput));

    printf("Loading %d images...\n", batch_size);
    for (int i = 0; i < batch_size; i++) {
        if (!load_dummy_image(&batch.images[i], 640, 480)) {
            printf("Failed to load image %d!\n", i);
            // Cleanup
            for (int j = 0; j < i; j++) {
                free(batch.images[j].data);
            }
            free(batch.images);
            c_efficientnet_pipeline_destroy(pipeline);
            return;
        }
    }

    // 4. Run batch inference
    C_EfficientNetBatchResult results = {0};
    printf("Running batch inference...\n");
    if (!c_efficientnet_infer_batch(pipeline, &batch, &results)) {
        const char* error = c_efficientnet_pipeline_get_last_error(pipeline);
        printf("Batch inference failed: %s\n", error ? error : "Unknown error");
        // Cleanup
        for (int i = 0; i < batch_size; i++) {
            free(batch.images[i].data);
        }
        free(batch.images);
        c_efficientnet_pipeline_destroy(pipeline);
        return;
    }

    // 5. Display results
    printf("\n--- Batch Results (%zu images) ---\n", results.count);
    for (size_t i = 0; i < results.count; i++) {
        printf("\nImage %zu:\n", i);
        printf("  Class: %d (confidence: %.4f)\n",
               results.results[i].class_id,
               results.results[i].confidence);
        printf("  Feature norm: ");

        // Calculate L2 norm of feature vector
        float norm = 0.0f;
        for (size_t j = 0; j < results.results[i].feature_size; j++) {
            float val = results.results[i].features[j];
            norm += val * val;
        }
        norm = sqrtf(norm);
        printf("%.4f\n", norm);
    }

    // 6. Cleanup
    c_efficientnet_batch_result_free(&results);
    for (int i = 0; i < batch_size; i++) {
        free(batch.images[i].data);
    }
    free(batch.images);
    c_efficientnet_pipeline_destroy(pipeline);

    printf("\nExample 2 completed successfully.\n");
}

/**
 * @brief Example 3: Feature similarity comparison
 */
static void example_feature_similarity(const char* engine_path) {
    printf("\n=== Example 3: Feature Similarity Comparison ===\n");

    // Create pipeline
    C_EfficientNetPipelineConfig config = c_efficientnet_pipeline_get_default_config();
    config.engine_path = engine_path;

    C_EfficientNetPipelineContext* pipeline = c_efficientnet_pipeline_create(&config);
    if (!pipeline) {
        printf("Failed to create pipeline!\n");
        return;
    }

    // Load two images
    C_ImageInput image1, image2;
    if (!load_dummy_image(&image1, 640, 480) || !load_dummy_image(&image2, 640, 480)) {
        printf("Failed to load images!\n");
        c_efficientnet_pipeline_destroy(pipeline);
        return;
    }

    // Run inference on both images
    C_EfficientNetResult result1 = {0}, result2 = {0};
    if (!c_efficientnet_infer_single(pipeline, &image1, &result1) ||
        !c_efficientnet_infer_single(pipeline, &image2, &result2)) {
        printf("Inference failed!\n");
        free(image1.data);
        free(image2.data);
        c_efficientnet_pipeline_destroy(pipeline);
        return;
    }

    // Calculate cosine similarity between feature vectors
    float dot_product = 0.0f;
    float norm1 = 0.0f, norm2 = 0.0f;

    for (size_t i = 0; i < result1.feature_size; i++) {
        dot_product += result1.features[i] * result2.features[i];
        norm1 += result1.features[i] * result1.features[i];
        norm2 += result2.features[i] * result2.features[i];
    }

    float cosine_sim = dot_product / (sqrtf(norm1) * sqrtf(norm2));

    printf("\nFeature Similarity Analysis:\n");
    printf("Image 1 class: %d\n", result1.class_id);
    printf("Image 2 class: %d\n", result2.class_id);
    printf("Cosine similarity: %.4f\n", cosine_sim);
    printf("(1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)\n");

    // Cleanup
    c_efficientnet_result_free(&result1);
    c_efficientnet_result_free(&result2);
    free(image1.data);
    free(image2.data);
    c_efficientnet_pipeline_destroy(pipeline);

    printf("\nExample 3 completed successfully.\n");
}

/**
 * @brief Main function
 */
int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <path_to_efficientnet_engine>\n", argv[0]);
        printf("\nExample:\n");
        printf("  %s /opt/models/efficientnet_b0_feat_logits.engine\n", argv[0]);
        return 1;
    }

    const char* engine_path = argv[1];

    printf("========================================\n");
    printf("EfficientNet Pipeline Examples\n");
    printf("========================================\n");
    printf("Engine: %s\n", engine_path);

    // Run examples
    example_single_inference(engine_path);
    example_batch_inference(engine_path);
    example_feature_similarity(engine_path);

    printf("\n========================================\n");
    printf("All examples completed!\n");
    printf("========================================\n");

    return 0;
}
