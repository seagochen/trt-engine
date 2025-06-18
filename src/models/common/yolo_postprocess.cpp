//
// Created by user on 6/18/25.
//

#include <string>
#include <cuda_runtime.h>

// Simple CUDA Toolkits (SCT) specific headers
#include <simple_cuda_toolkits/tsutils/filter.h>
#include <simple_cuda_toolkits/tsutils/sort.h>
#include <simple_cuda_toolkits/tsutils/maxmin.h>
#include <simple_cuda_toolkits/matrix/matrix.h>

// Serverlet headers
#include "serverlet/models/common/yolo_postprocess.h"
#include "serverlet/utils/logger.h"

// Define DEBUG for conditional compilation (consider moving to a global config)
#ifndef DEBUG
#define DEBUG 0
#endif


// --- Utility Function for YOLO Post-processing (CUDA-based) ---
int inferPostProcForYolo(const float* ptr_device, std::vector<float>& output,
                              int features, int samples, float cls, bool use_pose)
{
    float* ptr_device_temp0 = nullptr;
    float* ptr_device_temp1 = nullptr;
    size_t total_size = (size_t)features * samples * sizeof(float);

    cudaError_t err0 = cudaMalloc(&ptr_device_temp0, total_size);
    if (err0 != cudaSuccess) {
        LOG_ERROR("sct_yolo_post_proc", "Failed to allocate CUDA memory for temp tensor 0: " + std::string(cudaGetErrorString(err0)));
        return -1; // Indicate failure
    }

    cudaError_t err1 = cudaMalloc(&ptr_device_temp1, total_size);
    if (err1 != cudaSuccess) {
        cudaFree(ptr_device_temp0); // Clean up previous allocation
        LOG_ERROR("sct_yolo_post_proc", "Failed to allocate CUDA memory for temp tensor 1: " + std::string(cudaGetErrorString(err1)));
        return -1; // Indicate failure
    }

    cudaMemcpy(ptr_device_temp0, ptr_device, total_size, cudaMemcpyDeviceToDevice);

#if DEBUG
    // Ensure sctDumpCudaMemoryToCSV is accessible if DEBUG is enabled
    // sctDumpCudaMemoryToCSV(ptr_device_temp0, "before_op.csv", features, samples);
#endif

    sctMatrixTranspose(ptr_device_temp0, ptr_device_temp1, features, samples);

#if DEBUG
    // sctDumpCudaMemoryToCSV(ptr_device_temp1, "transpose.csv", samples, features);
#endif

    // If not a pose model, perform classification processing (e.g., argmax)
    // IMPORTANT: The original code had sctArgmax commented out and just swapped pointers.
    // If classification (finding class ID and confidence) is truly needed for object detection,
    // you must uncomment and correctly use sctArgmax here.
    // Current logic: if !use_pose, it effectively just swaps the pointers,
    // which means ptr_device_temp1 will hold the transposed data, and ptr_device_temp0 will be unused.
    if (!use_pose)
    {
        // Example if sctArgmax is actually needed:
        // sctArgmax(ptr_device_temp1, ptr_device_temp0, samples, features, /* class_start_idx */ 4, features, /* class_id_output_idx */ 5, /* prob_output_idx */ 4);
        // std::swap(ptr_device_temp0, ptr_device_temp1); // If sctArgmax writes to temp0, then swap to put results in temp1
        sctArgmax_dim1(ptr_device_temp0, ptr_device_temp1, samples, features, 4, 5, features - 5, 4, 5);

        // If no argmax/classification is intended for object detection, and the swap is merely
        // to move transposed data into ptr_device_temp1, this is fine.
        std::swap(ptr_device_temp0, ptr_device_temp1); // Faster than a memcpy
    }

    // Filter results based on confidence threshold
    int results = sctFilterGreater_dim1(
        ptr_device_temp1, // Input data (transposed, possibly with classification)
        ptr_device_temp0, // Output buffer for filtered results
        4,                // Dimension index for confidence score (e.g., 4th column for x,y,w,h,conf,...)
        cls,              // Confidence threshold
        samples,          // Total samples to check
        features          // Number of features per sample
    );

#if DEBUG
    // sctDumpCudaMemoryToCSV(ptr_device_temp0, "filter.csv", samples, features);
#endif

    if (results > 0)
    {
        // Sort results by confidence in descending order
        // Note: Sort only 'results' valid items, not 'samples' total items
        sctSortTensor_dim1_descending(ptr_device_temp0, ptr_device_temp1, results, features, 4);

#if DEBUG
        // sctDumpCudaMemoryToCSV(ptr_device_temp1, "sort.csv", results, features);
#endif

        // Copy processed results from device to host
        // Ensure output vector is large enough, resize if necessary
        output.resize((size_t)results * features);
        cudaMemcpy(output.data(), ptr_device_temp1, (size_t)results * features * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        results = -1; // Indicate no valid results found after filtering
    }

    // Clean up CUDA memory
    cudaFree(ptr_device_temp0);
    cudaFree(ptr_device_temp1);

    return results; // Return number of valid results
}