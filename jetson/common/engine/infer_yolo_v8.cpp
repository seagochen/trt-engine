//
// Created by orlando on 9/20/24.
//

#include "infer_yolo_v8.h"
#include "common/utils/logger.h"

#include <simple_cuda_toolkits/vision/colorspace.h>
#include <simple_cuda_toolkits/vision/normalization.h>
#include <simple_cuda_toolkits/tsutils/permute_3D.h>
#include <simple_cuda_toolkits/bbox/bbox.h>
#include <simple_cuda_toolkits/bbox/find_max.h>
#include <simple_cuda_toolkits/bbox/suppress.h>
#include <simple_cuda_toolkits/bbox/nms.h>
#include <simple_cuda_toolkits/matrix/matrix.h>


// Custom deleter for ICudaEngine
auto engineDeleter0 = [](nvinfer1::ICudaEngine* engine) {
    if (engine) {
        engine->destroy();
    }
};


// Custom deleter for IExecutionContext
auto contextDeleter0 = [](nvinfer1::IExecutionContext* context) {
    if (context) {
        context->destroy();
    }
};


InferYoloV8::InferYoloV8():engine(nullptr, engineDeleter0), context(nullptr, contextDeleter0){}

InferYoloV8::InferYoloV8(const std::string &engine_path,
            const std::map<std::string, std::string> &names,
            const nvinfer1::Dims4 &dims0,
            const nvinfer1::Dims3 &dims1,
            const int boxes):
        engine(nullptr, engineDeleter0), context(nullptr, contextDeleter0) {

    // Update the wrapper
    update(engine_path, names, dims0, dims1, boxes);
}


void InferYoloV8::update(const std::string &engine_path,
    const std::map<std::string, std::string> &names,
    const nvinfer1::Dims4 &dims0,
    const nvinfer1::Dims3 &dims1,
    int boxes) {

    // Load the TensorRT engine from the serialized engine file
    engine = loadEngineFromFile(engine_path);
    if (!engine) {
        LOG_ERROR_TOPIC("InferWrapper", "Engine", "Failed to load engine from file.");
        exit(EXIT_FAILURE);
    }

    // Create a context for executing the engine
    context = createExecutionContext(engine, names.at("input"), dims0);
    LOG_VERBOSE_TOPIC("InferWrapper", "Engine", "Context created successfully.");

    // Dimensions of input and output tensors
    input_dims = {dims0.d[0], dims0.d[1], dims0.d[2], dims0.d[3]}; // batch, channels, height, width
    output_dims = {dims1.d[0], dims1.d[1], dims1.d[2]}; // batch, features, samples
    this->boxes = boxes;

    // Allocate input and output buffers for CUDA
    std::map<std::string, std::vector<int>> trt_binding_dims;
    trt_binding_dims[names.at("input")] = input_dims;
    trt_binding_dims[names.at("output")] = output_dims;
    trt_buffers = allocateCudaTensors(trt_binding_dims);
    tensor_names = names;
    LOG_VERBOSE_TOPIC("InferWrapper", "TRTBuffer", "Buffers for TensorRT engine are ready.");

    // Allocate input and output buffers for CUDA
    cuda_input_buffers[0] = createZerosTensor<TensorType::FLOAT32>(input_dims[1] * input_dims[2] * input_dims[3]);
    cuda_input_buffers[1] = createZerosTensor<TensorType::FLOAT32>(input_dims[1] * input_dims[2] * input_dims[3]);
    cuda_output_buffers[0] = createZerosTensor<TensorType::FLOAT32>(output_dims[1] * output_dims[2]);
    cuda_output_buffers[1] = createZerosTensor<TensorType::FLOAT32>(output_dims[1] * output_dims[2]);
    LOG_VERBOSE_TOPIC("InferWrapper", "CUDABuffer", "Temporary buffers for CUDA are ready.");

    // Allocate temporary images for OpenCV
    temp_images["floated"] = cv::Mat(input_dims[2], input_dims[3], CV_32FC3);
    LOG_VERBOSE_TOPIC("InferWrapper", "OpenCVBuffer", "Temporary images for OpenCV are ready.");

    // Allocate Bitmap for NMS
    sctAllocateNMSBitmap(this->boxes);
    LOG_VERBOSE_TOPIC("InferWrapper", "NMS", "Bitmap for NMS is ready.");

    // Allocate results buffer
    int batch_size = (output_dims[0] > MAX_BATCH_SIZE) ? MAX_BATCH_SIZE : output_dims[0];
    for (int i = 0; i < batch_size; ++i) {
        results.emplace_back(createZerosTensor<TensorType::FLOAT32>(output_dims[1] * output_dims[2]));
    }
    LOG_VERBOSE_TOPIC("InferWrapper", "Output", "Reference batch size has been set.");
}


InferYoloV8::~InferYoloV8() {
    // Release resources
    trt_buffers.clear();
    cuda_input_buffers.clear();
    cuda_output_buffers.clear();
    temp_images.clear();

    // Destroy the execution context
    context.reset();

    // Destroy the engine
    engine.reset();

    // Release the results buffer
    sctFreeNMSBitmap();

    // std::cout << "[InferWrapper/End] VERBOSE: InferWrapper destructor called." << std::endl;
    LOG_VERBOSE_TOPIC("InferWrapper", "End", "InferWrapper destructor called.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* @brief Preprocess input image for inference
* @param image Input image for inference
*/
void InferYoloV8::addImage(const cv::Mat &image, bool isRGB) {
    // Check if the image index is within the batch size
    if (image_idx >= MAX_BATCH_SIZE) {
        // throw std::runtime_error("The batch size is exceeded.");
        LOG_ERROR_TOPIC("InferWrapper", "Preprocess", "The batch size is exceeded.");
        exit(EXIT_FAILURE);
    }

    // If the image's width or height is not equal to the input tensor's width or height
    if (image.cols != input_dims[3] || image.rows != input_dims[2]) {
        // throw std::runtime_error("The input image's width or height is not equal to the input tensor's width or height.");
        LOG_ERROR_TOPIC("InferWrapper", "Preprocess",
            "The input image's width or height is not equal to the input tensor's width or height.");
        exit(EXIT_FAILURE);
    }

    // Convert the image the float
    image.convertTo(temp_images["floated"], CV_32FC3);

    // Width, height, and channels
    int width = input_dims[3];
    int height = input_dims[2];
    int channels = input_dims[1];

    // PTR for the input buffers
    auto ptr0 = cuda_input_buffers[0].ptr();
    auto ptr1 = cuda_input_buffers[1].ptr();

    // Copy the image to the input buffer
    cudaMemcpy(ptr0, temp_images["floated"].data,
            channels * height * width * sizeof(float), cudaMemcpyHostToDevice);

    // If the image is in BGRA format, convert it to RGB
    if (!isRGB) {
        sctBGR2RGB(ptr0, ptr1, width, height, channels);
        sctNormalizeData(ptr1, ptr0, width, height, channels);
    } else {
        sctNormalizeData(ptr0, ptr0, width, height, channels);
    }

    // Permute the input tensor from HWC to CHW (BCHW for YOLO)
    sctPermute3D(ptr0, ptr1, width, height, channels, 2, 0, 1);

    // Copy the input tensor to the TensorRT buffer
    int offset = channels * height * width * image_idx;
    cudaMemcpy(trt_buffers[tensor_names.at("input")].ptr() + offset, ptr1,
            channels * height * width * sizeof(float), cudaMemcpyDeviceToDevice);

    // Increment the image index
    image_idx++;
}

/**
 * @brief Perform inference on the input images
 */
void InferYoloV8::inferObjectDetection(float cls_threshold, float nms_threshold, float alpha, float beta) {
    // Check if the image index is within the batch size
    if (image_idx == 0) {
        // throw std::runtime_error("No image is preprocessed.");
        LOG_ERROR_TOPIC("InferWrapper", "Inference", "No image is preprocessed.");
        exit(EXIT_FAILURE);
    }
    if (image_idx > results.size()) {
        // throw std::runtime_error("The batch size is exceeded.");
        LOG_ERROR_TOPIC("InferWrapper", "Inference", "The batch size is exceeded.");
        exit(EXIT_FAILURE);
    }

    // PTR for the output buffers
    auto ptr0 = cuda_output_buffers[0].ptr();
    auto ptr1 = cuda_output_buffers[1].ptr();

    // Features and samples
    int features = output_dims[1];
    int samples = output_dims[2];

    // Execute the inference
    inference(context, trt_buffers[tensor_names.at("input")], trt_buffers[tensor_names.at("output")]);

    // Postprocess the output tensor
    for (int i = 0; i < image_idx; ++i) {
        // Copy the output tensor to the CUDA buffer
        checkCudaErrors(cudaMemcpy(ptr0, trt_buffers[tensor_names.at("output")].ptr() + i * features * samples,
                features * samples * sizeof(float), cudaMemcpyDeviceToDevice),
                "CUDA cudaMemcpyDeviceToDevice failed.");

        // Transpose the output tensor
        sctMatrixTranspose(ptr0, ptr1, features, samples);

        // Find the maximum value in the tensor
        sctFindMaxScores(ptr1, ptr0, features, samples);

        // Suppress the bounding boxes
        sctSuppressResults_closed(ptr0, ptr1, cls_threshold, features, samples);

        // Convert XYWH to XYXY
        sctXYWH2XYXY(ptr1, ptr0, features, samples, alpha, beta);

        // Perform non-maximum suppression
        sctNMS_V2(ptr0, ptr1, nms_threshold, features, boxes, true);  // Only suppress the top boxes

        // Copy the output tensor to the result buffer
        cudaMemcpy(results[i].ptr(), ptr1, features * boxes * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Reset the image index
    image_idx = 0;
}


void InferYoloV8::inferPoseEstimation(float cls_threshold, float nms_threshold, float alpha, float beta) {
    // Check if the image index is within the batch size
    if (image_idx == 0) {
        throw std::runtime_error("No image is preprocessed.");
    }
    if (image_idx > results.size()) {
        throw std::runtime_error("The batch size is exceeded.");
    }

    // PTR for the output buffers
    auto ptr0 = cuda_output_buffers[0].ptr();
    auto ptr1 = cuda_output_buffers[1].ptr();

    // Features and samples
    int features = output_dims[1];
    int samples = output_dims[2];

    // Execute the inference
    inference(context, trt_buffers[tensor_names.at("input")], trt_buffers[tensor_names.at("output")]);

    // Postprocess the output tensor
    for (int i = 0; i < image_idx; ++i) {
        // Copy the output tensor to the CUDA buffer
        checkCudaErrors(cudaMemcpy(ptr0, trt_buffers[tensor_names.at("output")].ptr() + i * features * samples,
                features * samples * sizeof(float), cudaMemcpyDeviceToDevice),
                "CUDA cudaMemcpyDeviceToDevice failed.");

        // Transpose the output tensor
        sctMatrixTranspose(ptr0, ptr1, features, samples);

        // Suppress the output
        sctSuppressResults_closed(ptr1, ptr0, cls_threshold, features, samples);

        // Convert XYWH to XYXY
        sctXYWH2XYXY(ptr0, ptr1, features, samples, alpha, beta);

        // Perform non-maximum suppression
        sctNMS_V2(ptr1, ptr0, nms_threshold, features, boxes);  // Only suppress the top boxes

        // Copy the output tensor to the result buffer
        cudaMemcpy(results[i].ptr(), ptr0, features * boxes * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Reset the image index
    image_idx = 0;
}

/**
 * @brief Get the available slots count for storing preprocessed images
 * @return Number of available slots
 */
int InferYoloV8::getAvailableSlot() const {
    return results.size() - image_idx;
}