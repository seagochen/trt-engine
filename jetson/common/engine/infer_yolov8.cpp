#include "infer_yolov8.h"
#include "common/utils/logger.h"


InferYoloObjectv8::InferYoloObjectv8(const std::string& engine_path,
                                     const std::map<std::string, std::string>& names,
                                     const std::vector<int>& input_dims,  // shape: [batch, channels, height, width]
                                     const std::vector<int>& output_dims, // shape: [batch, features, targets]
                                     const int num_targets):
    InferModelBase(engine_path, names, input_dims, output_dims), batch_idx(0) {
    // Initialize the number of targets
    this->num_targets = num_targets;

    // Allocate images for resizing and normalization
    resized_image = cv::Mat(input_dims[2], input_dims[3], CV_8UC3);
    float_data = cv::Mat(input_dims[2], input_dims[3], CV_32FC3);
    LOG_VERBOSE_TOPIC("InferYoloObjectv8", "InferYoloObjectv8", "InferYoloObjectv8 object created successfully.");

    // Allocate bitmap for NMS
    sctAllocateNMSBitmap(num_targets);
    LOG_VERBOSE_TOPIC("InferYoloObjectv8", "InferYoloObjectv8", "NMS bitmap allocated successfully.");

    // Allocate local buffer for output
    local_buffer = new float[output_dims[1] * output_dims[2]];
    LOG_VERBOSE_TOPIC("InferYoloObjectv8", "InferYoloObjectv8", "Local buffer allocated successfully.");
}


InferYoloObjectv8::~InferYoloObjectv8() {
    // Release the images
    resized_image.release();
    float_data.release();
    LOG_VERBOSE_TOPIC("InferYoloObjectv8", "~InferYoloObjectv8", "InferYoloObjectv8 object destroyed successfully.");

    // Release the NMS bitmap
    sctFreeNMSBitmap();
    LOG_VERBOSE_TOPIC("InferYoloObjectv8", "~InferYoloObjectv8", "NMS bitmap released successfully.");

    // Release the local buffer
    delete[] local_buffer;
    LOG_VERBOSE_TOPIC("InferYoloObjectv8", "~InferYoloObjectv8", "Local buffer released successfully.");
}


std::any InferYoloObjectv8::infer(const cv::Mat &image) {
    // Preprocess the image
    preprocess(image);

    // Increment the batch index
    ++batch_idx;

    // If the batch index is less than the batch size, return an empty vector
    if (batch_idx < input_dims[0]) { // the batchidx will be [1, 2, 3, ..., batch_size]
        return std::vector<std::vector<float>>();
    }

    // Run inference on the TensorRT engine
    fireEngine();

    // Postprocess the output
    auto output = postprocess();

    // Reset the batch index
    batch_idx = 0;

    return output;
}


void InferYoloObjectv8::preprocess(const cv::Mat& image) {

    // Resize the image to the input dimensions
    cv::resize(image, resized_image, cv::Size(input_dims[3], input_dims[2]));

    // Convert the image to float data
    resized_image.convertTo(float_data, CV_32FC3);

    // Copy the float data to the cuda input buffer
    cudaMemcpy(cuda_input_buffers[0].ptr(), float_data.ptr(), 
        float_data.total() * float_data.elemSize(), cudaMemcpyHostToDevice);

    // Get the width, height, and channels of the input tensor
    int width = input_dims[3];
    int height = input_dims[2];
    int channels = input_dims[1];
    int total_size = width * height * channels;

    // Permute the input tensor to [batch, channels, height, width]
    sctPermute3D(cuda_input_buffers[0].ptr(), cuda_input_buffers[1].ptr(), width, height, channels, 2, 0, 1);

    // Normalize the input tensor
    sctNormalizeData(cuda_input_buffers[1].ptr(), cuda_input_buffers[0].ptr(), width, height, channels);

    // Copy the normalized data to the tensorrt engine buffer
    int offset = batch_idx * total_size;
    loadDataToEngine(cuda_input_buffers[0], total_size * sizeof(float), offset);
}


// Function to decode the raw output from the model
std::vector<Yolo> decode(float* raw, int boxes, int features) {

    std::vector<Yolo> results;

    for (int i = 0; i < boxes; i++) {
        if (raw[i * features + 4] > 0.0) {
            Yolo result;

            result.lx = int(raw[i * features + 0]);
            result.ly = int(raw[i * features + 1]);
            result.rx = int(raw[i * features + 2]);
            result.ry = int(raw[i * features + 3]);
            result.conf = raw[i * features + 4];
            result.cls = int(raw[i * features + 5]);

            results.push_back(result);
        }
    }

    return results;
}


std::vector<std::vector<Yolo>> InferYoloObjectv8::postprocess(float cls, float nms, float alpha, float beta)
{
    // Get the features, and samples
    int features = output_dims[1];
    int samples = output_dims[2];

    // Declare the output vector
    std::vector<std::vector<Yolo>> output;

    // Copy the output data from the tensorrt engine
    for (int i = 0; i < batch_idx; i++) {
        int offset = i * features * samples;
        loadDataFromEngine(cuda_output_buffers[0], features * samples * sizeof(float), offset);

        // Transpose the output tensor to [batch, targets, features]
        sctMatrixTranspose(cuda_output_buffers[0].ptr(), cuda_output_buffers[1].ptr(), features, samples);

        // Find the maximum value in the output tensor
        sctFindMaxScores(cuda_output_buffers[1].ptr(), cuda_output_buffers[0].ptr(), features, samples);

        // Suppress the bounding boxes
        sctSuppressResults_closed(cuda_output_buffers[0].ptr(), cuda_output_buffers[1].ptr(), cls, features, samples);

        // Convert the bounding boxes from XYWH to XYXY
        sctXYWH2XYXY(cuda_output_buffers[1].ptr(), cuda_output_buffers[0].ptr(), features, samples, alpha, beta);

        // Perform non-maximum suppression
        sctNMS_V2(cuda_output_buffers[0].ptr(), cuda_output_buffers[1].ptr(), nms, features, samples, true); // Only suppress the top boxes

        // Copy the device output to the host
        cudaMemcpy(local_buffer, cuda_output_buffers[1].ptr(), features * samples * sizeof(float), cudaMemcpyDeviceToHost);

        // Parsing the output
        std::vector<Yolo> results = decode(local_buffer, num_targets, features);
        output.push_back(results);
    }

    return output;
}
