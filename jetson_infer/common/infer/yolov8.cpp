//
// Created by ubuntu on 9/12/24.
//

#include "common/infer/yolov8.h"

#include <cores/cores.h>
#include <cuda_runtime.h>
#include <tensor_utils.hpp>
#include <vision/colorspace.h>
#include <vision/hwc2chw.h>
#include <vision/normalization.h>
#include <yolo8utils/yolov8.h>


// Hide this tensor to improve speed
int g_width = 0, g_height = 0, g_channels = 0;
void *ptrCudaRawData = nullptr, *ptrCudaFloatData = nullptr;

auto g_gpu_tensor = createZerosCudaTensor<float>(84, 8400);
auto g_cpu_tensor = createZerosCpuTensor<float>(84, 8400);

// Memory initialization function
void initCudaTemporaryBuffer(int width, int height, int channels) {
    releaseCudaTemporaryBuffer();  // Clean up old buffers if needed
    checkCudaError(cudaMalloc(&ptrCudaRawData, width * height * channels * sizeof(uchar)), "init ptrCudaRawData failed");
    checkCudaError(cudaMalloc(&ptrCudaFloatData, width * height * channels * sizeof(float)), "init ptrCudaFloatData failed");

    g_width = width;
    g_height = height;
    g_channels = channels;
}

// Memory release function
void releaseCudaTemporaryBuffer() {
    if (ptrCudaRawData) cudaFree(ptrCudaRawData);
    if (ptrCudaFloatData) cudaFree(ptrCudaFloatData);
}

// Preprocessing function for input image
void preprocess(cv::Mat &image, CudaTensor<float> &output) {
    // Resize the image to the target size
    cv::resize(image, image, cv::Size(g_width, g_height));

    // Copy the image data to the GPU
    cudaMemcpy(ptrCudaRawData, image.data, g_width * g_height * g_channels * sizeof(uchar), cudaMemcpyHostToDevice);

    // Use the sct kernel to convert the image to the correct format
    sctConvertUInt8ToFloat(ptrCudaRawData, ptrCudaFloatData, g_width, g_height, g_channels);

    // Convert the color space from BGR to RGB
    sctBGR2RGB(ptrCudaFloatData, ptrCudaFloatData, g_width, g_height, g_channels);

    // Normalize the image data
    sctNormalizeData(ptrCudaFloatData, ptrCudaFloatData, g_width, g_height, g_channels);

    // Convert the HWC format to CHW format
    sctHWC2CHW(ptrCudaFloatData, output.ptr(), g_width, g_height, g_channels);
}

// Postprocess helper to extract box and confidence information
YoloResult extractYoloResult(const std::vector<float> &data, int index, int features) {
    YoloResult result;
    result.lx = static_cast<int>(data[index * features]);
    result.ly = static_cast<int>(data[index * features + 1]);
    result.rx = static_cast<int>(data[index * features + 2]);
    result.ry = static_cast<int>(data[index * features + 3]);
    result.cls = static_cast<int>(data[index * features + 5]);
    result.conf = data[index * features + 4];
    return result;
}

// Object detection postprocessing
void obj_postprocess(CudaTensor<float> &output, float confidence, std::vector<YoloResult> &results) {
    sctYolov8ObjectPostProcessing(output.ptr(), g_gpu_tensor.ptr(), 84, 8400, confidence);
    g_cpu_tensor.copyFrom(g_gpu_tensor);
    const std::vector<float> &data = g_cpu_tensor.getData();

    results.clear();
    for (int i = 0; i < 8400; ++i) {
        if (data[i * 84 + 4] > confidence) {
            results.push_back(extractYoloResult(data, i, 84));
        }
    }
}

// Pose postprocessing with keypoints
void pose_postprocess(CudaTensor<float> &output, float confidence, std::vector<YoloResult> &results) {
    sctYolov8PosePostProcessing(output.ptr(), g_gpu_tensor.ptr(), 56, 8400, confidence);
    g_cpu_tensor.copyFrom(g_gpu_tensor);
    const std::vector<float> &data = g_cpu_tensor.getData();

    results.clear();
    for (int i = 0; i < 8400; ++i) {
        if (data[i * 56 + 4] > confidence) {
            YoloResult pose_result = extractYoloResult(data, i, 56);

            // Keypoints extraction
            for (int j = 0; j < 17; ++j) {
                YoloPoint keypoint{
                        static_cast<int>(data[i * 56 + 5 + j * 3]),
                        static_cast<int>(data[i * 56 + 5 + j * 3 + 1]),
                        data[i * 56 + 5 + j * 3 + 2]
                };
                pose_result.keypoints.push_back(keypoint);
            }
            results.push_back(pose_result);
        }
    }
}


void postprocess(const CudaTensor<float> &input, std::vector<YoloResult> &output,
                 float confidence, std::string model) {

    // Object detection postprocessing
    if (model == "yolov8") {
        obj_postprocess(const_cast<CudaTensor<float> &>(input), confidence, output);
    } else if (model == "yolov8-pose") {
        pose_postprocess(const_cast<CudaTensor<float> &>(input), confidence, output);
    }
}


// YoloPoint のシリアライズ関数
void to_json(const YoloPoint& p, nlohmann::json& j) {
    j = nlohmann::json{{"x", p.x}, {"y", p.y}, {"conf", p.conf}};
}

// YoloResult のシリアライズ関数
void to_json(const YoloResult& r, nlohmann::json& j) {
    // Convert r.keypoints to nlohmann::json
    std::vector<nlohmann::json> keypoints;
    for (const auto &p : r.keypoints) {
        nlohmann::json j_p;
        to_json(p, j_p);
        keypoints.push_back(j_p);
    }

    // Convert r to nlohmann::json
    j = nlohmann::json{
            {"lx", r.lx}, {"ly", r.ly}, {"rx", r.rx}, {"ry", r.ry},
            {"cls", r.cls}, {"conf", r.conf}, {"keypoints", keypoints}
    };
}

// YoloResult のデシリアライズ関数
void from_json(const std::string &str, YoloResult &r) {
    nlohmann::json j = nlohmann::json::parse(str);
    r.lx = j["lx"];
    r.ly = j["ly"];
    r.rx = j["rx"];
    r.ry = j["ry"];
    r.cls = j["cls"];
    r.conf = j["conf"];
    r.keypoints = j["keypoints"].get<std::vector<YoloPoint>>();
}

// YoloPoint のデシリアライズ関数
void from_json(const std::string &str, YoloPoint &p) {
    nlohmann::json j = nlohmann::json::parse(str);
    p.x = j["x"];
    p.y = j["y"];
    p.conf = j["conf"];
}

// vector<YoloResult> のシリアライズ関数
std::string to_json(const std::vector<YoloResult>& results) {
    // Convert results to nlohmann::json
    std::vector<nlohmann::json> j_results;
    for (const auto &r : results) {
        nlohmann::json j_r;
        to_json(r, j_r);
        j_results.push_back(j_r);
    }

    // Convert j_results to string
    nlohmann::json j = j_results;
    return j.dump();
}

// vector<YoloResult> のデシリアライズ関数
void from_json(const std::string &str, std::vector<YoloResult> &results) {
    nlohmann::json j = nlohmann::json::parse(str);
    results = j.get<std::vector<YoloResult>>();
}
