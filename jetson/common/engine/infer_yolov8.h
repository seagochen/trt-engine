//
// Created by orlando on 12/18/24.
//

#ifndef INFER_DEPTH_ANYTHING_V2_H
#define INFER_DEPTH_ANYTHING_V2_H

#include "common/engine/infer_model_base.h"
#include "common/yolo/yolo_def.h"

#include <simple_cuda_toolkits/vision/colorspace.h>
#include <simple_cuda_toolkits/vision/normalization.h>
#include <simple_cuda_toolkits/tsutils/permute_3D.h>
#include <simple_cuda_toolkits/bbox/bbox.h>
#include <simple_cuda_toolkits/bbox/find_max.h>
#include <simple_cuda_toolkits/bbox/suppress.h>
#include <simple_cuda_toolkits/bbox/nms.h>
#include <simple_cuda_toolkits/matrix/matrix.h>

#include <opencv2/opencv.hpp>


class InferYoloObjectv8 : public InferModelBase {

    int num_targets;            // Maximum number of targets to detect
    int batch_idx;              // Batch index for multiple images
    cv::Mat resized_image;      // Resized image for normalization
    cv::Mat float_data;         // Float data for normalization
    float* local_buffer;        // Local buffer for output

public:

    // Constructor and destructor
    explicit InferYoloObjectv8(const std::string& engine_path,
                                const std::map<std::string, std::string>& names,
                                const std::vector<int>& input_dims,  // shape: [batch, channels, height, width]
                                const std::vector<int>& output_dims, // shape: [batch, features, targets]
                                const int num_targets = 512);

    // Destructor
    ~InferYoloObjectv8() override;

    // Run inference
    std::any infer(const cv::Mat& image) override;

private:

    // Preprocess the image
    void preprocess(const cv::Mat& image);

    // Postprocess the output
    std::vector<std::vector<Yolo>> postprocess(float cls=0.5, float nms=0.5,  float alpha=0.f, float beta=640.f);
};



#endif //INFER_DEPTH_ANYTHING_V2_H
