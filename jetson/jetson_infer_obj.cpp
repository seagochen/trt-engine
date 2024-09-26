//
// Created by orlando on 9/25/24.
//

#include <opencv2/opencv.hpp>
#include <csignal>
#include <iostream>
#include <filesystem>
#include <map>
#include <string>
#include <memory>
#include <unistd.h>

#include "common/engine/infer_wrapper.h"
#include "common/utils/fps_counter.hpp"
#include "common/yolo/yolo_def.h"
#include "common/yolo/yolo_visualization.h"
#include "common/yolo/load_labels.h"
#include "common/args/parse_args.hpp"
#include "common/args/sys_signal.hpp"
#include "common/text/text_painter.h"

#include "extend/yaml/config.h"


#define CONFIG_PATH "./res/app.config.yaml"

void loadModelByConfig(ModelConfig& config, InferWrapper& infer) {

    // Tensor names for input and output
    std::map<std::string, std::string> ts_names = {
        {"input", config.input.name},
        {"output", config.output.name}
    };

    // Define the input and output dims
    nvinfer1::Dims4 input_dims = {config.input.dims[0], config.input.dims[1], config.input.dims[2], config.input.dims[3]};
    nvinfer1::Dims3 output_dims = {config.output.dims[0], config.output.dims[1], config.output.dims[2]};

    // Load the model with the configuration
    infer.update(config.path, ts_names, input_dims, output_dims, config.max_det);
}

int main(int argc, char *argv[]) {

    // Parse command line arguments
    auto args = parse_args_v3(argc, argv);

    //　Register Ctrl+C signal handler
    registerSIGINT();

    // Load the YAML file
    auto yaml_path = args.find("config") != args.end() ? args["config"] : CONFIG_PATH;
    auto mqtt_config = loadMQTTConfig(yaml_path);
    auto model_config = loadModelConfig(yaml_path);

    // Use a shared pointer to manage the life cycle of the InferWrapper object
    InferWrapper infer;
    loadModelByConfig(model_config, infer);

    return 0;
}