// //
// // Created by orlando on 12/19/24.
// //
//
// #include "infer_model_base.h"
// #include "common/utils/logger.h"
//
//
// // Custom deleter for ICudaEngine
// auto engineDeleter = [](nvinfer1::ICudaEngine* engine) {
//     if (engine) {
//         engine->destroy();
//     }
// };
//
//
// // Custom deleter for IExecutionContext
// auto contextDeleter = [](nvinfer1::IExecutionContext* context) {
//     if (context) {
//         context->destroy();
//     }
// };
//
//
// // 将 std::vector<int> 转换为 nvinfer1::Dims4
// nvinfer1::Dims4 toDims4(const std::vector<int>& dims) {
//     if (dims.size() != 4) {
//         throw std::runtime_error("Invalid dimensions for Dims4. Expected size 4.");
//     }
//     return nvinfer1::Dims4{dims[0], dims[1], dims[2], dims[3]};
// }
//
//
// InferModelBase::InferModelBase(int batch_size): engine(nullptr, engineDeleter), context(nullptr, contextDeleter),
//                                                  max_batch_size(batch_size) {}
//
// InferModelBase::~InferModelBase() {
//   trt_buffers.clear();
//   context.reset();
//   engine.reset();
// }
//
// void InferModelBase::loadEngine(
//     const std::string& engine_path,
//     const std::map<std::string, std::string>& names,
//     const std::vector<int>& input_dims,
//     const std::vector<int>& output_dims) {
//
//     // Load the TensorRT engine from the serialized engine file
//     engine = loadEngineFromFile(engine_path);
//     if (!engine) {
//         // throw std::runtime_error("Failed to load engine from file.");
//         LOG_ERROR_TOPIC("InferWrapper", "Engine", "Failed to load engine from file.");
//         exit(EXIT_FAILURE);
//     } else {
//         LOG_VERBOSE_TOPIC("InferWrapper", "Engine", "Engine loaded successfully.");
//     }
//
//     // Convert input_dims to nvinfer1::Dims4
//     nvinfer1::Dims4 dims4_input = toDims4(input_dims);
//
//     // Create a context for executing the engine
//     context = createExecutionContext(engine, names.at("input"), dims4_input);
//     LOG_VERBOSE_TOPIC("InferWrapper", "Engine", "Context created successfully.");
//
//     tensor_names = names;
//     this->input_dims = input_dims;
//     this->output_dims = output_dims;
//
//     // Allocate buffers for TensorRT
//     allocateBuffers();
// }
//
//
// void InferModelBase::allocateBuffers() {
//
//     // Allocate TensorRT buffers
//     std::map<std::string, std::vector<int>> trt_binding_dims;
//     trt_binding_dims[tensor_names["input"]] = input_dims;
//     trt_binding_dims[tensor_names["output"]] = output_dims;
//     trt_buffers = allocateCudaTensors(trt_binding_dims);
// }
//
//
// void InferModelBase::copyToDevice(const cv::Mat& image, const std::string& input_tensor_name) {
//     // Ensure input image dimensions match the model's expected input dimensions
//     if (image.cols != input_dims[3] || image.rows != input_dims[2]) {
//         throw std::runtime_error("Input image dimensions do not match model input dimensions.");
//     }
//
//     // Convert image to float and copy to GPU
//     Tensor<float> input_tensor = trt_buffers[input_tensor_name];
//     cudaMemcpy(input_tensor.ptr(), image.data, image.total() * sizeof(float), cudaMemcpyHostToDevice);
// }
