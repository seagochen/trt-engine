#ifndef COMBINEDPROJECT_MODEL_INIT_HELPER_HPP
#define COMBINEDPROJECT_MODEL_INIT_HELPER_HPP

#include <memory>       // For std::unique_ptr
#include <string>
#include <map>
#include <functional>   // For std::function
#include <any>          // For std::any (C++17 onwards)
#include <vector>

#include "trtengine/servlet/models/inference/infer_yolo_v8.hpp"
#include "trtengine/servlet/models/inference/infer_efficient_net.hpp"
#include "trtengine/servlet/models/common/yolo_coords.h"
#include "trtengine/servlet/models/infer_model_multi.h"


// 辅助宏用于类型转换，简化参数获取
#define GET_PARAM(map, key, type) std::any_cast<type>(map.at(key))

class ModelFactory {
public:
    // 定义模型创建函数的类型：返回 unique_ptr<InferModelBaseMulti>，接受一个参数map
    using ModelCreator = std::function<std::unique_ptr<InferModelBaseMulti>(const std::string& engine_path, const std::map<std::string, std::any>& params)>;

    // 注册模型创建函数
    static void registerModel(const std::string& name, ModelCreator creator) {
        getCreators()[name] = std::move(creator);
    }

    // 创建模型实例
    static std::unique_ptr<InferModelBaseMulti> createModel(const std::string& name, const std::string& engine_path, const std::map<std::string, std::any>& params) {
        auto& creators = getCreators();
        if (creators.count(name)) {
            return creators[name](engine_path, params);
        }
        // 可以抛出异常或返回nullptr，这里返回nullptr
        return nullptr;
    }

private:
    // 静态成员函数，用于获取和管理所有注册的模型创建器
    static std::map<std::string, ModelCreator>& getCreators() {
        static std::map<std::string, ModelCreator> creators;
        return creators;
    }
};

/**
 * @brief 在程序启动时注册模型
 */
inline void registerModels()
{
    // 注册 YOLOv8 对象检测模型
    ModelFactory::registerModel("YoloV8_Detection", [](const std::string& engine_path, const std::map<std::string, std::any>& params) {

        // 从参数中获取必要的配置
        int maximum_batch = GET_PARAM(params, "maximum_batch", int);
        int maximum_items = GET_PARAM(params, "maximum_items", int);
        int infer_features = GET_PARAM(params, "infer_features", int);
        int infer_samples = GET_PARAM(params, "infer_samples", int);

        std::vector<TensorDefinition> output_tensor_defs = std::vector<TensorDefinition>{{"output0", {maximum_batch, infer_features, infer_samples}}};

        // YOLOv8 对象检测的转换函数
        auto obj_converter = [](const std::vector<float>& input, std::vector<Yolo>& output, int features, int results) {
            cvtXYWHCoordsToYolo(input, output, features, results);
        };

        return std::make_unique<InferYoloV8<Yolo, decltype(obj_converter)>>(
            engine_path, maximum_batch, maximum_items, infer_features, output_tensor_defs, obj_converter
        );
    });

    // 注册 YOLOv8 姿态估计模型
    ModelFactory::registerModel("YoloV8_Pose", [](const std::string& engine_path, const std::map<std::string, std::any>& params) {

        // 从参数中获取必要的配置
        int maximum_batch = GET_PARAM(params, "maximum_batch", int);
        int maximum_items = GET_PARAM(params, "maximum_items", int);
        int infer_features = GET_PARAM(params, "infer_features", int);
        int infer_samples = GET_PARAM(params, "infer_samples", int);

        std::vector<TensorDefinition> output_tensor_defs = std::vector<TensorDefinition>{{"output0", {maximum_batch, infer_features, infer_samples}}};

        // YOLOv8 姿态估计的转换函数
        auto pose_converter = [](const std::vector<float>& input, std::vector<YoloPose>& output, int features, int results) {
            cvtXYWHCoordsToYoloPose(input, output, features, results);
        };

        return std::make_unique<InferYoloV8<YoloPose, decltype(pose_converter)>>(
            engine_path, maximum_batch, maximum_items, infer_features, output_tensor_defs, pose_converter
        );
    });

    // 注册 EfficientNet 模型
    ModelFactory::registerModel("EfficientNet", [](const std::string& engine_path, const std::map<std::string, std::any>& params) {
        int maximum_batch = GET_PARAM(params, "maximum_batch", int);
        return std::make_unique<EfficientFeats>(engine_path, maximum_batch);
    });
};

#endif // COMBINEDPROJECT_MODEL_INIT_HELPER_HPP
