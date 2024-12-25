#include "include/c_inference.h"
#include "include/c_vit_infer.h"
#include "include/c_yolo_infer.h"

#include "common/models/infer_yolov8_obj.h"
#include "common/models/infer_yolov8_pose.h"
#include "common/models/infer_google_vit.h"

#include "common/utils/logger.h"
#include "common/yolo/nms.hpp"

#include <opencv2/opencv.hpp>

// The ptr of model
void* vptr_model = nullptr;

// The name of model
std::string str_modelName;

// The results of yolov8
std::vector<Yolo> results_of_yolov8; 

// The results of yolov8 pose
std::vector<YoloPose> results_of_posev8; 

// The struct of Yolo
YoloStruct struct_yolo;

// The struct of YoloPoint
YoloPointStruct struct_pose;

// The struct of GoogleVitStruct
GoogleVitStruct struct_vit;

// The results of int
std::vector<std::tuple<int, float>> results_of_vit;


#define DEBUG 0


extern "C" {

    // bool c_model_init(const char* cstr_modelPath, const char* cstr_modelType)
    // {

    //     // Convert char* to string
    //     std::string str_modelPath(cstr_modelPath);
    //     std::string str_modelType(cstr_modelType);

    //     // Create a new model object based on the model type
    //     if (str_modelType == "yolov8n") {
    //         std::vector input_params = {4, 3, 640, 640};
    //         std::vector output_params = {4, 84, 8400};

    //         vptr_model = c_yolo_init(cstr_modelPath,
    //             "images", 4, input_params.data(),
    //             "output0", 3, output_params.data());
    //     }
    //     else if (str_modelType == "yolov8n-pose") {
    //         std::vector input_params = {4, 3, 640, 640};
    //         std::vector output_params = {4, 56, 8400};

    //         vptr_model = c_pose_init(cstr_modelPath,
    //             "images", 4, input_params.data(),
    //             "output0", 3, output_params.data());
    //     }
    //     else if (str_modelType == "efficient-vit") {
    //         std::vector input_params = {4, 3, 224, 224};
    //         std::vector output_params = {4, 4};

    //         vptr_model = c_vit_init(cstr_modelPath,
    //             "input", 4, input_params.data(),
    //             "output", 2, output_params.data());
    //     }
    //     else
    //     {
    //         // Display the message
    //         LOG_ERROR("C_API::Error", "Model type not supported");
    //         return false;
    //     }

    //     // Set the model name
    //     str_modelName = str_modelType;

    //     return true;
    // }


    void* c_yolo_init(const char *model_path) {

        // Avoid shadowing by renaming the local variable
        // std::vector<int> local_input_params = {input_params[0], input_params[1], input_params[2], input_params[3]};
        // std::vector<int> local_output_params = {output_params[0], output_params[1], output_params[2]};

        // Avoid shadowing by renaming the local variable
        std::vector local_input_params = {4, 3, 640, 640};
        std::vector local_output_params = {4, 84, 8400};

        // Create a new InferYoloV8Obj object and assign it to vptr_model
        vptr_model = new InferYoloV8Obj(model_path,
            "images", local_input_params,
            "output0", local_output_params);

        str_modelName = "yolov8n";

        // Display the message
        LOG_INFO("C_API::Initialized", "YoloV8 Object Detection model loaded successfully");

        return vptr_model;
    }


    void* c_pose_init(const char *model_path) {

        // Avoid shadowing by renaming the local variable
        // std::vector<int> local_input_params = {input_params[0], input_params[1], input_params[2], input_params[3]};
        // std::vector<int> local_output_params = {output_params[0], output_params[1], output_params[2]};
        
        // Avoid shadowing by renaming the local variable
        std::vector local_input_params = {4, 3, 640, 640};
        std::vector local_output_params = {4, 56, 8400};

        // Create a new InferYoloV8Pose object and assign it to vptr_model
        vptr_model = new InferYoloV8Pose(model_path,
            "images", local_input_params,
            "output0", local_output_params);

        str_modelName = "yolov8n-pose";

        // Display the message
        LOG_INFO("C_API::Initialized", "YoloV8 Pose Estimation model loaded successfully");

        return vptr_model;
    }


    void* c_vit_init(const char *model_path) {

        // Avoid shadowing by renaming the local variable
        // std::vector<int> local_input_params = {input_params[0], input_params[1], input_params[2], input_params[3]};
        // std::vector<int> local_output_params = {output_params[0], output_params[1]};

        // Avoid shadowing by renaming the local variable
        std::vector local_input_params = {4, 3, 224, 224};
        std::vector local_output_params = {4, 4};

        // Create a new InferHumanActionVit object and assign it to vptr_model
        vptr_model = new InferGoogleVit(model_path,
            "input", local_input_params,
            "output", local_output_params);

        str_modelName = "efficient-vit";

        // Display the message
        LOG_INFO("C_API::Initialized", "Google VIT model loaded successfully");

        return vptr_model;
    }


    bool c_release_model()
    {
        if (vptr_model != nullptr)
        {
            if (str_modelName == "yolov8n") {
                delete static_cast<InferYoloV8Obj*>(vptr_model);
            }
            else if (str_modelName == "yolov8n-pose") {
                delete static_cast<InferYoloV8Pose*>(vptr_model);
            }
            else if (str_modelName == "efficient-vit") {
                delete static_cast<InferGoogleVit*>(vptr_model);
            }
            else {
                throw std::runtime_error("Model type not supported");
                return false;
            }

            vptr_model = nullptr;

            // Display the message
            LOG_INFO("C_API::Released", "Model released successfully");
            return true;
        }

        // Display the message
        LOG_ERROR("C_API::Error", "Model not initialized");
        return false;
    }


    bool c_add_image(int n_index, unsigned char* cstr, int n_channels, int n_width, int n_height)
    {
        // Check if the model is initialized
        if (vptr_model == nullptr)
        {
            // Display the message
            LOG_ERROR("C_API::Error", "Model not initialized");
            return false;
        }

        // Convert the unsigned char* to cv::Mat
        cv::Mat img;
        if (n_channels == 1)
        {
            img = cv::Mat(n_height, n_width, CV_8UC1, cstr);
        }
        else if (n_channels == 3)
        {
            img = cv::Mat(n_height, n_width, CV_8UC3, cstr);
        }
        else
        {
            // Display the message
            LOG_ERROR("C_API::Error", "Number of channels not supported");
            return false;
        }

        // Add the image to the model
        static_cast<InferModelBase*>(vptr_model)->preprocess(img, n_index);

        return true;
    }


    bool c_do_inference()
    {
        // Check if the model is initialized
        if (vptr_model == nullptr)
        {
            // Display the message
            LOG_ERROR("C_API::Error", "Model not initialized");
            return false;
        }

        // Perform inference
        return static_cast<InferModelBase*>(vptr_model)->inference();
    }


    int c_results_of_google_vit(int n_index, float f_threshold, int n_topk)
    {
        // Check if the model is initialized
        if (vptr_model == nullptr)
        {
            // Display the message
            LOG_ERROR("C_API::Error", "Model not initialized");
            return false;
        }

        // Convert the pointer to InferHumanActionVit object
        auto* model = static_cast<InferGoogleVit*>(vptr_model);

        // Get the results
        results_of_vit = model->postprocess(n_index, f_threshold, n_topk);

#if DEBUG
        // Print out every result of vit
        for (auto i = 0; i < results_of_vit.size(); i++)
        {
            auto item = results_of_vit[i];
            std::cout << "Index: " << std::get<0>(item) << ", Confidence: " << std::get<1>(item) << std::endl;
        }
#endif

        return static_cast<int>(results_of_vit.size());
    }


    GoogleVitStruct* c_get_value_of_google_vit(int n_itemIdx)
    {
        // Check if the model is initialized
        if (vptr_model == nullptr)
        {
            // Display the message
            LOG_ERROR("C_API::Error", "Model not initialized");
            return nullptr;
        }

        // Check if the index is valid
        if (n_itemIdx < 0 || n_itemIdx >= results_of_vit.size())
        {
            // Display the message
            LOG_ERROR("C_API::Error", "Invalid item index");
            return nullptr;
        }

        struct_vit.index = std::get<0>(results_of_vit[n_itemIdx]);
        struct_vit.conf = std::get<1>(results_of_vit[n_itemIdx]);

        return &struct_vit;
    }


    int c_results_of_yolov8_obj(int n_index, float f_clsThreshold, float f_nmsThreshold)
    {
        // Check if the model is initialized
        if (vptr_model == nullptr)
        {
            // Display the message
            LOG_ERROR("C_API::Error", "Model not initialized");
            return false;
        }

        // Convert the pointer to InferYoloV8Obj object
        auto* model = static_cast<InferYoloV8Obj*>(vptr_model);

        // Get the results
        auto results = model->postprocess(n_index, f_clsThreshold);

        // Use NMS to filter the results
        results_of_yolov8 = NMS(results, f_nmsThreshold);

        return static_cast<int>(results_of_yolov8.size());
    }


    YoloStruct* c_get_value_of_yolov8_obj(int n_itemIdx)
    {
        // Check if the model is initialized
        if (vptr_model == nullptr)
        {
            // Display the message
            LOG_ERROR("C_API::Error", "Model not initialized");
            return nullptr;
        }

        // Check if the index is valid
        if (n_itemIdx < 0 || n_itemIdx >= results_of_yolov8.size())
        {
            // Display the message
            LOG_ERROR("C_API::Error", "Invalid item index");
            return nullptr;
        }

        // Get the item
        auto item = results_of_yolov8[n_itemIdx];

        // Set the values
        struct_yolo.lx = item.lx;
        struct_yolo.ly = item.ly;
        struct_yolo.rx = item.rx;
        struct_yolo.ry = item.ry;
        struct_yolo.cls = item.cls;
        struct_yolo.conf = item.conf;

        return &struct_yolo;
    }


    int c_results_of_yolov8_pose(int n_index, float f_clsThreshold, float f_nmsThreshold)
    {
        // Check if the model is initialized
        if (vptr_model == nullptr)
        {
            // Display the message
            LOG_ERROR("C_API::Error", "Model not initialized");
            return false;
        }

        // Convert the pointer to InferYoloV8Pose object
        auto* model = static_cast<InferYoloV8Pose*>(vptr_model);

        // Get the results
        auto results = model->postprocess(n_index, f_clsThreshold);

        // Use NMS to filter the results
        results_of_posev8 = NMS(results, f_nmsThreshold);
     
        return static_cast<int>(results_of_posev8.size());
    }


    YoloStruct* c_get_value_of_yolov8_pose(int n_itemIdx)
    {
        // Check if the model is initialized
        if (vptr_model == nullptr)
        {
            // Display the message
            LOG_ERROR("C_API::Error", "Model not initialized");
            return nullptr;
        }

        // Check if the index is valid
        if (n_itemIdx < 0 || n_itemIdx >= results_of_posev8.size())
        {
            // Display the message
            LOG_ERROR("C_API::Error", "Invalid item index");
            return nullptr;
        }

        // Get the item
        const auto& item = results_of_posev8[n_itemIdx];

        // Set the values
        struct_yolo.lx = item.lx;
        struct_yolo.ly = item.ly;
        struct_yolo.rx = item.rx;
        struct_yolo.ry = item.ry;
        struct_yolo.cls = 0;
        struct_yolo.conf = item.conf;

        return &struct_yolo;
    }


    YoloPointStruct* c_get_value_of_yolov8_pose_keypoint(int n_itemIdx, int n_keypointIdx) 
    {
        // Check if the model is initialized
        if (vptr_model == nullptr)
        {
            // Display the message
            LOG_ERROR("C_API::Error", "Model not initialized");
            return nullptr;
        }

        // Check if the index is valid
        if (n_itemIdx < 0 || n_itemIdx >= results_of_posev8.size())
        {
            // Display the message
            LOG_ERROR("C_API::Error", "Invalid item index");
            return nullptr;
        }

        // Check if the keypoint index is valid
        if (n_keypointIdx < 0 || n_keypointIdx >= 17)
        {
            // Display the message
            LOG_ERROR("C_API::Error", "Invalid keypoint index");
            return nullptr;
        }

        // Get the item
        const auto& item = results_of_posev8[n_itemIdx];

        // Get the keypoint
        auto keypoint = item.pts[n_keypointIdx];

        // Set the values
        struct_pose.x = keypoint.x;
        struct_pose.y = keypoint.y;
        struct_pose.conf = keypoint.conf;

        return &struct_pose;
    }
};
