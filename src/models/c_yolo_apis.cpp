//
// Created by user on 3/21/25.
//

#include <vector>
#include "serverlet/models/c_yolo_apis.h"
#include "serverlet/models/yolo/infer_yolo_obj.h"
#include "serverlet/models/yolo/infer_yolo_pose.h"

#include "serverlet/utils/logger.h"
#include "serverlet/models/yolo/nms.hpp"

// Instead of void*, use the base class pointer
InferModelBase* vptr_model = nullptr;

// Use this flag to indicate whether the model is yolo-pose
bool b_model_pose = false;

// The results of yolov8
std::vector<Yolo> results_of_objects;

// The results of yolov8 pose
std::vector<YoloPose> results_of_poses;

// The output buffer
float* ptr_buffer = nullptr;



#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief 初始化模型资源
     * @param model_path 模型路径
     * @param b_use_pose 是否使用姿态估计
    */
    void c_yolo_init(const char *model_path, bool b_use_pose) {

        if (b_use_pose) {
            // Avoid shadowing by renaming the local variable
            std::vector local_input_params = {4, 3, 640, 640};
            std::vector local_output_params = {4, 56, 8400};

            // Create a new InferYoloV8Obj object and assign it to vptr_model
            vptr_model = new InferYoloV8Obj(model_path,
                                            "images", local_input_params,
                                            "output0", local_output_params);

            // Display the message
            LOG_INFO("C_API::Initialized", "YoloV8 Pose Detection model loaded successfully");

        } else {
            // Avoid shadowing by renaming the local variable
            std::vector local_input_params = {4, 3, 640, 640};
            std::vector local_output_params = {4, 84, 8400};

            // Create a new InferYoloV8Pose object and assign it to vptr_model
            vptr_model = new InferYoloV8Pose(model_path,
                                             "images", local_input_params,
                                             "output0", local_output_params);

            // Display the message
            LOG_INFO("C_API::Initialized", "YoloV8 Object Detection model loaded successfully");
        }

        // フラグを記録
        b_model_pose = b_use_pose;

        // バッファを初期化
        ptr_buffer = new float[1024];
    }

    /**
     * @brief 释放模型资源
     * @return 是否释放成功
    */
    bool c_yolo_release() {

        // バッファを解放
        if (ptr_buffer != nullptr)
        {
            delete[] ptr_buffer;
            ptr_buffer = nullptr;
        }

        if (vptr_model != nullptr)
        {
            delete vptr_model;  // This now works because InferBase has a virtual destructor
            vptr_model = nullptr;

            LOG_INFO("C_API::Released", "Model released successfully");
            return true;
        } else {
            return false;
        }
    }

    /**
     * @brief 添加图片至模型中
     * @param n_index 索引
     * @param cstr 图片数据指针
     * @param n_channels 通道数
     * @param n_width 宽度
     * @param n_height 高度
     * @return 是否添加成功
    */
    bool c_yolo_add_image(int n_index, unsigned char* cstr, int n_channels, int n_width, int n_height) {
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

    /**
     * @brief 执行推理
     * @return 是否推理成功
    */
    bool c_yolo_inference() {
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

    /**
     * 获取yolo8模型的推理结果
     * @param n_index 索引
     * @param f_clsThreshold 置信度阈值
     * @param f_nmsThreshold nms阈值
     * @return 返回 n_index 索引的可用结果数量
     */
    int c_yolo_available_results(int n_index, float f_clsThreshold, float f_nmsThreshold) {
        // Check if the model is initialized
        if (vptr_model == nullptr)
        {
            // Display the message
            LOG_ERROR("C_API::Error", "Model not initialized");
            return false;
        }

        // Get the count of results available
        if (b_model_pose) {
            // Convert the pointer to InferYoloV8Pose object
            auto* model = dynamic_cast<InferYoloV8Pose*>(vptr_model);

            // Get the results
            auto results = model->postprocess(n_index, f_clsThreshold);

            // Use NMS to filter the results
            results_of_poses = NMS(results, f_nmsThreshold);

            return static_cast<int>(results_of_poses.size());

        } else {
            // Convert the pointer to InferYoloV8Pose object
            auto* model = dynamic_cast<InferYoloV8Obj*>(vptr_model);

            // Get the results
            auto results = model->postprocess(n_index, f_clsThreshold);

            // Use NMS to filter the results
            results_of_objects = NMS(results, f_nmsThreshold);

            return static_cast<int>(results_of_objects.size());
        }
    }

    /**
     * 获取yolo8模型的推理结果
     * @param n_index 索引
     * @param n_itemIdx 结果索引
     * @param n_size 返回的数据大小
     * @return 返回 n_index 索引的第 n_result 个结果
     */
    float* c_yolo_get_result(int n_itemIndex, int& n_size) {
        // Check if the model is initialized
        if (vptr_model == nullptr)
        {
            // Display the message
            LOG_ERROR("C_API::Error", "Model not initialized");
            return nullptr;
        }

        // Get the count of results available
        size_t ln_maxSize = b_model_pose ? results_of_poses.size() : results_of_objects.size();

        // Check if the index is valid
        if (n_itemIndex < 0 || n_itemIndex >= ln_maxSize)
        {
            // Display the message
            LOG_ERROR("C_API::Error", "Invalid item index");
            return nullptr;
        }

        // Get the item
        if (b_model_pose) {
            auto& item = results_of_poses[n_itemIndex];
            // Copy the data to the buffer
            ptr_buffer[0] = static_cast<float>(item.lx);
            ptr_buffer[1] = static_cast<float>(item.ly);
            ptr_buffer[2] = static_cast<float>(item.rx);
            ptr_buffer[3] = static_cast<float>(item.ry);
            ptr_buffer[4] = 0.0f;
            ptr_buffer[5] = item.conf;

            // Copy pts
            for (int i = 0; i < 17; i++) {
                ptr_buffer[6 + i * 3 + 0] = static_cast<float>(item.pts[i].x);
                ptr_buffer[6 + i * 3 + 1] = static_cast<float>(item.pts[i].y);
                ptr_buffer[6 + i * 3 + 2] = static_cast<float>(item.pts[i].conf);
            }

        } else {
            auto& item = results_of_objects[n_itemIndex];
            // Copy the data to the buffer
            // Copy the data to the buffer
            ptr_buffer[0] = static_cast<float>(item.lx);
            ptr_buffer[1] = static_cast<float>(item.ly);
            ptr_buffer[2] = static_cast<float>(item.rx);
            ptr_buffer[3] = static_cast<float>(item.ry);
            ptr_buffer[4] = static_cast<float>(item.cls);
            ptr_buffer[5] = item.conf;
        }

        return ptr_buffer;
    }


#ifdef __cplusplus
};
#endif