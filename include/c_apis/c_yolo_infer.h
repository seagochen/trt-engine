//
// Created by vipuser on 25-1-22.
//

#ifndef C_YOLO_H
#define C_YOLO_H


#ifdef __cplusplus
extern "C" {
#endif

    struct YoloStruct {
        int lx, ly, rx, ry;
        float conf;
        int cls;
    };

    struct YoloPointStruct {
        int x, y;
        float conf;
    };

    /**
    * @brief 获取yolov8模型的推理结果
    * @param n_index 索引
    * @param f_clsThreshold 类别阈值
    * @param f_nmsThreshold nms阈值
    * @return 检测到的物体个数
    */
    int c_results_of_yolov8_obj(int n_index, float f_clsThreshold, float f_nmsThreshold);

    /**
     * @brief 获取yolov8模型的推理结果
     * @param n_itemIdx
     * @return
     */
    YoloStruct* c_get_value_of_yolov8_obj(int n_itemIdx);

    /**
    * @brief 获取yolov8-pose模型的推理结果
    * @param n_index 索引
    * @param f_clsThreshold 类别阈值
    * @param f_nmsThreshold nms阈值
    * @return 检测到的物体个数
    */
    int c_results_of_yolov8_pose(int n_index, float f_clsThreshold, float f_nmsThreshold);

    /**
     * @brief 获取yolov8-pose模型的推理结果
     * @param n_itemIdx
     * @return
     */
    YoloStruct* c_get_value_of_yolov8_pose(int n_itemIdx);

    /**
     * @brief 获取yolov8-pose模型的推理结果
     * @param n_itemIdx
     * @param n_keypointIdx
     * @return
     */
    YoloPointStruct* c_get_value_of_yolov8_pose_keypoint(int n_itemIdx, int n_keypointIdx);


#ifdef __cplusplus
};
#endif

#endif //C_YOLO_H
