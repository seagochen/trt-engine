//
// Created by vipuser on 25-1-22.
//

#ifndef C_VIT_INFER_H
#define C_VIT_INFER_H


#ifdef __cplusplus
extern "C" {
#endif

    struct GoogleVitStruct {
        int index;
        float conf;
    };

    /**
    * @brief 获取human-action-vit模型的推理结果
    * @param n_index 索引
    * @param f_threshold 阈值
    * @param n_topk topk
    * @return 检测到的物体个数
    */
    int c_results_of_google_vit(int n_index, float f_threshold, int n_topk);

    /**
     * @brief 获取human-action-vit模型的推理结果
     * @param n_itemIdx
     * @return
     */
    GoogleVitStruct* c_get_value_of_google_vit(int n_itemIdx);

#ifdef __cplusplus
};
#endif

#endif //C_VIT_INFER_H
