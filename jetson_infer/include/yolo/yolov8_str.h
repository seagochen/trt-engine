//
// Created by ubuntu on 9/10/24.
//

#ifndef JETSON_INFER_YOLOV8_STR_H
#define JETSON_INFER_YOLOV8_STR_H


#include "yolo/yolov8_utils.h"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

// YoloPoint のシリアライズ関数
void to_json(nlohmann::json& j, const YoloPoint& p);

// YoloResult のシリアライズ関数
void to_json(nlohmann::json& j, const YoloResult& r);

// YoloResult のデシリアライズ関数
void from_json(const std::string &str, YoloResult &r);

// YoloPoint のデシリアライズ関数
void from_json(const std::string &str, YoloPoint &p);

// vector<YoloResult> のシリアライズ関数
std::string to_json(const std::vector<YoloResult>& results);

// vector<YoloResult> のデシリアライズ関数
std::vector<YoloResult> from_json(const std::string &str);

#endif //JETSON_INFER_YOLOV8_STR_H
