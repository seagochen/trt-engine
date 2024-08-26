//
// Created by ubuntu on 9/10/24.
//

#include "yolo/yolov8_str.h"

void to_json(nlohmann::json& j, const YoloPoint& p) {
    j = nlohmann::json{{"x", p.x}, {"y", p.y}, {"conf", p.conf}};
}

// YoloResult のシリアライズ関数
void to_json(nlohmann::json& j, const YoloResult& r) {
    j = nlohmann::json{
            {"lx", r.lx}, {"ly", r.ly}, {"rx", r.rx}, {"ry", r.ry},
            {"cls", r.cls}, {"conf", r.conf}, {"keypoints", r.keypoints}
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
    nlohmann::json j = results;
    return j.dump();
}

// vector<YoloResult> のデシリアライズ関数
std::vector<YoloResult> from_json(const std::string &str) {
    nlohmann::json j = nlohmann::json::parse(str);
    std::vector<YoloResult> results = j.get<std::vector<YoloResult>>();
    return results;
}
