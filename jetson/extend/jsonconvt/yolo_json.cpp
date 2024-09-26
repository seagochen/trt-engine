//
// Created by orlando on 9/26/24.
//

#include "yolo_json.h"
#include "nlohmann/json.hpp"
#include "common/utils/logger.h"


// Helper function to convert YoloPoint object to json
void to_json(nlohmann::json &j, const YoloPoint &p) {
    j = nlohmann::json{
        {"x", p.x},
        {"y", p.y},
        {"conf", p.conf}
    };
}

// Helper function to convert json to YoloPoint object
void from_json(const nlohmann::json &j, YoloPoint &p) {
    j.at("x").get_to(p.x);
    j.at("y").get_to(p.y);
    j.at("conf").get_to(p.conf);
}

// Helper function to convert YoloPose object to json
void to_json(nlohmann::json &j, const YoloPose &pose) {
    j = nlohmann::json{
        {"lx", pose.lx},
        {"ly", pose.ly},
        {"rx", pose.rx},
        {"ry", pose.ry},
        {"conf", pose.conf},
        {"pts", pose.pts}  // Vector of YoloPoint objects
    };
}

// Helper function to convert json to YoloPose object
void from_json(const nlohmann::json &j, YoloPose &pose) {
    j.at("lx").get_to(pose.lx);
    j.at("ly").get_to(pose.ly);
    j.at("rx").get_to(pose.rx);
    j.at("ry").get_to(pose.ry);
    j.at("conf").get_to(pose.conf);
    j.at("pts").get_to(pose.pts);  // Vector of YoloPoint objects
}

// Helper function to convert Yolo object to json
void to_json(nlohmann::json &j, const Yolo &y) {
    j = nlohmann::json{
        {"lx", y.lx},
        {"ly", y.ly},
        {"rx", y.rx},
        {"ry", y.ry},
        {"cls", y.cls},
        {"conf", y.conf}
    };
}

// Helper function to convert json to Yolo object
void from_json(const nlohmann::json &j, Yolo &y) {
    j.at("lx").get_to(y.lx);
    j.at("ly").get_to(y.ly);
    j.at("rx").get_to(y.rx);
    j.at("ry").get_to(y.ry);
    j.at("cls").get_to(y.cls);
    j.at("conf").get_to(y.conf);
}


std::vector<Yolo> json2yolo(const std::string &json_str) {
    std::vector<Yolo> yolos;
    try {
        nlohmann::json json_data = nlohmann::json::parse(json_str);
        yolos = json_data.get<std::vector<Yolo>>();
    } catch (const nlohmann::json::exception &e) {
        // Handle parsing errors if needed
        LOG_ERROR("json2yolo", "Error deserializing JSON: " + std::string(e.what()));
    }
    return yolos;
}

std::string yolo2json(const std::vector<Yolo> &yolos) {
    try {
        nlohmann::json json_data = yolos;
        return json_data.dump();
    } catch (const nlohmann::json::exception &e) {
        // Handle serialization errors if needed
        LOG_ERROR("yolo2json", "Error serializing JSON: " + std::string(e.what()));
        return "";
    }
}

std::vector<YoloPose> json2yoloPose(const std::string &json_str) {
    std::vector<YoloPose> yoloPoses;
    try {
        nlohmann::json json_data = nlohmann::json::parse(json_str);
        yoloPoses = json_data.get<std::vector<YoloPose>>();
    } catch (const nlohmann::json::exception &e) {
        LOG_ERROR("json2yoloPose", "Error deserializing JSON: " + std::string(e.what()));
    }
    return yoloPoses;
}

std::string yoloPose2json(const std::vector<YoloPose> &yoloPoses) {
    try {
        nlohmann::json json_data = yoloPoses;
        return json_data.dump();
    } catch (const nlohmann::json::exception &e) {
        LOG_ERROR("yoloPose2json", "Error serializing JSON: " + std::string(e.what()));
        return "";
    }
}
