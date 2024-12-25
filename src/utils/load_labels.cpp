//
// Created by orlando on 9/24/24.
//

#include <iostream>
#include <fstream>

#include "common/nlohmann/json.hpp"
#include "common/utils/logger.h"
#include "common/utils/load_labels.h"

// 使用 nlohmann::json 命名空间
using json = nlohmann::json;


std::map<int, std::string> loadLabelsFromJson(const std::string& filePath) {
    // 定义结果字典
    std::map<int, std::string> id2labelMap;

    // 打开文件
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filePath);
    }

    // 读取并解析 JSON 文件
    json jsonData;
    file >> jsonData;

    // 获取 id2label 字段
    if (jsonData.contains("id2label")) {
        auto id2label = jsonData["id2label"];
        for (auto it = id2label.begin(); it != id2label.end(); ++it) {
            int key = std::stoi(it.key());   // 将 key 转换为 int
            std::string value = it.value(); // 获取 value
            id2labelMap[key] = value;       // 插入到 map 中
        }
    } else {
        throw std::runtime_error("JSON 数据中缺少 'id2label' 字段");
    }

    return id2labelMap;
}


std::vector<std::string> loadLabelsFromTxt(const std::string& filePath) {
    std::vector<std::string> labels;
    std::ifstream file(filePath);

    if (!file.is_open()) {
        LOG_ERROR("LoadLabels", "Failed to open file: " + filePath);
        exit(EXIT_FAILURE);
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines
        if (!line.empty()) {
            // Convert CRLF to LF
            if (line.back() == '\r') {
                line.pop_back();
            }

            // Add the label to the vector
            labels.push_back(line);
        }
    }

    file.close();
    return labels;
}