//
// Created by orlando on 9/24/24.
//

#include "load_labels.h"
#include <iostream>
#include <fstream>

#include "common/utils/logger.h"


std::vector<std::string> readLabelsFromFile(const std::string& filePath) {
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