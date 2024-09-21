//
// Created by orlando on 9/24/24.
//

#include "load_labels.h"
#include <iostream>
#include <fstream>

std::vector<std::string> readLabelsFromFile(const std::string& filePath) {
    std::vector<std::string> labels;
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return labels;  // Return an empty vector in case of error
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines
        if (!line.empty()) {
            labels.push_back(line);
        }
    }

    file.close();
    return labels;
}