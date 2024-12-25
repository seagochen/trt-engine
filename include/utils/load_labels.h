//
// Created by orlando on 9/24/24.
//

#ifndef LOAD_LABELS_H
#define LOAD_LABELS_H


#include <vector>
#include <string>
#include <map>


// Load labels from a txt
std::vector<std::string> loadLabelsFromTxt(const std::string& filePath);

// Load labels from a json
std::map<int, std::string> loadLabelsFromJson(const std::string& filePath);

#endif //LOAD_LABELS_H
