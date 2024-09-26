//
// Created by orlando on 9/26/24.
//

#ifndef YOLO_JSON_H
#define YOLO_JSON_H

#include "common/yolo/yolo_def.h"

#include <vector>
#include <string>

/**
 * Convert a json string to a vector of Yolo objects.
 * @param json The json string to convert.
 * @return A vector of Yolo objects.
 */
std::vector<Yolo> json2yolo(const std::string &json);

/**
 * Convert a json string to a vector of YoloPose objects.
 * @param json The json string to convert.
 * @return A vector of YoloPose objects.
 */
std::vector<YoloPose> json2yoloPose(const std::string &json);

/**
 * Convert a vector of Yolo objects to a json string.
 * @param yolos The vector of Yolo objects to convert.
 * @return A json string.
 */
std::string yolo2json(const std::vector<Yolo> &yolos);

/**
 * Convert a vector of YoloPose objects to a json string.
 * @param yolos The vector of YoloPose objects to convert.
 * @return A json string.
 */
std::string yoloPose2json(const std::vector<YoloPose> &yolos);

#endif //YOLO_JSON_H