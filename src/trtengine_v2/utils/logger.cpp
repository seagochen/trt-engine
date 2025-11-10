//
// Created by user on 3/21/25.
//

#include <iostream>
#include <sstream>
#include <ctime>

#include "trtengine_v2/utils/logger.h"
#include "trtengine_v2/utils/console_schema.h"


// 处理 module 和 message 的日志输出
void Logger::log(LogLevel level, const std::string& module, const std::string& message) {
    std::string logMessage = formatLogMessage(level, module, "", message);
    std::cout << logMessage;
}

// 处理 module, topic 和 message 的日志输出
void Logger::log(LogLevel level, const std::string& module, const std::string& topic, const std::string& message) {
    std::string logMessage = formatLogMessage(level, module, topic, message);
    std::cout << logMessage;
}

// 格式化日志信息
std::string Logger::formatLogMessage(LogLevel level, const std::string& module, const std::string& topic, const std::string& message) {
    std::string logLevelStr = getLogLevelString(level);
    std::string timestamp = getCurrentTimestamp();

    // 选择颜色
    std::string color = getLogLevelColor(level);

    std::ostringstream logStream;
    logStream << color;                         // 设置颜色
    logStream << "[" << logLevelStr << "] ";    // 输出日志级别
    logStream << "<" << timestamp << "> ";      // 输出时间戳
    
    logStream << "[" << module;                 // 输出模块名
    if (!topic.empty()) {
        logStream << "::" << topic;
    }
    logStream << "] - " << message;             // 输出消息内容

    // 重置颜色
    logStream << CONSOLE_COLOR_RESET << std::endl;

    // 返回格式化后的日志字符串
    return logStream.str();
}

// 获取日志级别的字符串表示
std::string Logger::getLogLevelString(LogLevel level) {
    switch (level) {
        case VERBOSE: return "VERBOSE";
        case INFO: return "INFO";
        case WARNING: return "WARNING";
        case ERROR: return "ERROR";
        case DEBUG_LV1: return "LEVEL1";
        case DEBUG_LV2: return "LEVEL2";
        case DEBUG_LV3: return "LEVEL3";
        case DEBUG_LV4: return "LEVEL4";
        case DEBUG_LV5: return "LEVEL5";
        default: return "UNKNOWN";
    }
}

// 根据日志级别选择颜色
std::string Logger::getLogLevelColor(LogLevel level) {
    switch (level) {
        case VERBOSE: return CONSOLE_COLOR_CYAN;      // 青色
        case INFO: return CONSOLE_COLOR_GREEN;        // 绿色
        case WARNING: return CONSOLE_COLOR_YELLOW;    // 黄色
        case ERROR: return CONSOLE_COLOR_RED;         // 红色
        case DEBUG_LV1: return CONSOLE_COLOR_GRAY;          // 灰色
        case DEBUG_LV2: return CONSOLE_COLOR_LIGHT_GRAY;    // 浅灰色
        case DEBUG_LV3: return CONSOLE_COLOR_WHITE;         // 白色
        case DEBUG_LV4: return CONSOLE_COLOR_LIGHT_BLUE;    // 浅蓝色
        case DEBUG_LV5: return CONSOLE_COLOR_LIGHT_PURPLE;  // 浅紫色
        // 如果需要更多颜色，可以继续添加
        default: return CONSOLE_COLOR_RESET;
    }
}

// 获取当前时间戳
std::string Logger::getCurrentTimestamp() {
    std::time_t now = std::time(nullptr);
    char buf[100];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return buf;
}