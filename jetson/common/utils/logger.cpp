#include "logger.h"
#include <iostream>
#include <sstream>
#include <ctime>

// ANSI 转义码定义颜色
#define RESET "\033[0m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define CYAN "\033[36m"
#define BLACK "\033[30m"

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
    logStream << color << "[" << module;

    if (!topic.empty()) {
        logStream << "/" << topic;
    }

    logStream << "] " << logLevelStr << " " << timestamp << ": " << message << RESET << std::endl;  // 输出后重置颜色
    return logStream.str();
}

// 获取日志级别的字符串表示
std::string Logger::getLogLevelString(LogLevel level) {
    switch (level) {
        case VERBOSE: return "VERBOSE";
        case INFO: return "INFO";
        case WARNING: return "WARNING";
        case ERROR: return "ERROR";
        case DEBUG: return "DEBUG";
        default: return "UNKNOWN";
    }
}

// 根据日志级别选择颜色
std::string Logger::getLogLevelColor(LogLevel level) {
    switch (level) {
        case VERBOSE: return CYAN;      // 青色
        case INFO: return GREEN;        // 绿色
        case WARNING: return YELLOW;    // 黄色
        case ERROR: return RED;         // 红色
        case DEBUG: return BLACK;       // 黑色
        default: return RESET;
    }
}

// 获取当前时间戳
std::string Logger::getCurrentTimestamp() {
    std::time_t now = std::time(nullptr);
    char buf[100];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return buf;
}
