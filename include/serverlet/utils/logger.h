//
// Created by user on 3/21/25.
//

#ifndef COMBINEDPROJECT_LOGGER_H
#define COMBINEDPROJECT_LOGGER_H

#include <string>

// 日志级别枚举类型
enum LogLevel {
    VERBOSE,
    INFO,
    WARNING,
    ERROR,
    DEBUG
};

// Logger 类声明
class Logger {
public:
    // 处理 module 和 message 的日志输出
    static void log(LogLevel level, const std::string& module, const std::string& message);

    // 处理 module, topic 和 message 的日志输出
    static void log(LogLevel level, const std::string& module, const std::string& topic, const std::string& message);

private:
    static std::string formatLogMessage(LogLevel level, const std::string& module, const std::string& topic, const std::string& message);
    static std::string getLogLevelString(LogLevel level);

    // 新增的函数声明，用于获取日志级别对应的颜色
    static std::string getLogLevelColor(LogLevel level);

    static std::string getCurrentTimestamp();
};

// 宏用于简化日志输出
#define LOG_VERBOSE(module, message) Logger::log(VERBOSE, module, message)
#define LOG_VERBOSE_TOPIC(module, topic, message) Logger::log(VERBOSE, module, topic, message)
#define LOG_INFO(module, message) Logger::log(INFO, module, message)
#define LOG_INFO_TOPIC(module, topic, message) Logger::log(INFO, module, topic, message)
#define LOG_WARNING(module, message) Logger::log(WARNING, module, message)
#define LOG_WARNING_TOPIC(module, topic, message) Logger::log(WARNING, module, topic, message)
#define LOG_ERROR(module, message) Logger::log(ERROR, module, message)
#define LOG_ERROR_TOPIC(module, topic, message) Logger::log(ERROR, module, topic, message)
#define LOG_DEBUG(module, message) Logger::log(DEBUG, module, message)
#define LOG_DEBUG_TOPIC(module, topic, message) Logger::log(DEBUG, module, topic, message)

#endif //COMBINEDPROJECT_LOGGER_H
