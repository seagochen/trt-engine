//
// Created by user on 3/21/25.
//

#ifndef COMBINEDPROJECT_INFER_LOGGER_H
#define COMBINEDPROJECT_INFER_LOGGER_H

#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

// 日志级别枚举类型
enum LogLevel {
    VERBOSE,
    INFO,
    WARNING,
    ERROR,
    DEBUG_LV1,
    DEBUG_LV2,
    DEBUG_LV3,
    DEBUG_LV4,
    DEBUG_LV5,
};

// Logger 类声明 - 异步非阻塞日志系统
class Logger {
public:
    // 处理 module 和 message 的日志输出
    static void log(LogLevel level, const std::string& module, const std::string& message);

    // 处理 module, topic 和 message 的日志输出
    static void log(LogLevel level, const std::string& module, const std::string& topic, const std::string& message);

    // 初始化异步日志系统（可选，首次调用 log 时会自动初始化）
    static void init();

    // 关闭异步日志系统，确保所有日志都被写入
    static void shutdown();

    // 刷新所有待处理的日志（阻塞直到队列清空）
    static void flush();

private:
    static std::string formatLogMessage(LogLevel level, const std::string& module, const std::string& topic, const std::string& message);
    static std::string getLogLevelString(LogLevel level);

    // 新增的函数声明，用于获取日志级别对应的颜色
    static std::string getLogLevelColor(LogLevel level);

    static std::string getCurrentTimestamp();

    // 异步日志相关
    static void ensureInitialized();
    static void workerThread();

    static std::queue<std::string> logQueue_;
    static std::mutex queueMutex_;
    static std::condition_variable queueCondition_;
    static std::thread workerThread_;
    static std::atomic<bool> running_;
    static std::atomic<bool> initialized_;
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
#define LOG_DEBUG_V1(module, message) Logger::log(DEBUG_LV1, module, message)
#define LOG_DEBUG_TOPIC(module, topic, message) Logger::log(DEBUG_LV1, module, topic, message)
#define LOG_DEBUG_V2(module, message) Logger::log(DEBUG_LV2, module, message)
#define LOG_DEBUG_V2_TOPIC(module, topic, message) Logger::log(DEBUG_LV2, module, topic, message)
#define LOG_DEBUG_V3(module, message) Logger::log(DEBUG_LV3, module, message)
#define LOG_DEBUG_V3_TOPIC(module, topic, message) Logger::log(DEBUG_LV3, module, topic, message)
#define LOG_DEBUG_V4(module, message) Logger::log(DEBUG_LV4, module, message)
#define LOG_DEBUG_V4_TOPIC(module, topic, message) Logger::log(DEBUG_LV4, module, topic, message)
#define LOG_DEBUG_V5(module, message) Logger::log(DEBUG_LV5, module, message)
#define LOG_DEBUG_V5_TOPIC(module, topic, message) Logger::log(DEBUG_LV5, module, topic, message)

#endif // COMBINEDPROJECT_INFER_LOGGER_H
