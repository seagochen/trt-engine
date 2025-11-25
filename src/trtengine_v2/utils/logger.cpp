//
// Created by user on 3/21/25.
//

#include <iostream>
#include <sstream>
#include <ctime>

#include "trtengine_v2/utils/logger.h"
#include "trtengine_v2/utils/console_schema.h"

// 静态成员变量定义
std::queue<std::string> Logger::logQueue_;
std::mutex Logger::queueMutex_;
std::condition_variable Logger::queueCondition_;
std::thread Logger::workerThread_;
std::atomic<bool> Logger::running_{false};
std::atomic<bool> Logger::initialized_{false};

// 确保日志系统已初始化
void Logger::ensureInitialized() {
    if (!initialized_.load()) {
        init();
    }
}

// 初始化异步日志系统
void Logger::init() {
    bool expected = false;
    if (initialized_.compare_exchange_strong(expected, true)) {
        running_.store(true);
        workerThread_ = std::thread(workerThread);
    }
}

// 关闭异步日志系统
void Logger::shutdown() {
    if (initialized_.load()) {
        running_.store(false);
        queueCondition_.notify_one();
        if (workerThread_.joinable()) {
            workerThread_.join();
        }
        initialized_.store(false);
    }
}

// 刷新所有待处理的日志
void Logger::flush() {
    if (!initialized_.load()) {
        return;
    }

    std::unique_lock<std::mutex> lock(queueMutex_);
    // 等待队列清空
    queueCondition_.wait(lock, []() {
        return logQueue_.empty();
    });
}

// 后台工作线程
void Logger::workerThread() {
    while (running_.load() || !logQueue_.empty()) {
        std::string logMessage;
        {
            std::unique_lock<std::mutex> lock(queueMutex_);

            // 等待有日志消息或停止信号
            queueCondition_.wait(lock, []() {
                return !logQueue_.empty() || !running_.load();
            });

            if (!logQueue_.empty()) {
                logMessage = std::move(logQueue_.front());
                logQueue_.pop();
            }

            // 如果队列为空，通知可能正在等待 flush 的线程
            if (logQueue_.empty()) {
                queueCondition_.notify_all();
            }
        }

        // 在锁外执行 I/O 操作
        if (!logMessage.empty()) {
            std::cout << logMessage;
        }
    }

    // 确保剩余的日志都被写入
    std::cout.flush();
}

// 处理 module 和 message 的日志输出
void Logger::log(LogLevel level, const std::string& module, const std::string& message) {
    ensureInitialized();
    std::string logMessage = formatLogMessage(level, module, "", message);

    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        logQueue_.push(std::move(logMessage));
    }
    queueCondition_.notify_one();
}

// 处理 module, topic 和 message 的日志输出
void Logger::log(LogLevel level, const std::string& module, const std::string& topic, const std::string& message) {
    ensureInitialized();
    std::string logMessage = formatLogMessage(level, module, topic, message);

    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        logQueue_.push(std::move(logMessage));
    }
    queueCondition_.notify_one();
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

// 自动清理类，确保程序退出时日志被正确刷新
namespace {
    struct LoggerCleanup {
        ~LoggerCleanup() {
            Logger::shutdown();
        }
    };
    static LoggerCleanup loggerCleanup;
}
