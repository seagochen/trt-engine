import logging
import sys
import atexit
import queue
import threading
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener

# ANSI 转义码定义颜色 (保持不变)
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
GRAY = "\033[90m"
WHITE = "\033[97m"

# 日志级别的颜色映射 (保持不变)
LOG_LEVEL_COLORS = {
    "VERBOSE": CYAN,
    "INFO": GREEN,
    "WARNING": YELLOW,
    "ERROR": RED,
    "DEBUG": GRAY,
    "CRITICAL": WHITE
}

# 自定义日志级别 VERBOSE (保持不变)
VERBOSE_LEVEL = 15
logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")


class Logger:
    """异步非阻塞日志系统"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.logger = logging.getLogger("Logger")
        # 初始级别设为最低的 DEBUG，以便默认情况下所有日志都能被处理
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.hasHandlers():
            # 创建异步日志队列
            self._log_queue = queue.Queue(-1)  # 无限制队列大小

            # 创建实际的输出处理器
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(self.CustomFormatter())

            # 创建队列处理器（非阻塞，只是把日志放入队列）
            queue_handler = QueueHandler(self._log_queue)
            self.logger.addHandler(queue_handler)

            # 创建队列监听器（后台线程处理队列中的日志）
            self._queue_listener = QueueListener(
                self._log_queue,
                stream_handler,
                respect_handler_level=True
            )
            self._queue_listener.start()

            # 注册退出时的清理函数
            atexit.register(self._shutdown)

    def _shutdown(self):
        """关闭异步日志系统，确保所有日志都被写入"""
        if hasattr(self, '_queue_listener') and self._queue_listener is not None:
            self._queue_listener.stop()
            self._queue_listener = None

    def flush(self):
        """刷新所有待处理的日志（阻塞直到队列清空）"""
        if hasattr(self, '_log_queue'):
            self._log_queue.join()

    # --- 新增方法: set_level ---
    def set_level(self, level_name: str):
        """
        动态设置日志记录器的最低级别。
        :param level_name: 日志级别名称 (e.g., "DEBUG", "INFO", "WARNING")
        """
        level_name_upper = level_name.upper()
        level = logging.getLevelName(level_name_upper)

        # getLevelName 对于自定义级别会返回数字，对于标准级别也会返回数字
        # 如果是无效的字符串，会返回一个字符串 "Level " + levelName
        if isinstance(level, int):
            # 特殊处理我们自定义的 VERBOSE 级别
            if level_name_upper == "VERBOSE":
                level = VERBOSE_LEVEL

            self.logger.setLevel(level)
            self.info("Logger", f"日志级别已设置为: {level_name_upper}")
        else:
            self.warning("Logger", f"无效的日志级别: '{level_name}'")

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            log_color = LOG_LEVEL_COLORS.get(record.levelname, RESET)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 支持 module::topic 格式，与 C++ 保持一致
            module_topic = getattr(record, 'module_topic', record.msg.split(' - ', 1)[0] if ' - ' in record.msg else '')
            message = getattr(record, 'log_message', record.msg.split(' - ', 1)[1] if ' - ' in record.msg else record.msg)

            # 如果 module_topic 已经包含在 msg 中（向后兼容），直接使用 msg
            if ' - ' in record.msg and record.msg.startswith('['):
                formatted_message = f"{log_color}[{record.levelname}] <{timestamp}> {record.msg}{RESET}"
            else:
                # 新格式：使用 module_topic 和 message 属性
                formatted_message = f"{log_color}[{record.levelname}] <{timestamp}> {record.msg}{RESET}"

            if record.exc_info:
                traceback_str = self.formatException(record.exc_info)
                formatted_message += f"\n{log_color}{traceback_str}{RESET}"
            return formatted_message

    def log(self, level, module, message, topic=None, exc_info=False):
        """
        记录日志消息
        :param level: 日志级别
        :param module: 模块名
        :param message: 日志消息
        :param topic: 可选的主题/子模块名，用于更细粒度的分类
        :param exc_info: 是否包含异常堆栈信息
        """
        # 构建模块::主题格式，与 C++ 保持一致
        if topic:
            module_topic = f"{module}::{topic}"
        else:
            module_topic = module

        log_message = f"[{module_topic}] - {message}"
        level_upper = level.upper()

        # 使用 getattr 动态调用，简化代码
        if level_upper == "VERBOSE":
            self.logger.log(VERBOSE_LEVEL, log_message, exc_info=exc_info)
        else:
            # 直接使用标准日志级别进行调用
            # self.logger.log 会处理级别过滤
            self.logger.log(logging.getLevelName(level_upper), log_message, exc_info=exc_info)

    def verbose(self, module, message, topic=None):
        self.log("VERBOSE", module, message, topic=topic)

    def info(self, module, message, topic=None):
        self.log("INFO", module, message, topic=topic)

    def warning(self, module, message, topic=None):
        self.log("WARNING", module, message, topic=topic)

    def error_trace(self, module, message, topic=None):
        self.log("ERROR", module, message, topic=topic, exc_info=True)

    def error(self, module, message, topic=None):
        self.log("ERROR", module, message, topic=topic, exc_info=False)

    def debug(self, module, message, topic=None):
        self.log("DEBUG", module, message, topic=topic)

    def critical(self, module, message, topic=None):
        self.log("CRITICAL", module, message, topic=topic)


# 定义一个 logger 实例
logger = Logger()

# 示例用法
if __name__ == "__main__":
    logger.info("Main", "开始演示 Logger 功能... (默认级别: DEBUG)")
    logger.debug("Main", "这是一条默认情况下会显示的 DEBUG 信息。")
    print("-" * 50)

    # 场景1: 捕获异常并打印调用栈
    logger.info("Main", "场景1: 捕获异常并打印调用栈")
    try:
        1 / 0
    except Exception as e:
        logger.error_trace("ModuleA", f"计算时发生严重错误: {e}")

    print("-" * 50)

    # 场景2: 记录一个无需调用栈的错误
    logger.info("Main", "场景2: 记录一个无需调用栈的错误")
    logger.error("ModuleC", "用户输入格式不正确。")

    print("-" * 50)

    # --- 新增测试场景 3: 动态设置日志级别 ---
    logger.info("Main", "场景3: 动态设置日志级别")

    # 1. 设置级别为 INFO
    logger.set_level("INFO")

    # 2. 尝试打印 DEBUG 和 INFO 级别的日志
    logger.debug("Main", "这条 DEBUG 信息现在应该不会显示。")
    logger.info("Main", "这条 INFO 信息应该会正常显示。")
    logger.warning("Main", "WARNING 及以上级别的信息也会显示。")

    print("-" * 50)

    # 3. 将级别改回 DEBUG，以进行其他调试
    logger.info("Main", "将日志级别恢复为 DEBUG")
    logger.set_level("DEBUG")
    logger.debug("Main", "这条 DEBUG 信息现在又可以显示了。")

    # 确保所有日志都被写入
    logger.flush()
