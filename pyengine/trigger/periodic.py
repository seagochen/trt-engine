import time
import threading
from typing import Callable
from pyengine.utils.logger import logger


class PeriodicTask:

    def __init__(self, interval_ms: int, task: Callable, *args, **kwargs):
        """
        初始化定期任务。

        :param interval_ms: 定期任务执行的时间间隔（单位为毫秒）。
        :param task: 定期执行的任务，传入一个回调函数。
        :param *args: 传递给任务的可变位置参数。
        :param **kwargs: 传递给任务的可变关键字参数。
        """
        self.interval_ms = interval_ms
        self.task = task
        self.task_args = args
        self.task_kwargs = kwargs
        self.last_run_time = time.time() * 1000
        self.running = False
        self.task_thread = None

        logger.info("PeriodicTask", f"Initialized with interval: {self.interval_ms} ms")

    def start(self):
        """启动定期任务的调度，在独立线程中运行"""
        if self.running:
            logger.warning("PeriodicTask", "Attempted to start an already running task.")
            return
        
        self.running = True
        self.task_thread = threading.Thread(target=self._run_task)
        self.task_thread.start()
        logger.info("PeriodicTask", "Task started.")

    def _run_task(self):
        """在独立线程中定期运行任务"""
        while self.running:
            current_time = time.time() * 1000
            elapsed_time = current_time - self.last_run_time

            if elapsed_time >= self.interval_ms:
                # logger.debug("PeriodicTask", f"Executing task after {elapsed_time} ms.")
                try:
                    self.task(*self.task_args, **self.task_kwargs)
                except Exception as e:
                    logger.error_trace("PeriodicTask", f"Error occurred during task execution: {str(e)}")
                self.last_run_time = current_time

            time.sleep(0.01)  # Avoid tight loop by sleeping briefly

    def stop(self):
        """停止定期任务"""
        if not self.running:
            logger.warning("PeriodicTask", "Attempted to stop a non-running task.")
            return
        
        self.running = False
        if self.task_thread:
            self.task_thread.join()
            logger.info("PeriodicTask", "Task stopped.")

# 示例任务函数
def example_task(name, count, greeting="Hello"):
    logger.verbose("PeriodicTask", f"Task executed: {greeting}, {name}! Count: {count}")

if __name__ == "__main__":
    # 初始化一个每2秒执行一次任务的PeriodicTask，并传入不定长度的参数
    periodic_task = PeriodicTask(2000, example_task, "Alice", 1, greeting="Hi")

    # 启动任务
    try:
        logger.info("Main", "Starting periodic task. It will run every 2 seconds.")
        periodic_task.start()

        # 按键终止任务
        while True:
            time.sleep(1)  # 模拟主线程的其他工作
    except KeyboardInterrupt:
        # 捕获Ctrl+C来停止任务
        logger.info("Main", "Stopping periodic task.")
        periodic_task.stop()
