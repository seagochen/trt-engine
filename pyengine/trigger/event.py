import time
from typing import Callable

from pyengine.trigger.periodic import PeriodicTask
from pyengine.utils.logger import logger


class EventTracker:

    def __init__(self, task: Callable, delay_alarm_time=1, max_age_time=10, timeout=1000, threshold=10):
        self.delay_alarm_time = delay_alarm_time  # 延迟报警时间
        self.max_age_time = max_age_time  # 最大存在时间，超过这个时间会被清理
        self.timeout = timeout  # 定期清理的间隔时间（毫秒）
        self.threshold = threshold  # 触发任务的阈值
        self.task = task  # 需要执行的任务
        self.event_dict = {}  # 存储事件及其时间和触发次数
        self.last_event_time = time.time()

        # 使用 PeriodicTask 来定期清理过期任务，无需手动线程
        self.cleaner = PeriodicTask(self.timeout, self.clean)
        
        logger.info("EventTracker", f"Initialized EventTracker with delay_alarm_time={self.delay_alarm_time}, "
                                    f"max_age_time={self.max_age_time}, timeout={self.timeout}, "
                                    f"threshold={self.threshold}.")

    def track_event(self, event_name, **kwargs):
        """追踪事件，并在满足条件时触发任务"""
        current_time = time.time()

        # 如果事件触发时间超过延迟时间，记录日志
        if current_time - self.last_event_time > self.delay_alarm_time:
            logger.critical("EventTracker", f"Event {event_name} is triggered.")
            self.last_event_time = current_time

        # 如果事件已经存在，更新事件触发的时间和计数
        if event_name in self.event_dict:
            last_time, count = self.event_dict[event_name]

            # 如果超过了最大存在时间，重置时间和计数
            if current_time - last_time > self.max_age_time:
                self.event_dict[event_name] = (current_time, 0)
                # logger.debug("EventTracker", f"Event {event_name} exceeded max_age_time. Resetting count.")

            else:
                count += 1
                # logger.verbose("EventTracker", f"Event {event_name} count incremented to {count}.")

            # 当事件触发的次数超过阈值且超过延迟时间时，执行任务
            if count > self.threshold and \
                    current_time - last_time > self.delay_alarm_time and \
                    self.task is not None:
                self.task(current_time, event_name, **kwargs)

            # 更新事件触发的时间和计数
            self.event_dict[event_name] = (last_time, count)

        # 如果事件是新事件，添加到事件字典中
        else:
            self.event_dict[event_name] = (current_time, 0)
            # logger.debug("EventTracker", f"New event tracked: {event_name}. Added to event_dict.")

    def start(self):
        """启动清理任务"""
        self.cleaner.start()
        logger.info("EventTracker", "Periodic cleaner task started.")

    def stop(self):
        """停止清理任务"""
        self.cleaner.stop()
        logger.info("EventTracker", "Periodic cleaner task stopped.")

    def clean(self):
        """定期清理过期任务"""
        current_time = time.time()
        # logger.verbose("EventTracker", "Starting periodic cleaning of expired events.")

        # 删除超过最大存在时间的事件
        for event_name, (last_time, _) in list(self.event_dict.items()):
            if current_time - last_time > self.max_age_time:
                del self.event_dict[event_name]
                # logger.debug("EventTracker", f"Event {event_name} expired and was removed.")

        # logger.verbose("EventTracker", "Periodic cleaning completed.")
