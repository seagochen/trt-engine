import time
from typing import Optional
from pyengine.utils.logger import logger

class PowerController:
    """
    通用节流/省电控制器：
    - 长时间无“活动事件” -> 进入 idle(省电/低频)
    - idle 模式下：按 idle_interval 节流执行
    - 收到“活动事件” -> 退出 idle，恢复高频
    """

    def __init__(
        self,
        idle_after: float = 30.0,     # ≈ detection_timeout
        idle_interval: float = 1.0,   # 省电/idle模式下的执行间隔
        active_interval: float = 0.0, # 活跃模式节流间隔(0=不节流，可按需设置)
        name: str = "APP_MIDWARE_POSE"
    ):
        self._name = name
        self._idle_after = idle_after
        self._idle_interval = idle_interval
        self._active_interval = active_interval

        now = time.time()
        self._idle = False
        self._last_activity_ts = now      # ≈ last_detection_time
        self._last_execute_ts = 0.0       # ≈ last_inference_time
        self._next_allowed_ts: float = 0.0  # 统一节流门(dense/idle都用它判断)

    # ---------- 通用命名的API ----------
    def maybe_enter_idle(self, now: Optional[float] = None) -> bool:
        """
        长时间无活动则进入 idle。返回是否刚刚进入。
        """
        now = now or time.time()
        if (not self._idle) and (now - self._last_activity_ts) >= self._idle_after:
            self._idle = True
            self._next_allowed_ts = 0.0
            logger.info(self._name, "Entering idle mode")
            return True
        return False

    def should_throttle(self, now: Optional[float] = None) -> bool:
        """
        是否应跳过本次执行(基于统一的 next_allowed_ts)。
        """
        now = now or time.time()
        return now < self._next_allowed_ts

    def mark_executed(self, now: Optional[float] = None):
        """
        一次执行完成后调用：推进下次允许执行时间(idle/active 各自的 interval)。
        """
        now = now or time.time()
        self._last_execute_ts = now
        interval = self._idle_interval if self._idle else self._active_interval
        self._next_allowed_ts = now + max(0.0, interval)

    def record_activity(self, now: Optional[float] = None):
        """
        记录一次“活动事件”(例如检测到目标、有人移动、音量触发等)。
        会刷新 last_activity，并在 idle 时唤醒为 active。
        """
        now = now or time.time()
        self._last_activity_ts = now
        if self._idle:
            self._idle = False
            logger.info(self._name, "Exiting idle mode")

    def is_idle(self) -> bool:
        return self._idle

    # ---------- 参数调节(可选) ----------
    def set_idle_after(self, seconds: float): self._idle_after = float(seconds)
    def set_idle_interval(self, seconds: float): self._idle_interval = float(seconds)
    def set_active_interval(self, seconds: float): self._active_interval = float(seconds)


# import time
# from power_controller import PowerController  # 假设你把类放在 power_controller.py
#
# def fake_inference_step():
#     """假装推理，返回是否检测到目标"""
#     # 举例：每 5 次循环才“检测到目标”
#     return int(time.time()) % 5 == 0
#
# def main_loop():
#     pc = PowerController(idle_after=10.0, idle_interval=1.0, active_interval=0.0)
#
#     for step in range(50):
#         now = time.time()
#
#         # 1. 先检查是否需要进入 idle
#         pc.maybe_enter_idle(now)
#
#         # 2. 如果当前应该节流 -> 跳过这次推理
#         if pc.should_throttle(now):
#             print(f"[{step}] skip (idle={pc.is_idle()})")
#             time.sleep(0.2)
#             continue
#
#         # 3. 执行一次推理
#         detected = fake_inference_step()
#         print(f"[{step}] run inference, detected={detected}, idle={pc.is_idle()}")
#
#         # 4. 记录执行时间，决定下次允许执行的时间
#         pc.mark_executed(now)
#
#         # 5. 如果有检测到目标 -> 记录活动，可能会唤醒 idle
#         if detected:
#             pc.record_activity(now)
#
#         # 模拟帧间隔
#         time.sleep(0.2)
#
# if __name__ == "__main__":
#     main_loop()
