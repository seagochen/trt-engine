# -*- coding: utf-8 -*-
"""
Heartbeat Receiver Plugin
- 订阅一个或多个心跳主题（可含 MQTT 通配符 +/#，需底层/Bus 支持）
- 解析 JSON 载荷：{ "client_id": str, "status": "online"|"offline", "timestamp": int, ... }
- 为每个“实际 topic”维护最新状态与时间戳
- 超时自动将 online → stale(timeout)
- 对外提供三态查询:
    receiver.get_state("<exact topic>") -> "online" | "stale" | "offline" | None
    receiver.get_detail("<exact topic>") -> HeartbeatState | None
    receiver.get_snapshot() -> Dict[str, HeartbeatState]
"""

import json
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from pyengine.io.network.mqtt_plugins import MqttPlugin, IMqttHost
from pyengine.utils.logger import logger


# ====== 状态数据结构 ======
@dataclass
class HeartbeatState:
    topic: str                       # 实际接收到消息的 topic（非订阅模式串）
    client_id: str
    last_status: str                 # 'online' | 'offline' | 'stale(timeout)'
    last_hb_ts: int                  # 发送端心跳内的 timestamp（秒）
    last_rx_monotonic: float         # 本机接收该消息的 monotonic 时间
    extra: Dict[str, Any] = field(default_factory=dict)  # 透传其它字段


# ====== 插件实现 ======
class HeartbeatReceiverPlugin(MqttPlugin):
    """
    心跳接收插件（按“实际 topic”建档）：
    - 支持传入 topics=['pipelines/+/status', 'magistrates/+/status'] 等
    - 三态：
        online  -> 收到 online 且未超时
        stale   -> 上次为 online，当前超过 timeout_sec 未再收到心跳
        offline -> 收到显式 offline
    - on_status_change(client_id, is_online: bool, state: HeartbeatState)
      * 注意：当转为 stale 时，is_online=False
    - on_sweep(snapshot: Dict[str, HeartbeatState])
    """

    def __init__(
        self,
        topics: Optional[List[str]] = None,
        topic: Optional[str] = None,
        qos: int = 0,
        timeout_sec: float = 45.0,
        sweep_interval_sec: float = 5.0,
        on_status_change: Optional[Callable[[str, bool, HeartbeatState], None]] = None,
        on_sweep: Optional[Callable[[Dict[str, HeartbeatState]], None]] = None,
        debug: bool = False,
    ):
        if topic and topics:
            raise ValueError("Use either 'topic' or 'topics', not both.")
        self.topics: List[str] = topics or ([topic] if topic else [])
        self.qos = qos
        self.timeout_sec = float(timeout_sec)
        self.sweep_interval_sec = float(sweep_interval_sec)
        self.on_status_change = on_status_change
        self.on_sweep = on_sweep
        self.debug = debug

        self._host: Optional[IMqttHost] = None
        self._stop_evt = threading.Event()
        self._sweep_thread: Optional[threading.Thread] = None
        self._subs: List[Tuple[str, Any]] = []   # (topic_pattern, handle_or_none)

        # 状态：按“实际 topic”索引
        self._states_by_topic: Dict[str, HeartbeatState] = {}
        # 可选：便于通过 client_id 查最近一次 topic
        self._last_topic_by_client: Dict[str, str] = {}

        self._lock = threading.RLock()

    # ===== 生命周期 =====
    def start(self, host: IMqttHost) -> None:
        if not self.topics:
            raise ValueError("HeartbeatReceiverPlugin requires at least one topic.")
        self._host = host
        self._stop_evt.clear()

        # 订阅
        for t in self.topics:
            handle = None
            try:
                # 你的 MqttBus 支持 subscribe(topic, handler, qos)
                handle = host.subscribe(t, self._on_message, qos=self.qos)  # type: ignore
                logger.info("HeartbeatReceiver", f"Subscribed '{t}' (qos={self.qos})")
            except TypeError:
                # 若底层是某些 client 封装：subscribe(topic, qos)
                handle = host.subscribe(t, self.qos)  # type: ignore
                logger.info("HeartbeatReceiver", f"Subscribed(client) '{t}' (qos={self.qos})")
            except Exception as e:
                logger.error_trace("HeartbeatReceiver", f"Subscribe failed on {t}: {e}")
            self._subs.append((t, handle))

        # 巡检线程
        self._sweep_thread = threading.Thread(
            target=self._sweep_loop,
            name="HeartbeatReceiverSweep",
            daemon=True,
        )
        self._sweep_thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._sweep_thread:
            self._sweep_thread.join(timeout=1.0)
        self._sweep_thread = None

        # 退订
        if self._host:
            for t, h in self._subs:
                try:
                    if hasattr(self._host, "unsubscribe"):
                        # 优先用句柄退订；否则按 topic 退订
                        try:
                            self._host.unsubscribe(h)  # type: ignore
                        except Exception:
                            self._host.unsubscribe(t)  # type: ignore
                except Exception:
                    pass
        self._subs.clear()
        self._host = None

    # ===== 消息处理 =====
    def _on_message(self, topic: str, payload: bytes) -> None:
        """
        标准心跳 JSON 载荷：
        {
            "client_id": "...",
            "status": "online" | "offline",
            "timestamp": 172xxxxxxx,   # 秒
            ... 其他字段
        }
        """
        now_mono = time.monotonic()
        try:
            obj = json.loads(payload.decode("utf-8"))
            client_id = str(obj.get("client_id", "unknown"))
            status = str(obj.get("status", "online")).lower()
            hb_ts = int(obj.get("timestamp", int(time.time())))
        except Exception as e:
            logger.warning("HeartbeatReceiver", f"Bad payload on {topic}: {e}")
            return

        # 仅接受 online/offline 两类源状态；其余一律按 online 处理
        if status not in ("online", "offline"):
            status = "online"

        extra = {k: v for k, v in obj.items() if k not in ("client_id", "status", "timestamp")}

        with self._lock:
            old = self._states_by_topic.get(topic)
            new_state = HeartbeatState(
                topic=topic,
                client_id=client_id,
                last_status=status,           # online/offline；stale 只在巡检中产生
                last_hb_ts=hb_ts,
                last_rx_monotonic=now_mono,
                extra=extra,
            )
            self._states_by_topic[topic] = new_state
            self._last_topic_by_client[client_id] = topic

        # 仅当 online/offline 发生变化时回调
        if (old is None) or (old.last_status != status):
            if self.on_status_change:
                try:
                    self.on_status_change(client_id, status == "online", new_state)
                except Exception as e:
                    logger.error_trace("HeartbeatReceiver", f"on_status_change error: {e}")

        if self.debug:
            logger.debug("HeartbeatReceiver", f"HB {topic} {client_id} -> {status} @ {hb_ts}")

    # ===== 巡检：online 超时 → stale =====
    def _sweep_loop(self) -> None:
        while not self._stop_evt.is_set():
            time.sleep(self.sweep_interval_sec)
            now_mono = time.monotonic()

            with self._lock:
                snapshot = {t: HeartbeatState(**vars(st)) for t, st in self._states_by_topic.items()}

            updates: List[HeartbeatState] = []
            for t, st in snapshot.items():
                # 仅当“记录为 online 且超时”才转 stale
                if st.last_status == "online" and (now_mono - st.last_rx_monotonic) > self.timeout_sec:
                    st.last_status = "stale(timeout)"
                    updates.append(st)

            if updates:
                # 写回与回调
                with self._lock:
                    for st in updates:
                        self._states_by_topic[st.topic] = st
                for st in updates:
                    if self.on_status_change:
                        try:
                            self.on_status_change(st.client_id, False, st)
                        except Exception as e:
                            logger.error_trace("HeartbeatReceiver", f"on_status_change error: {e}")

            if self.on_sweep:
                try:
                    self.on_sweep(snapshot)
                except Exception as e:
                    logger.error_trace("HeartbeatReceiver", f"on_sweep error: {e}")

    # ===== 对外查询接口 =====
    def get_state(self, topic: str) -> Optional[str]:
        """
        返回 'online' | 'stale' | 'offline' 或 None（未知 topic）
        - stale：记录为 'stale(timeout)'，或虽为 online 但此刻已超时
        - offline：显式收到 offline
        - online：记录为 online 且未超时
        """
        with self._lock:
            st = self._states_by_topic.get(topic)
            if not st:
                return None
            # 显式 offline 永远视为 offline
            if st.last_status == "offline":
                return "offline"
            # 记录为 stale(timeout)
            if st.last_status == "stale(timeout)":
                return "stale"
            # 记录为 online：再做一次即时超时校验，避免边界时刻误判
            if (time.monotonic() - st.last_rx_monotonic) > self.timeout_sec:
                return "stale"
            return "online"

    def get_detail(self, topic: str) -> Optional[HeartbeatState]:
        with self._lock:
            st = self._states_by_topic.get(topic)
            return HeartbeatState(**vars(st)) if st else None

    def get_snapshot(self) -> Dict[str, HeartbeatState]:
        with self._lock:
            return {t: HeartbeatState(**vars(st)) for t, st in self._states_by_topic.items()}

    # ===== 便捷：通过 client_id 找到其最近一次心跳 topic（可选） =====
    def get_last_topic_by_client(self, client_id: str) -> Optional[str]:
        with self._lock:
            return self._last_topic_by_client.get(client_id)
