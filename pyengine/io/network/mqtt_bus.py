# mqtt_bus.py
import threading
import time
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Callable, Dict, Optional, List

from pyengine.io.network.mqtt_client import MQTTClient  # 你现成的封装
from pyengine.utils.logger import logger

MessageHandler = Callable[[str, bytes], None]

@dataclass
class _Sub:
    topic: str
    qos: int
    handler: MessageHandler
    queue: Queue
    worker: threading.Thread
    active: bool = True

class MqttBus:
    """
    单实例 MQTT 总线：
    - 统一管理 connect/reconnect、遗嘱、心跳；
    - 扇出消息到各订阅者(每订阅一路独立消费队列+线程，避免阻塞网络回调)；
    - 允许复用外部 MQTTClient，或内部创建并“拥有”它。
    """
    def __init__(self,
                 host: Optional[str] = None,
                 port: Optional[int] = None,
                 client_id: str = "app_mqtt_bus",
                 mqtt_client: Optional[MQTTClient] = None,
                 reconnect_interval: float = 5.0,
                 max_queue_per_sub: int = 256):
        self._owns_client = mqtt_client is None
        self.client = mqtt_client or MQTTClient(host=host, port=port, client_id=client_id)
        self.client_id = client_id
        self.reconnect_interval = reconnect_interval
        self.max_queue_per_sub = max_queue_per_sub

        self._subs: Dict[str, List[_Sub]] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._net_thread = None
        self._heartbeat_timer = None

        # 装配“扇出回调”
        self.client.set_message_callback(self._on_message)

        # 确定是否连接
        self._was_connected = bool(getattr(self.client, "is_connected", False))

        # 可选：遗嘱
        self._will_set = False
        self._status_topic = None

    # ===== 基础连接控制 =====
    def set_will(self, topic: str, payload: bytes, retain: bool = True, qos: int = 0):
        self.client.set_will(topic, payload, retain=retain, qos=qos)
        self._will_set = True
        self._status_topic = topic

    def start(self, connect_timeout: int = 10):
        if self.client.is_connected:
            logger.info("MqttBus", "Client already connected.")
        elif self._owns_client:
            ok = self.client.connect(timeout=connect_timeout)
            if not ok:
                logger.warning("MqttBus", "Initial connect failed, will retry in background.")
        else:
            logger.info("MqttBus", "External MQTTClient provided; waiting for it to connect.")

        self._net_thread = threading.Thread(target=self._network_loop, name="MqttBusLoop", daemon=True)
        self._net_thread.start()

        if self._status_topic and self.is_connected:
            self.publish(self._status_topic, self._status_payload("online"), retain=True)

    def stop(self):
        self._stop.set()
        # 先发“offline”
        if self._status_topic:
            try:
                self.publish(self._status_topic, self._status_payload("offline"), retain=True)
                time.sleep(0.3)
            except Exception:
                pass

        # 关闭订阅工作线程
        with self._lock:
            for lst in self._subs.values():
                for sub in lst:
                    sub.active = False
                    sub.queue.put(None)  # 让 worker 退出
        # 等待网络线程结束
        if self._net_thread:
            self._net_thread.join(timeout=2.0)

        # 只有我们“拥有”client 时才断开
        if self._owns_client:
            try:
                self.client.disconnect()
            except Exception:
                pass
        logger.info("MqttBus", "Stopped.")

    @property
    def is_connected(self) -> bool:
        return bool(getattr(self.client, "is_connected", False))

    # ===== 订阅 / 发布 API(给下游组件用)=====
    def subscribe(self, topic: str, handler: MessageHandler, qos: int = 0):
        """
        注册订阅：同一 topic 可注册多个 handler。每个 handler 拥有独立队列与 worker。
        """
        q = Queue(maxsize=self.max_queue_per_sub)

        def _worker():
            while True:
                try:
                    item = q.get(timeout=0.5)
                except Empty:
                    if not sub.active and q.empty():
                        break
                    continue
                if item is None:
                    break
                t, payload = item
                try:
                    handler(t, payload)
                except Exception as e:
                    logger.error_trace("MqttBus", f"handler error on {t}: {e}")

        w = threading.Thread(target=_worker, name=f"MqttBusSub[{topic}]", daemon=True)
        sub = _Sub(topic=topic, qos=qos, handler=handler, queue=q, worker=w)
        with self._lock:
            self._subs.setdefault(topic, []).append(sub)

        # 若已连接，立即向 broker 订阅一次(幂等)
        if self.is_connected:
            self.client.subscribe(topic, qos=qos)   # 现在 wrapper 已支持 qos
        w.start()
        logger.info("MqttBus", f"Subscribed handler on '{topic}' (qos={qos}).")
        return sub  # 返回句柄以便未来取消订阅

    def unsubscribe(self, sub: _Sub):
        with self._lock:
            lst = self._subs.get(sub.topic, [])
            if sub in lst:
                lst.remove(sub)
                sub.active = False
                sub.queue.put(None)
                if not lst and self.is_connected:
                    try:
                        self.client.unsubscribe(sub.topic)  # 现在 wrapper 已提供
                    except Exception:
                        pass
        logger.info("MqttBus", f"Unsubscribed from '{sub.topic}'.")

    def publish(self, topic: str, payload: bytes, qos: int = 0, retain: bool = False):
        if not self.is_connected:
            logger.warning("MqttBus", f"publish while disconnected: {topic} ({len(payload)} bytes)")
        return self.client.publish(topic, payload, qos=qos, retain=retain)

    # ===== 内部：mqtt 回调扇出 / 自动重连 / 订阅恢复 =====
    # def _on_message(self, topic: str, payload: bytes):
    #     # 扇出：按精确 topic 分发(若需要通配符，可在这里做匹配拓展)
    #     with self._lock:
    #         targets = list(self._subs.get(topic, []))
    #     # 将消息入各自队列；满了就丢(避免阻塞网络线程)
    #     for sub in targets:
    #         try:
    #             sub.queue.put_nowait((topic, payload))
    #         except Exception:
    #             # 背压：丢弃最旧一条再放入(可选策略)
    #             try:
    #                 sub.queue.get_nowait()
    #                 sub.queue.put_nowait((topic, payload))
    #             except Exception:
    #                 pass

    def _on_message(self, topic: str, payload: bytes):
        # 扇出：支持“精确匹配 + 通配符匹配”
        targets = []
        with self._lock:
            # 精确
            targets.extend(self._subs.get(topic, []))
            # 通配符：遍历所有已注册订阅键，挑出含 '+' / '#' 或与当前 topic 非等的订阅
            for sub_topic, lst in self._subs.items():
                if sub_topic == topic:
                    continue
                if ('+' in sub_topic) or ('#' in sub_topic):
                    if _mqtt_topic_match(sub_topic, topic):
                        targets.extend(lst)

        # 入各自队列；满了就丢最旧（避免阻塞网络线程）
        for sub in targets:
            try:
                sub.queue.put_nowait((topic, payload))
            except Exception:
                try:
                    sub.queue.get_nowait()
                    sub.queue.put_nowait((topic, payload))
                except Exception:
                    pass

    def _network_loop(self):
        while not self._stop.is_set():
            now_connected = self.is_connected

            # 拥有客户端 → 才尝试重连；外部注入时交由外部或其自身机制处理
            if not now_connected and self._owns_client:
                try:
                    ok = self.client.connect(timeout=5)
                    if not ok:
                        logger.warning("MqttBus", "Reconnect failed.")
                except Exception as e:
                    logger.warning("MqttBus", f"Reconnect exception: {e}")

            # 任何来源的 False->True(包括底层 auto-reconnect)→ 统一恢复订阅 & 状态
            if now_connected and not self._was_connected:
                logger.info("MqttBus", "Connection restored; resubscribing all topics.")
                with self._lock:
                    topics = {s.topic: s.qos for lst in self._subs.values() for s in lst}
                for t, qos in topics.items():
                    try:
                        self.client.subscribe(t, qos=qos)
                    except Exception:
                        pass
                if self._status_topic:
                    self.publish(self._status_topic, self._status_payload("online"), retain=True)

            self._was_connected = now_connected
            time.sleep(self.reconnect_interval)

    def _status_payload(self, status: str) -> bytes:
        """
        生成状态消息的有效载荷。
        :param status: 状态字符串，例如 "online" 或 "offline"。
        :return: 编码为字节的有效载荷。
        """
        import json
        # 可以根据实际需求构建更复杂的 JSON 结构
        payload = {
            "client_id": self.client_id,
            "status": status,
            "timestamp": int(time.time())
        }
        return json.dumps(payload).encode("utf-8")


# --- 新增：MQTT 主题匹配（支持 + / #） ---
def _mqtt_topic_match(sub: str, topic: str) -> bool:
    # 快路径：完全相等
    if sub == topic:
        return True
    s_parts = sub.split('/')
    t_parts = topic.split('/')

    for i, s in enumerate(s_parts):
        if s == '#':
            # '#' 只能出现在最后一个层级；匹配其余全部
            return i == len(s_parts) - 1
        if s == '+':
            # 单层通配
            if i >= len(t_parts):
                return False
            continue
        # 普通文本
        if i >= len(t_parts) or s != t_parts[i]:
            return False
    # 所有订阅层级都匹配完毕，则只有当主题没有剩余层级时才算匹配
    return len(t_parts) == len(s_parts)