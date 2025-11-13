# mqtt_bus.py
import threading
import time
from collections import deque
from dataclasses import dataclass, field
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
    dropped_count: int = 0  # Track number of dropped messages due to queue full
    failed_messages: deque = field(default_factory=lambda: deque(maxlen=100))  # Keep last 100 failed messages
    max_retries: int = 3  # Maximum retry attempts per message
    retry_count: int = 0  # Total number of retries performed

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
                 max_queue_per_sub: int = 256,
                 use_topic_tree: bool = True):  # 新增：是否使用 TopicTree 优化
        self._owns_client = mqtt_client is None
        self.client = mqtt_client or MQTTClient(host=host, port=port, client_id=client_id)
        self.client_id = client_id
        self.reconnect_interval = reconnect_interval
        self.max_queue_per_sub = max_queue_per_sub
        self.use_topic_tree = use_topic_tree

        self._subs: Dict[str, List[_Sub]] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._net_thread = None
        self._heartbeat_timer = None

        # 新增：高效主题匹配树
        self._topic_tree = TopicTree() if use_topic_tree else None

        # 装配"扇出回调"
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

        Thread-safe implementation:
        1. Create subscription object first
        2. Register under lock
        3. Subscribe to broker outside lock (to avoid deadlock)
        4. Start worker thread
        """
        q = Queue(maxsize=self.max_queue_per_sub)

        # Create subscription object first (without thread)
        sub = _Sub(topic=topic, qos=qos, handler=handler, queue=q, worker=None)

        # Register subscription under lock
        should_subscribe_broker = False
        with self._lock:
            self._subs.setdefault(topic, []).append(sub)
            should_subscribe_broker = self.is_connected

            # 更新 TopicTree
            if self._topic_tree:
                self._topic_tree.insert(topic, [sub])

        # Subscribe to broker outside lock to avoid deadlock
        if should_subscribe_broker:
            try:
                self.client.subscribe(topic, qos=qos)
            except Exception as e:
                logger.error("MqttBus", f"Failed to subscribe to broker for topic '{topic}': {e}")
                # Remove from subscription list on failure
                with self._lock:
                    try:
                        self._subs[topic].remove(sub)
                    except (KeyError, ValueError):
                        pass
                raise

        # Define worker function with retry mechanism
        def _worker():
            retry_queue = deque()  # Queue for messages that need retry

            while True:
                # 1) Process retry queue first (prioritize failed messages)
                while retry_queue:
                    item, retry_num = retry_queue.popleft()
                    t, payload = item

                    try:
                        handler(t, payload)
                        logger.debug("MqttBus", f"Retry {retry_num} succeeded for topic '{t}'")
                        sub.retry_count += 1
                    except Exception as e:
                        if retry_num < sub.max_retries:
                            # Retry again
                            logger.warning("MqttBus",
                                f"Handler retry {retry_num}/{sub.max_retries} failed for topic '{t}': {e}, will retry")
                            retry_queue.append((item, retry_num + 1))
                            sub.retry_count += 1
                        else:
                            # Max retries exceeded, log and record failure
                            logger.error("MqttBus",
                                f"Handler failed after {retry_num} retries for topic '{t}': {e}")
                            sub.failed_messages.append({
                                'topic': t,
                                'payload_size': len(payload),
                                'error': str(e),
                                'timestamp': time.time(),
                                'retries': retry_num
                            })

                # 2) Process new messages from queue
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
                    # First failure, add to retry queue
                    logger.warning("MqttBus", f"Handler failed for topic '{t}': {e}, will retry")
                    retry_queue.append((item, 1))

        # Create and start worker thread
        w = threading.Thread(target=_worker, name=f"MqttBusSub[{topic}]", daemon=True)
        sub.worker = w
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

                # 从 TopicTree 中移除
                if self._topic_tree:
                    self._topic_tree.remove(sub.topic, sub)

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

    def get_subscription_stats(self):
        """
        Get statistics about all subscriptions including retry and failure information.

        Returns:
            dict: Statistics for each subscription including:
                - dropped_count: Number of messages dropped due to full queue
                - retry_count: Total number of retry attempts
                - failed_count: Number of messages that failed after max retries
                - recent_failures: List of recent failure details
        """
        stats = {}
        with self._lock:
            for topic, subs in self._subs.items():
                topic_stats = []
                for idx, sub in enumerate(subs):
                    topic_stats.append({
                        'handler_index': idx,
                        'dropped_count': sub.dropped_count,
                        'retry_count': sub.retry_count,
                        'failed_count': len(sub.failed_messages),
                        'recent_failures': list(sub.failed_messages),
                        'queue_size': sub.queue.qsize(),
                        'active': sub.active
                    })
                stats[topic] = topic_stats
        return stats

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
        """
        扇出消息到所有匹配的订阅者。
        支持精确匹配和通配符匹配。

        使用 TopicTree 实现 O(深度) 的高效匹配，相比传统的 O(n*m) 线性扫描提升显著。

        Backpressure handling:
        - If queue is full, drop oldest message and add new one
        - Log warnings and track dropped message count
        """
        # 扇出：使用 TopicTree 进行高效匹配
        targets = []
        with self._lock:
            if self._topic_tree:
                # 使用 TopicTree 进行 O(深度) 匹配
                targets = self._topic_tree.find_matches(topic)
            else:
                # Fallback: 传统的 O(n*m) 线性扫描
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
            except Exception as e:
                # Queue is full, apply backpressure by dropping oldest message
                logger.warning("MqttBus", f"Queue full for topic '{topic}' (handler on '{sub.topic}'), dropping oldest message")
                try:
                    # Drop oldest message
                    sub.queue.get_nowait()
                    # Try to add new message
                    sub.queue.put_nowait((topic, payload))
                    # Track dropped messages
                    sub.dropped_count += 1
                    # Log if too many drops
                    if sub.dropped_count % 100 == 0:
                        logger.warning("MqttBus", f"Topic '{sub.topic}' has dropped {sub.dropped_count} messages due to slow handler")
                except Exception as inner_e:
                    logger.error("MqttBus", f"Failed to handle backpressure for topic '{topic}': {inner_e}")

    def _network_loop(self):
        while not self._stop.is_set():
            now_connected = self.is_connected

            # 拥有客户端 → 才尝试重连；外部注入时交由外部或其自身机制处理
            # if not now_connected and self._owns_client:
            #     try:
            #         ok = self.client.connect(timeout=5)
            #         if not ok:
            #             logger.warning("MqttBus", "Reconnect failed.")
            #     except Exception as e:
            #         logger.warning("MqttBus", f"Reconnect exception: {e}")

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
    """
    Legacy topic matching function (kept for backward compatibility).
    新代码应该使用 TopicTree 类进行高效匹配。
    """
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


class TopicTree:
    """
    高效的 MQTT 主题树，使用 Trie 结构实现 O(深度) 的主题匹配。

    相比传统的 O(n*m) 线性扫描（n=订阅数，m=主题深度），
    TopicTree 使用字典树结构，将匹配复杂度降低到 O(m)。

    支持 MQTT 通配符：
    - '+': 单层通配符，匹配一个层级
    - '#': 多层通配符，匹配剩余所有层级（只能在末尾）

    示例：
        tree = TopicTree()
        tree.insert("sensor/+/temperature", handler1)
        tree.insert("sensor/#", handler2)
        matches = tree.find_matches("sensor/room1/temperature")
        # 返回 [handler1, handler2] 的订阅列表
    """

    def __init__(self):
        """初始化空的主题树"""
        self.root = {}  # {level: {children: {...}, handlers: [...]}}

    def insert(self, topic_pattern: str, subscriptions: List[_Sub]):
        """
        插入主题模式和对应的订阅列表。

        Args:
            topic_pattern: MQTT 主题模式，可包含 '+' 和 '#' 通配符
            subscriptions: 订阅该模式的 _Sub 对象列表
        """
        if not subscriptions:
            return

        parts = topic_pattern.split('/')
        node = self.root

        for idx, part in enumerate(parts):
            # 为每个层级创建或获取节点
            if part not in node:
                node[part] = {'children': {}, 'handlers': []}

            # 对于 '#' 通配符，直接在当前层级添加处理器并结束
            if part == '#':
                node[part]['handlers'].extend(subscriptions)
                return

            # 移动到下一层
            node = node[part]['children']

        # 在叶子节点添加处理器
        node.setdefault('$', {'children': {}, 'handlers': []})
        node['$']['handlers'].extend(subscriptions)

    def find_matches(self, topic: str) -> List[_Sub]:
        """
        查找匹配指定主题的所有订阅。

        Args:
            topic: 实际的 MQTT 主题（不含通配符）

        Returns:
            匹配的 _Sub 对象列表
        """
        parts = topic.split('/')
        matches = []
        self._find_recursive(parts, 0, self.root, matches)
        return matches

    def _find_recursive(self, parts: List[str], idx: int, node: dict, matches: List[_Sub]):
        """
        递归查找匹配的订阅。

        Args:
            parts: 主题分割后的部分
            idx: 当前匹配到的层级索引
            node: 当前树节点
            matches: 累积的匹配结果列表
        """
        # 已经匹配完所有层级
        if idx == len(parts):
            # 查找精确匹配的处理器（标记为 '$'）
            if '$' in node:
                matches.extend(node['$']['handlers'])
            return

        current_part = parts[idx]

        # 1. 精确匹配当前层级
        if current_part in node:
            self._find_recursive(parts, idx + 1, node[current_part]['children'], matches)

        # 2. '+' 通配符匹配当前层级
        if '+' in node:
            self._find_recursive(parts, idx + 1, node['+']['children'], matches)

        # 3. '#' 通配符匹配剩余所有层级
        if '#' in node:
            matches.extend(node['#']['handlers'])

    def remove(self, topic_pattern: str, subscription: _Sub):
        """
        移除指定主题模式的订阅。

        Args:
            topic_pattern: MQTT 主题模式
            subscription: 要移除的 _Sub 对象
        """
        parts = topic_pattern.split('/')
        self._remove_recursive(parts, 0, self.root, subscription)

    def _remove_recursive(self, parts: List[str], idx: int, node: dict, subscription: _Sub) -> bool:
        """
        递归移除订阅。

        Returns:
            是否应该清理当前节点（如果为空）
        """
        if idx == len(parts):
            # 到达叶子节点
            if '$' in node and subscription in node['$']['handlers']:
                node['$']['handlers'].remove(subscription)
                # 如果处理器列表为空，标记可以清理
                return len(node['$']['handlers']) == 0
            return False

        part = parts[idx]
        if part == '#' and part in node:
            if subscription in node[part]['handlers']:
                node[part]['handlers'].remove(subscription)
                return len(node[part]['handlers']) == 0
            return False

        if part not in node:
            return False

        # 递归处理子节点
        should_clean = self._remove_recursive(parts, idx + 1, node[part]['children'], subscription)

        # 如果子节点为空且没有处理器，清理当前节点
        if should_clean and not node[part]['handlers'] and not node[part]['children']:
            del node[part]
            return True

        return False

    def rebuild(self, subs_dict: Dict[str, List[_Sub]]):
        """
        从订阅字典重建整个主题树。

        Args:
            subs_dict: {topic_pattern: [_Sub, ...]} 字典
        """
        self.root = {}
        for topic, subscriptions in subs_dict.items():
            self.insert(topic, subscriptions)