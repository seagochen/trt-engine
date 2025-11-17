import paho.mqtt.client as mqtt
from typing import Optional, Callable
from pyengine.utils.logger import logger
import time
import threading


class MQTTClient:
    def __init__(self, host: str, port: int, client_id: str,
                 username: Optional[str] = None, password: Optional[str] = None,
                 max_reconnect_attempts: int = 5, reconnect_interval: int = 10):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.username = "" if username is None else username
        self.password = "" if password is None else password
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_interval = reconnect_interval

        self.client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311, transport="tcp")
        self.is_connected = False
        self.message_callback = None

        # --- 新增 1: 用于同步连接的 Event 对象 ---
        # threading.Event 是一个简单的线程同步原语。
        self.connected_event = threading.Event()
        # 锁保护连接事件操作，防止 clear/set 之间的竞态条件
        self._connect_lock = threading.Lock()

        # 设置用户名和密码
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        # 设置回调函数
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message

    # --- 新增: 设置遗嘱消息的方法 ---
    def set_will(self, topic: str, payload: bytes, qos: int = 1, retain: bool = True):
        """
        设置客户端的遗嘱消息 (Last Will and Testament).
        这必须在 connect() 之前调用。
        """
        try:
            self.client.will_set(topic, payload, qos, retain)
            logger.info("MQTTClient", f"Will set on topic '{topic}'")
        except Exception as e:
            logger.error_trace("MQTTClient", f"Failed to set will: {str(e)}")

    def set_message_callback(self, callback: Callable[[str, bytes], None]):
        """设置接收消息的回调函数"""
        self.message_callback = callback

    def subscribe(self, topic: str, qos: int = 0) -> bool:
        """订阅主题(支持 qos)"""
        if not self.is_connected:
            logger.warning("MQTTClient", "Client not connected. Cannot subscribe to topic.")
            return False
        try:
            self.client.subscribe(topic, qos=qos)
            return True
        except Exception as e:
            logger.error_trace("MQTTClient", f"Failed to subscribe to {topic} - {str(e)}")
            return False

    def unsubscribe(self, topic: str) -> bool:
        """退订主题"""
        if not self.is_connected:
            logger.warning("MQTTClient", "Client not connected. Cannot unsubscribe.")
            return False
        try:
            self.client.unsubscribe(topic)
            return True
        except Exception as e:
            logger.error_trace("MQTTClient", f"Failed to unsubscribe from {topic} - {str(e)}")
            return False

    # --- 修改 1: 重写 connect 方法，使其同步阻塞 ---
    def connect(self, timeout: int = 10) -> bool:
        """
        同步连接到MQTT broker。
        此方法将阻塞直到连接成功或超时。

        修复:
        1. 确保在异常情况下也正确停止loop
        2. 使用锁保护连接事件，避免 clear/set 之间的竞态条件
        """
        loop_started = False

        try:
            # 使用锁保护连接事件的清除和后续等待
            with self._connect_lock:
                self.connected_event.clear()  # 在锁内重置事件状态
                self.client.connect(self.host, self.port, 60)
                self.client.loop_start()
                loop_started = True
                logger.info("MQTTClient", f"Waiting for connection to {self.host}:{self.port}...")

            # 锁外等待事件 - 允许回调函数设置事件
            # .wait() 会阻塞当前线程，直到另一个线程调用 .set() 或超时
            # 如果事件被设置，返回 True；如果超时，返回 False
            if self.connected_event.wait(timeout=timeout):
                # 事件被触发，返回由 on_connect 设置的最终连接状态
                logger.info("MQTTClient", f"Successfully connected to {self.host}:{self.port}")
                return self.is_connected
            else:
                # 等待超时
                logger.error("MQTTClient",
                             f"Connection to {self.host}:{self.port} timed out after {timeout} seconds.")
                self.disconnect()  # 连接超时，清理资源
                return False

        except Exception as e:
            logger.error_trace("MQTTClient", f"Failed to initiate connect to {self.host}:{self.port} - {str(e)}")

            # 确保清理: 如果loop已启动但连接失败，必须停止loop
            if loop_started:
                try:
                    self.client.loop_stop()
                    logger.debug("MQTTClient", "Stopped MQTT loop after connection exception")
                except Exception as stop_e:
                    logger.error("MQTTClient", f"Error stopping MQTT loop: {stop_e}")

            return False

    def disconnect(self):
        """断开连接"""
        # 检查客户端是否还在运行，以避免在超时后重复操作
        if self.client.is_connected() or self.is_connected:
            self.client.loop_stop()
            self.client.disconnect()

    def publish(self, topic: str, payload: bytes, qos: int = 1, retain: bool = False) -> bool:
        """发布消息"""
        if not self.is_connected:
            logger.warning("MQTTClient", "Client not connected. Cannot publish message.")
            return False
        try:
            # Pass the qos and retain arguments to the underlying client
            self.client.publish(topic, payload, qos=qos, retain=retain)
            return True
        except Exception as e:
            logger.error_trace("MQTTClient", f"Failed to publish message to {topic} - {str(e)}")
            return False

    # --- 修改 2: 修改 on_connect 回调 ---
    def on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            logger.info("MQTTClient", f"Client [{self.client_id}] connected to the MQTT broker at {self.host}:{self.port}")
            self.is_connected = True
        else:
            logger.error("MQTTClient", f"Failed to connect: {mqtt.error_string(rc)}")
            self.is_connected = False

        # --- 新增 2: 无论成功与否，都设置事件，以唤醒并解除 connect 方法的阻塞 ---
        self.connected_event.set()

    # --- 修改 3: 修改 on_disconnect 回调 ---
    def on_disconnect(self, client, userdata, rc):
        """断开连接回调"""
        self.is_connected = False
        self.connected_event.clear()  # 连接已断开，清除事件状态

        if rc == 0:
            logger.info("MQTTClient", "Disconnected from the MQTT broker gracefully.")
        else:
            logger.error("MQTTClient", f" Client [{self.client_id}] experienced an unexpected disconnect: {mqtt.error_string(rc)}")
            # 当非正常断开时，启动重连线程
            threading.Thread(target=self._attempt_reconnect, daemon=True).start()

    def on_message(self, client, userdata, msg):
        """消息回调"""
        if self.message_callback:
            self.message_callback(msg.topic, msg.payload)

    def _attempt_reconnect(self):
        """按照指定次数和间隔尝试重新连接MQTT服务器"""
        # (此函数无需修改)
        attempts = 0
        while attempts < self.max_reconnect_attempts and not self.is_connected:
            attempts += 1
            try:
                logger.info("MQTTClient", f"Client [{self.client_id}] is attempting to reconnect... (Attempt {attempts})")
                # reconnect() 会自动处理连接逻辑，并触发 on_connect 回调
                self.client.reconnect()
                # 等待 on_connect 确认
                self.connected_event.wait(timeout=5)
            except Exception as e:
                logger.error_trace("MQTTClient", f"Reconnection attempt {attempts} failed: {str(e)}")

            if not self.is_connected:
                time.sleep(self.reconnect_interval)
        if not self.is_connected:
            logger.error("MQTTClient", f"Could not reconnect after {self.max_reconnect_attempts} attempts.")


# Example usage (示例用法保持不变，但现在它的行为是同步的)
if __name__ == "__main__":

    def message_handler(topic, payload):
        logger.info("MQTTClient", f"Received message on topic '{topic}': {payload.decode('utf-8')}")


    mqtt_client = MQTTClient(host="localhost", port=1883, client_id="test_client_sync")
    mqtt_client.set_message_callback(message_handler)

    # connect() 现在会阻塞，直到连接成功或超时
    if mqtt_client.connect():
        logger.info("MQTTClient", "Connection successful!")
        mqtt_client.subscribe("test/topic")
        mqtt_client.publish("test/topic", b"Hello, Synchronous MQTT!")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            mqtt_client.disconnect()
    else:
        logger.error("MQTTClient", "Connection failed!")