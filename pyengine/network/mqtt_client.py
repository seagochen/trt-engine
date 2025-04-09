import paho.mqtt.client as mqtt
from typing import Optional, Callable
from pyengine.utils.logger import logger
import time
import threading

class MQTTClient:
    def __init__(self, address: str, port: int, client_id: str, 
                 username: Optional[str] = None, password: Optional[str] = None,
                 max_reconnect_attempts: int = 5, reconnect_interval: int = 10):
        self.address = address
        self.port = port
        self.client_id = client_id
        self.username = "" if username is None else username
        self.password = "" if password is None else password
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_interval = reconnect_interval

        self.client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311, transport="tcp")
        self.is_connected = False
        self.message_callback = None

        # 设置用户名和密码
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        # 设置回调函数
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message

    def set_message_callback(self, callback: Callable[[str, bytes], None]):
        """设置接收消息的回调函数"""
        self.message_callback = callback

    def connect(self) -> bool:
        """连接到MQTT broker"""
        try:
            self.client.connect(self.address, self.port, 60)
            # 启动事件循环线程
            self.client.loop_start()
            return True
        except Exception as e:
            logger.error_trace("MQTTClient", f"Failed to connect to {self.address}:{self.port} - {str(e)}")
            return False

    def disconnect(self):
        """断开连接"""
        if self.is_connected:
            self.client.disconnect()
            self.client.loop_stop()

    def publish(self, topic: str, payload: bytes) -> bool:
        """发布消息"""
        try:
            self.client.publish(topic, payload)
            return True
        except Exception as e:
            logger.error_trace("MQTTClient", f"Failed to publish message to {topic} - {str(e)}")
            return False

    def subscribe(self, topic: str) -> bool:
        """订阅主题"""
        try:
            self.client.subscribe(topic)
            return True
        except Exception as e:
            logger.error_trace("MQTTClient", f"Failed to subscribe to {topic} - {str(e)}")
            return False

    # 回调函数
    def on_connect(self, client, userdata, flags, rc):
        """连接回调"""
        if rc == 0:
            logger.info("MQTTClient", f"Connected to the MQTT broker at {self.address}:{self.port}")
            self.is_connected = True
        else:
            logger.error("MQTTClient", f"Failed to connect: {mqtt.error_string(rc)}")

    def on_disconnect(self, client, userdata, rc):
        """断开连接回调"""
        if rc == 0:
            logger.info("MQTTClient", "Disconnected from the MQTT broker gracefully.")
        else:
            logger.error("MQTTClient", f"Unexpected disconnection: {mqtt.error_string(rc)}")
            # 当非正常断开时，启动重连线程
            threading.Thread(target=self._attempt_reconnect, daemon=True).start()
        self.is_connected = False

    def on_message(self, client, userdata, msg):
        """消息回调"""
        if self.message_callback:
            self.message_callback(msg.topic, msg.payload)

    def _attempt_reconnect(self):
        """按照指定次数和间隔尝试重新连接MQTT服务器"""
        attempts = 0
        while attempts < self.max_reconnect_attempts and not self.is_connected:
            attempts += 1
            try:
                logger.info("MQTTClient", f"Attempting to reconnect... (Attempt {attempts})")
                self.client.reconnect()
                # 如果连接成功，on_connect回调会将 is_connected 置为 True
            except Exception as e:
                logger.error_trace("MQTTClient", f"Reconnection attempt {attempts} failed: {str(e)}")
            if not self.is_connected:
                time.sleep(self.reconnect_interval)
        if not self.is_connected:
            logger.error("MQTTClient", f"Could not reconnect after {self.max_reconnect_attempts} attempts.")

# Example usage
if __name__ == "__main__":

    def message_handler(topic, payload):
        print(f"Received message on topic '{topic}': {payload.decode('utf-8')}")

    # Initialize the MQTT client, 设定重连最多 5 次，每次等待 10 秒
    mqtt_client = MQTTClient(address="localhost", port=1883, client_id="test_client", 
                             username="", password="", max_reconnect_attempts=5, reconnect_interval=10)

    # Set message callback
    mqtt_client.set_message_callback(message_handler)

    # Connect to the broker
    if mqtt_client.connect():
        # Subscribe to a topic
        mqtt_client.subscribe("test/topic")

        # Publish a message
        mqtt_client.publish("test/topic", b"Hello, MQTT!")

        # Wait for messages
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            # Disconnect when exiting
            mqtt_client.disconnect()
