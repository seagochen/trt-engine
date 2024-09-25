import paho.mqtt.client as mqtt
from typing import Callable, Optional, Any


class MQTTClient:
    def __init__(self, address: str, port: int, client_id: str, username: str = "", password: str = ""):
        self.address = address
        self.port = port
        self.client_id = client_id
        self.username = username
        self.password = password
        self.client = mqtt.Client(client_id=client_id)
        self.is_connected = False
        self.message_callback: Optional[Callable[[str, bytes, Any], None]] = None
        self.message_callback_arg: Any = None

        if username and password:
            self.client.username_pw_set(username, password)

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def set_message_callback(self, callback: Callable[[str, bytes, Any], None], arg: Any = None):
        self.message_callback = callback
        self.message_callback_arg = arg

    def connect(self) -> bool:
        try:
            self.client.connect(self.address, self.port, 60)
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        if self.is_connected:
            self.client.disconnect()

    def publish(self, topic: str, payload: bytes) -> bool:
        try:
            self.client.publish(topic, payload)
            return True
        except Exception as e:
            print(f"Failed to publish: {e}")
            return False

    def subscribe(self, topic: str) -> bool:
        try:
            self.client.subscribe(topic)
            return True
        except Exception as e:
            print(f"Failed to subscribe: {e}")
            return False

    def listen(self, timeout: float = 1.0) -> bool:
        try:
            self.client.loop(timeout=timeout)
            return True
        except Exception as e:
            print(f"Error in message loop: {e}")
            return False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to the MQTT broker successfully.")
            self.is_connected = True
        else:
            print(f"Failed to connect: {mqtt.connack_string(rc)}")

    def _on_disconnect(self, client, userdata, rc):
        if rc == 0:
            print("Disconnected from the MQTT broker gracefully.")
        else:
            print(f"Unexpected disconnection from the MQTT broker: {mqtt.error_string(rc)}")
        self.is_connected = False

    def _on_message(self, client, userdata, msg):
        if self.message_callback:
            self.message_callback(msg.topic, msg.payload, self.message_callback_arg)

    def __del__(self):
        self.disconnect()