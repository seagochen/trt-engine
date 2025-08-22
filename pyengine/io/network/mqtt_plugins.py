# mqtt_plugins.py
"""
    用法示例

    插到 MqttBus：

    from mqtt_bus import MqttBus
    from mqtt_plugins import MqttPluginManager, HeartbeatPlugin

    bus = MqttBus(host="127.0.0.1", port=1883, client_id="app_bus")
    bus.start()

    pm = MqttPluginManager(bus)
    pm.register(HeartbeatPlugin(status_topic=f"pipelines/{bus.client_id}/status", interval=15))
    pm.start()

    # ... 程序退出时：
    pm.stop()
    bus.stop()


    插到 MQTTClient：

    from pyengine.io.network.mqtt_client import MQTTClient
    from mqtt_plugins import MqttPluginManager, HeartbeatPlugin

    cli = MQTTClient(host="127.0.0.1", port=1883, client_id="standalone")
    cli.connect(timeout=10)  # 同步阻塞式连接(你当前实现)  # :contentReference[oaicite:2]{index=2}

    pm = MqttPluginManager(cli)
    pm.register(HeartbeatPlugin(status_topic=f"clients/{cli.client_id}/status"))
    pm.start()

    # ... 退出
    pm.stop()
    cli.disconnect()
"""

from typing import Protocol, Any, Callable


# --- Duck-typed Host 协议(Bus 或 Client 任一实现都可) ---
class IMqttHost(Protocol):
    client_id: str  # 可选，但建议有
    @property
    def is_connected(self) -> bool: ...
    def publish(self, topic: str, payload: bytes, qos: int = 0, retain: bool = False) -> bool: ...
    # 这些方法由 MqttBus 提供；若宿主是 MQTTClient 则需额外的扇出适配
    def subscribe(self, topic: str, handler: Callable[[str, bytes], None], qos: int = 0): ...
    def unsubscribe(self, handle: Any): ...

# --- 插件基类 ---
class MqttPlugin:
    def start(self, host: IMqttHost) -> None: ...
    def stop(self) -> None: ...


# --- 插件管理器：可托管多个插件，共用一个宿主 ---
class MqttPluginManager:
    def __init__(self, host: IMqttHost):
        self.host = host
        self._plugins: list[MqttPlugin] = []
        self._started = False

    def register(self, plugin: MqttPlugin):
        self._plugins.append(plugin)
        if self._started:
            plugin.start(self.host)

    def start(self):
        if self._started: return
        self._started = True
        for p in self._plugins:
            p.start(self.host)

    def stop(self):
        for p in self._plugins:
            p.stop()
        self._started = False
