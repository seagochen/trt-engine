# --- demo plugins: hello world sender & receiver ---
import json, threading, socket
from typing import Optional

from pyengine.io.network.mqtt_plugins import IMqttHost, MqttPlugin
from pyengine.utils.logger import logger


def _get_local_ip() -> str:
    """更可靠地取本机出口IP(优先走UDP探测，失败回退到hostname解析)"""
    s = None
    ip = ""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 不需要真的连通，操作系统会选一块网卡给出本地地址
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            ip = "0.0.0.0"
    finally:
        try:
            s.close()
        except Exception:
            pass
    return ip

class HelloWorldSenderPlugin(MqttPlugin):
    """每 interval 秒发布一条 {"msg":"hello world","ip":"x.x.x.x"} 到指定 topic"""
    def __init__(self, topic: str, interval: float = 10.0, qos: int = 0, retain: bool = False):
        self.topic = topic
        self.interval = interval
        self.qos = qos
        self.retain = retain
        self._host: Optional[IMqttHost] = None
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None
        self._ip = _get_local_ip()

    def start(self, host: IMqttHost) -> None:
        self._host = host
        self._stop.clear()
        self._t = threading.Thread(target=self._run, name="HelloWorldSender", daemon=True)
        self._t.start()

    def _run(self):
        payload = {"msg": "hello world", "ip": self._ip}
        while not self._stop.is_set():
            try:
                if self._host:
                    self._host.publish(self.topic, json.dumps(payload).encode("utf-8"),
                                       qos=self.qos, retain=self.retain)
            except Exception:
                # 静默忽略，下一轮继续
                pass
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0)
        self._t = None
        self._host = None


class HelloWorldReceiverPlugin(MqttPlugin):
    """
    订阅指定 topic，若收到 {"msg":"hello world","ip":"..."} 则打印：
      received hello world from <ip>
    需要宿主支持 subscribe()/unsubscribe()(MqttBus 支持；直连 MQTTClient 需额外扇出器)
    """
    def __init__(self, topic: str):
        self.topic = topic
        self._host: Optional[IMqttHost] = None
        self._sub_handle = None

    def start(self, host: IMqttHost) -> None:
        self._host = host
        # MqttBus: subscribe(topic, handler) → 返回订阅句柄；MQTTClient 直连不支持多处理器
        if hasattr(host, "subscribe") and hasattr(host, "unsubscribe"):
            self._sub_handle = host.subscribe(self.topic, self._on_message)  # type: ignore[attr-defined]
        else:
            # 退而求其次：若是纯 MQTTClient，可在此处提示或抛出异常
            print("[HelloWorldReceiverPlugin] Host does not support per-topic subscribe(). Use MqttBus.")

    def _on_message(self, t: str, payload: bytes):
        try:
            obj = json.loads(payload.decode("utf-8"))
        except Exception:
            return
        if isinstance(obj, dict) and obj.get("msg") == "hello world":
            ip = obj.get("ip", "unknown")
            logger.debug("HelloWorldReceiverPlugin", f"received hello world from {ip}")

    def stop(self) -> None:
        if self._host and self._sub_handle and hasattr(self._host, "unsubscribe"):
            try:
                self._host.unsubscribe(self._sub_handle)  # type: ignore[attr-defined]
            except Exception:
                pass
        self._sub_handle = None
        self._host = None


"""
import time
from pyengine.io.network.mqtt_bus import MqttBus
from pyengine.io.network.mqtt_plugins import MqttPluginManager
from pyengine.io.network.plugins.hello_world import HelloWorldSenderPlugin, HelloWorldReceiverPlugin

# 1) 启动 Bus(单处创建，统一管理连接/重连/扇出)
bus = MqttBus(host="127.0.0.1", port=1883, client_id="demo_bus")
bus.start()  # 内含重连与订阅恢复逻辑，发布接口可被插件直接复用。:contentReference[oaicite:3]{index=3}

# 2) 装插件
pm = MqttPluginManager(bus)  # 多插件共享一个宿主(bus)运行。:contentReference[oaicite:4]{index=4}
topic = "demo/hello"

pm.register(HelloWorldSenderPlugin(topic=topic, interval=1))
pm.register(HelloWorldReceiverPlugin(topic=topic))

# 3) 启动插件
pm.start()

# 主线程阻塞30秒
while True:
    time.sleep(30)
    break  # 这里可以替换为其他业务逻辑

# ……此时每 1 秒会看到一条输出：received hello world from <你的IP> ……

# 4) 退出时
pm.stop(); bus.stop()
"""