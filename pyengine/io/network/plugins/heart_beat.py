import threading, time, json
from typing import Optional

from pyengine.io.network.mqtt_plugins import MqttPlugin, IMqttHost
from pyengine.utils.logger import logger


# --- 心跳插件：替代 HeartbeatThread，既可插 Bus 也可插 Client ---
class HeartbeatPlugin(MqttPlugin):
    def __init__(self,
                 topic: str,
                 interval: float = 15.0,
                 qos: int = 0,
                 retain: bool = True,
                 debug: bool = False):
        self.topic = topic
        self.interval = interval
        self.qos = qos
        self.retain = retain
        self.debug = debug
        self._host: Optional[IMqttHost] = None
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    def start(self, host: IMqttHost) -> None:
        self._host = host
        self._stop.clear()
        self._t = threading.Thread(target=self._run, name="HeartbeatPlugin", daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0)
        self._t = None
        self._host = None

    def _default_online_payload(self, host: IMqttHost) -> dict:
        return {"client_id": getattr(host, "client_id", "unknown"),
                "status": "online",
                "timestamp": int(time.time())}
    
    def _default_offline_payload(self, host: IMqttHost) -> dict:
        return {"client_id": getattr(host, "client_id", "unknown"),
                "status": "offline",
                "timestamp": int(time.time())}

    def _run(self):
        # 心跳检测主循环
        while not self._stop.is_set():
            host = self._host
            if host:
                # 即使断线也尝试 publish(宿主会返回 False/告警)，不阻塞心跳节奏
                payload_obj = self._default_online_payload(host)
                host.publish(self.topic,
                             json.dumps(payload_obj).encode("utf-8"),
                             qos=self.qos, retain=self.retain)
                if self.debug:
                    logger.debug("HeartbeatPlugin", f"Published heartbeat to {self.topic}: {payload_obj}")
            self._stop.wait(self.interval)
        # end-while

        # 发送离线状态
        if host:
            payload_obj = self._default_offline_payload(host)
            host.publish(self.topic,
                         json.dumps(payload_obj).encode("utf-8"),
                         qos=self.qos, retain=self.retain)
            if self.debug:
                logger.debug("HeartbeatPlugin", f"Published offline heartbeat to {self.topic}: {payload_obj}")
