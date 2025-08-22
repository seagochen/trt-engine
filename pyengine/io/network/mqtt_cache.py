import threading
import time
from typing import Any, Optional, Dict, List


class MQTTCacheEntry:
    """
    表示缓存中的一条数据，数据以字典形式存储，可以包含图像、推论、以及其他任意格式的数据。
    """
    def __init__(self, index: int, data: Optional[Dict[str, Any]] = None):
        self.index = index
        self.data: Dict[str, Any] = {}  # 存储任意数据，键为数据类型标识
        self.timestamp: float = time.time()  # 记录数据添加的时间

        if data is not None:
            self.data.update(data)

    def add_data(self, key: str, value: Any) -> None:
        """
        添加一种数据到缓存条目中
        """
        self.data[key] = value
        self.timestamp = time.time()  # 更新数据时间戳

    def is_complete(self, required_keys: Optional[List[str]] = None) -> bool:
        """
        判断当前条目是否完整。
        如果提供了 required_keys，则认为只有当所有必备数据均存在时才完整；
        如果未指定 required_keys，则只要数据字典非空就认为条目完整。
        """
        if required_keys is None:
            return bool(self.data)
        return all(key in self.data for key in required_keys)


class MQTTCache:
    """
    MQTT缓存，负责按帧号存储任意数据，并提供添加、获取、清理等接口。
    你可以通过配置 required_keys 来定义判断一条数据完整的标准。
    """
    def __init__(self, timeout: float = 10.0, required_keys: Optional[List[str]] = None):
        self.cache: Dict[int, MQTTCacheEntry] = {}
        self.timeout: float = timeout
        self.required_keys: Optional[List[str]] = required_keys
        self.lock = threading.Lock()

    def add_data(self, index: int, key: str, data: Any) -> None:
        """
        添加任意类型的数据到缓存中
        :param index: 数据对应的编号
        :param key: 数据类型标识(例如 'image', 'inference', 'lidar', ...)
        :param data: 数据内容
        """
        with self.lock:
            if index not in self.cache:
                self.cache[index] = MQTTCacheEntry(index)
            self.cache[index].add_data(key, data)
            self._cleanup_locked()

    def _cleanup_locked(self) -> None:
        """
        清理超过超时限制的条目
        """
        current_time = time.time()
        keys_to_delete = [idx for idx, entry in self.cache.items()
                          if (current_time - entry.timestamp) > self.timeout]
        for idx in keys_to_delete:
            del self.cache[idx]

    def cleanup(self) -> None:
        """
        提供一个外部调用的清理接口
        """
        with self.lock:
            self._cleanup_locked()

    def count(self) -> int:
        """
        返回当前缓存中完整数据条目的数目
        """
        with self.lock:
            self._cleanup_locked()
            return sum(1 for entry in self.cache.values() if entry.is_complete(self.required_keys))

    def get_min_index(self) -> Optional[int]:
        """
        返回缓存中最小的完整数据的index
        """
        with self.lock:
            self._cleanup_locked()
            sorted_keys = sorted(self.cache.keys())
            for key in sorted_keys:
                if self.cache[key].is_complete(self.required_keys):
                    return key
            return None

    def get(self, index: int) -> Optional[MQTTCacheEntry]:
        """
        获取指定 index 的完整数据，并从缓存中删除该条数据
        """
        with self.lock:
            entry = self.cache.get(index)
            if entry and entry.is_complete(self.required_keys):
                del self.cache[index]
                return entry
            return None

    def release(self) -> None:
        """
        释放所有缓存数据
        """
        with self.lock:
            self.cache.clear()
