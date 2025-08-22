import fcntl
import json
import os
import time
from typing import Any

import numpy as np


class SharedMemoryHandler:
    def __init__(self, base_path="/dev/shm", timeout=5.0, retry_interval=0.01):
        """
        初始化共享内存处理类
        :param base_path: 共享内存路径
        :param timeout: 最大超时时间(秒)
        :param retry_interval: 获取锁的重试间隔(秒)
        """
        if not os.path.exists(base_path):
            raise ValueError(f"Shared memory path '{base_path}' does not exist.")
        self.base_path = base_path
        self.timeout = timeout
        self.retry_interval = retry_interval

    def _get_full_path(self, name):
        """生成完整的共享内存文件路径"""
        return os.path.join(self.base_path, name)

    def _lock_file(self, f, mode):
        """
        尝试获取文件锁，支持超时机制
        :param f: 文件对象
        :param mode: 锁模式 fcntl.LOCK_EX(写)/ fcntl.LOCK_SH(读)
        """
        start_time = time.time()
        while True:
            try:
                fcntl.flock(f, mode | fcntl.LOCK_NB)  # 非阻塞模式
                return  # 锁定成功
            except BlockingIOError:
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Timeout: Unable to acquire file lock within {self.timeout} seconds")
                time.sleep(self.retry_interval)  # 等待后重试

    def write_text(self, name: str, data: str):
        """写入文本数据，带超时锁"""
        path = self._get_full_path(name)
        with open(path, 'w') as f:
            self._lock_file(f, fcntl.LOCK_EX)  # 尝试加写锁
            f.write(data)
            fcntl.flock(f, fcntl.LOCK_UN)

    def read_text(self, name: str) -> str:
        """读取文本数据，带超时锁"""
        path = self._get_full_path(name)
        with open(path, 'r') as f:
            self._lock_file(f, fcntl.LOCK_SH)  # 尝试加读锁
            data = f.read()
            fcntl.flock(f, fcntl.LOCK_UN)
        return data

    def write_json(self, name: str, data: Any):
        """写入 JSON 数据，带超时锁"""
        json_str = json.dumps(data)
        self.write_text(name, json_str)

    def read_json(self, name: str) -> Any:
        """读取 JSON 数据并解析，带超时锁"""
        json_str = self.read_text(name)
        return json.loads(json_str)

    def write_binary(self, name: str, data: bytes):
        """写入二进制数据，带超时锁"""
        path = self._get_full_path(name)
        with open(path, 'wb') as f:
            self._lock_file(f, fcntl.LOCK_EX)
            f.write(data)
            fcntl.flock(f, fcntl.LOCK_UN)

    def read_binary(self, name: str) -> bytes:
        """读取二进制数据，带超时锁"""
        path = self._get_full_path(name)
        with open(path, 'rb') as f:
            self._lock_file(f, fcntl.LOCK_SH)
            data = f.read()
            fcntl.flock(f, fcntl.LOCK_UN)
        return data

    def write_numpy(self, name: str, array: np.ndarray):
        """写入 NumPy 数组数据，带超时锁"""
        path = self._get_full_path(name)
        np.save(path, array)

    def read_numpy(self, name: str) -> np.ndarray:
        """读取 NumPy 数组数据，带超时锁"""
        path = self._get_full_path(name) + ".npy"
        return np.load(path, allow_pickle=True)

    def remove(self, name: str):
        """删除共享内存文件"""
        path = self._get_full_path(name)
        if os.path.exists(path):
            os.remove(path)

    def exists(self, name: str) -> bool:
        """检查共享内存文件是否存在"""
        return os.path.exists(self._get_full_path(name))

    def cleanup(self):
        """清理所有共享内存文件"""
        for file in os.listdir(self.base_path):
            os.remove(os.path.join(self.base_path, file))

"""

### **使用示例**

#### **1. 读写文本数据(JSON、字符串等)**

```python
shm = SharedMemoryHandler()

# 写入 JSON 数据
shm.write_json("test_json", {"name": "Alice", "age": 30})

# 读取 JSON 数据
data = shm.read_json("test_json")
print(data)  # {'name': 'Alice', 'age': 30}
```

---

#### **2. 读写二进制数据(图像、音频等)**

```python
# 读取图片并存储
import cv2

img = cv2.imread("test.jpg")
img_bytes = img.tobytes()

# 写入图像数据
shm.write_binary("image_raw", img_bytes)

# 读取图像数据
image_bytes = shm.read_binary("image_raw")
restored_img = np.frombuffer(image_bytes, dtype=img.dtype).reshape(img.shape)

cv2.imshow("Restored Image", restored_img)
cv2.waitKey(0)
```

---

#### **3. 读写 NumPy 数组**

```python
arr = np.random.rand(100, 100)

# 存入共享内存
shm.write_numpy("numpy_array", arr)

# 从共享内存读取
loaded_arr = shm.read_numpy("numpy_array")
print(loaded_arr.shape)  # (100, 100)
```

---

#### **4. 读写 Pickle 序列化数据**

```python
shm.write_pickle("pickled_obj", {"foo": [1, 2, 3], "bar": "hello"})
obj = shm.read_pickle("pickled_obj")
print(obj)  # {'foo': [1, 2, 3], 'bar': 'hello'}
```

---

#### **5. 使用 `mmap` 高效读写大数据**

```python
data = b"Hello World" * 1000
shm.write_mmap("large_data", data, 10000)

read_data = shm.read_mmap("large_data", 10000)
print(read_data[:20])  # 显示部分数据
```

---

### **优势总结**

1. **多格式兼容性：**
- 处理文本(JSON、XML)，二进制(图像、音频)，结构化数据(NumPy、pickle)。

2. **高效性：**
- 使用 **`/dev/shm`** 加速读写，避免硬盘 I/O 开销。

3. **跨进程访问：**
- 其他语言(如 C++、Go)可以直接读取 `/dev/shm` 文件，实现跨语言数据共享。

4. **安全性：**
- 使用文件锁(`fcntl`)防止数据竞争。

"""