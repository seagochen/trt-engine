from typing import Optional, Union
from urllib.parse import quote


def generate_rtsp_url(address: str,
                      port: Optional[Union[int, str]] = None,
                      path: Optional[str] = None,
                      username: Optional[str] = None,
                      password: Optional[str] = None) -> str:
    """
    构造 RTSP URL（面向 OpenCV 等客户端）。

    - 忽略 "none"/空串/空白：统一视为 None
    - 自动剥离传入的 rtsp:// 或 rtsps:// 前缀
    - IPv6 地址在带端口时自动加方括号
    - 用户名/密码做 URL 编码；路径保留斜杠

    返回: 形如 rtsp://user:pass@host:port/path 的字符串
    generate_rtsp_url("rtsp://192.168.1.10", 554, "stream1", "user", "p@ss:123")
    # rtsp://user:p%40ss%3A123@192.168.1.10:554/stream1

    generate_rtsp_url("fe80::1", "8554", "/live.sdp")
    # rtsp://[fe80::1]:8554/live.sdp

    generate_rtsp_url("192.168.1.10", None, None, "admin", None)
    # rtsp://admin@192.168.1.10
    """

    def _norm(v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = str(v).strip()
        return None if v == "" or v.lower() == "none" else v

    # --- 归一化输入 ---
    if not isinstance(address, str) or not address.strip():
        raise ValueError("address 不能为空")
    address = address.strip()

    # 去掉协议前缀
    lower = address.lower()
    if lower.startswith("rtsp://"):
        address = address[7:]
    elif lower.startswith("rtsps://"):
        address = address[8:]  # 这里只构造 rtsp://，如需 rtsps 可扩展

    path = _norm(path)
    username = _norm(username)
    password = _norm(password)

    # 端口处理：允许传 str，能转 int 则使用
    port_val: Optional[int] = None
    if port is not None:
        try:
            port_val = int(str(port).strip())
            if port_val <= 0 or port_val > 65535:
                port_val = None
        except (ValueError, TypeError):
            port_val = None

    # IPv6 包装（仅当显式带端口/需要加冒号时）
    # 判断：包含':' 且不含 '['（粗略判断是 IPv6 字面量）
    host = address
    needs_brackets = (":" in host) and not host.startswith("[")
    if needs_brackets and port_val is not None:
        host = f"[{host}]"

    # 用户认证（编码敏感字符）
    if username and password:
        auth = f"{quote(username, safe='')}:{quote(password, safe='')}@"
    elif username and not password:
        auth = f"{quote(username, safe='')}@"
    else:
        auth = ""

    # 端口段
    port_part = f":{port_val}" if port_val is not None else ""

    # 路径段（保留斜杠；编码空格等，避免破坏 URL）
    if path:
        # 去重前导斜杠，只保留一个
        path_clean = "/" + path.lstrip("/")
        path_part = quote(path_clean, safe="/:@$-_.+!*'(),")
    else:
        path_part = ""

    return f"rtsp://{auth}{host}{port_part}{path_part}"
