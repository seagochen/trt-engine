import numpy as np


def apply_sma_filter(history: np.ndarray, cx_col: int, cy_col: int, window_size: int = 5) -> np.ndarray:
    """
    对单个物体的历史轨迹(cx, cy)应用简单移动平均(SMA)滤波。
    使用卷积实现，效率很高。

    Args:
        history (np.ndarray): 单个track_id的所有历史数据。
        cx_col (int): cx列的索引。
        cy_col (int): cy列的索引。
        window_size (int): SMA的滑动窗口大小，必须为奇数。

    Returns:
        np.ndarray: cx, cy列被平滑后新的history数组。
    """
    if history.shape[0] < window_size:
        return history

    # 确保窗口大小为奇数，以保证窗口中心对齐
    if window_size % 2 == 0:
        window_size += 1

    smoothed_history = history.copy()

    # 创建平均权重
    weights = np.ones(window_size) / window_size

    # 对cx和cy坐标分别应用卷积滤波
    # mode='same' 确保输出数组和输入数组长度一致
    smoothed_history[:, cx_col] = np.convolve(history[:, cx_col], weights, mode='same')
    smoothed_history[:, cy_col] = np.convolve(history[:, cy_col], weights, mode='same')

    return smoothed_history


def apply_ema_filter(history: np.ndarray, cx_col: int, cy_col: int, alpha: float = 0.3) -> np.ndarray:
    """
    对单个物体的历史轨迹(cx, cy)应用指数滑动平均滤波。

    Args:
        history (np.ndarray): 单个track_id的所有历史数据。
        cx_col (int): cx列的索引。
        cy_col (int): cy列的索引。
        alpha (float): EMA的平滑因子，值越小越平滑。

    Returns:
        np.ndarray: cx, cy列被平滑后新的history数组。
    """
    if history.shape[0] < 2:
        return history

    # 复制一份数据以避免修改原始缓存
    smoothed_history = history.copy()

    # 初始化第一个平滑点
    smoothed_cx = smoothed_history[0, cx_col]
    smoothed_cy = smoothed_history[0, cy_col]

    for i in range(1, smoothed_history.shape[0]):
        # 应用EMA公式
        smoothed_cx = alpha * smoothed_history[i, cx_col] + (1 - alpha) * smoothed_cx
        smoothed_cy = alpha * smoothed_history[i, cy_col] + (1 - alpha) * smoothed_cy

        # 将平滑后的值写回新数组
        smoothed_history[i, cx_col] = smoothed_cx
        smoothed_history[i, cy_col] = smoothed_cy

    return smoothed_history