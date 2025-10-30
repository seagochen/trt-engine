import numpy as np


def apply_sma_filter_1d(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    对1D数组应用简单移动平均(SMA)滤波。
    改进边界处理：使用可变窗口大小，避免零填充。

    Args:
        data (np.ndarray): 1D数组数据。
        window_size (int): SMA的滑动窗口大小，必须为奇数。

    Returns:
        np.ndarray: 平滑后的1D数组。
    """
    if len(data) < window_size:
        return data.copy()

    # 确保窗口大小为奇数，以保证窗口中心对齐
    if window_size % 2 == 0:
        window_size += 1

    smoothed_data = np.empty_like(data, dtype=np.float64)
    half_window = window_size // 2

    for i in range(len(data)):
        # 动态调整窗口边界，确保不越界
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)

        # 计算当前窗口内的平均值
        smoothed_data[i] = np.mean(data[start:end])

    return smoothed_data


def apply_ema_filter_1d(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    对1D数组应用指数滑动平均滤波。

    EMA 公式: S_t = α * X_t + (1-α) * S_{t-1}
    其中 α 是平滑因子，值越小越平滑（通常 0.1-0.5）

    Args:
        data (np.ndarray): 1D数组数据。
        alpha (float): EMA的平滑因子，值越小越平滑。范围 [0, 1]。

    Returns:
        np.ndarray: 平滑后的1D数组。
    """
    if len(data) < 2:
        return data.copy()

    # 确保 alpha 在合理范围内
    alpha = np.clip(alpha, 0.0, 1.0)

    # 复制一份数据以避免修改原始数据
    smoothed_data = np.empty_like(data, dtype=np.float64)

    # 初始化第一个平滑点
    smoothed_data[0] = data[0]

    for i in range(1, len(data)):
        # 应用EMA公式
        smoothed_data[i] = alpha * data[i] + (1 - alpha) * smoothed_data[i - 1]

    return smoothed_data


def apply_sma_filter_1d_vectorized(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    对1D数组应用简单移动平均(SMA)滤波（向量化版本，更快）。
    使用 uniform_filter1d 实现，自动处理边界。

    Args:
        data (np.ndarray): 1D数组数据。
        window_size (int): SMA的滑动窗口大小。

    Returns:
        np.ndarray: 平滑后的1D数组。
    """
    from scipy.ndimage import uniform_filter1d

    if len(data) < window_size:
        return data.copy()

    # 确保窗口大小为奇数
    if window_size % 2 == 0:
        window_size += 1

    # mode='nearest' 使用最近的有效值填充边界，避免零填充
    smoothed_data = uniform_filter1d(data, size=window_size, mode='nearest')

    return smoothed_data


def apply_sma_filter_1d_pandas(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    对1D数组应用简单移动平均(SMA)滤波（Pandas版本）。
    使用 Pandas 的 rolling，可以更灵活地处理边界。

    Args:
        data (np.ndarray): 1D数组数据。
        window_size (int): SMA的滑动窗口大小。

    Returns:
        np.ndarray: 平滑后的1D数组。
    """
    import pandas as pd

    if len(data) < window_size:
        return data.copy()

    # 确保窗口大小为奇数
    if window_size % 2 == 0:
        window_size += 1

    # 使用 Pandas rolling，center=True 使窗口居中
    # min_periods=1 确保边界处也能计算（使用较小的窗口）
    series = pd.Series(data)
    smoothed_series = series.rolling(
        window=window_size,
        center=True,
        min_periods=1
    ).mean()

    return smoothed_series.to_numpy()


def apply_savgol_filter_1d(data: np.ndarray, window_size: int = 5, polyorder: int = 2) -> np.ndarray:
    """
    对1D数组应用 Savitzky-Golay 滤波（推荐用于轨迹平滑）。
    比简单移动平均更好地保留数据的局部特征（如转折点）。

    Args:
        data (np.ndarray): 1D数组数据。
        window_size (int): 滤波窗口大小，必须为奇数。
        polyorder (int): 多项式阶数，通常为 2 或 3。

    Returns:
        np.ndarray: 平滑后的1D数组。
    """
    from scipy.signal import savgol_filter

    if len(data) < window_size:
        return data.copy()

    # 确保窗口大小为奇数
    if window_size % 2 == 0:
        window_size += 1

    # polyorder 必须小于 window_size
    polyorder = min(polyorder, window_size - 1)

    # mode='nearest' 避免边界的零填充问题
    smoothed_data = savgol_filter(data, window_length=window_size, polyorder=polyorder, mode='nearest')

    return smoothed_data


# ============================================================================
# 测试和对比
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 创建测试数据（带噪声的正弦波）
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 100)
    clean_signal = np.sin(t)
    noisy_signal = clean_signal + np.random.normal(0, 0.3, len(t))

    # 应用不同的滤波方法
    window_size = 7

    sma_old = np.convolve(noisy_signal, np.ones(window_size) / window_size, mode='same')  # 旧方法
    sma_new = apply_sma_filter_1d(noisy_signal, window_size)  # 新方法（循环）
    sma_vectorized = apply_sma_filter_1d_vectorized(noisy_signal, window_size)  # 向量化
    sma_pandas = apply_sma_filter_1d_pandas(noisy_signal, window_size)  # Pandas
    ema = apply_ema_filter_1d(noisy_signal, alpha=0.3)
    savgol = apply_savgol_filter_1d(noisy_signal, window_size, polyorder=2)

    # 可视化对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Different Smoothing Methods Comparison', fontsize=16)

    # 1. 原始数据 vs 旧SMA（有边界问题）
    ax = axes[0, 0]
    ax.plot(t, noisy_signal, 'gray', alpha=0.5, label='Noisy', linewidth=1)
    ax.plot(t, clean_signal, 'g--', label='Clean', linewidth=2)
    ax.plot(t, sma_old, 'r-', label='SMA (old, with zero-padding)', linewidth=2)
    ax.set_title('Old SMA (Zero-Padding Problem)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 新SMA（循环版本）
    ax = axes[0, 1]
    ax.plot(t, noisy_signal, 'gray', alpha=0.5, label='Noisy', linewidth=1)
    ax.plot(t, clean_signal, 'g--', label='Clean', linewidth=2)
    ax.plot(t, sma_new, 'b-', label='SMA (new, variable window)', linewidth=2)
    ax.set_title('New SMA (Variable Window)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. SMA向量化版本
    ax = axes[0, 2]
    ax.plot(t, noisy_signal, 'gray', alpha=0.5, label='Noisy', linewidth=1)
    ax.plot(t, clean_signal, 'g--', label='Clean', linewidth=2)
    ax.plot(t, sma_vectorized, 'c-', label='SMA (scipy)', linewidth=2)
    ax.set_title('SMA (Scipy Vectorized)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Pandas版本
    ax = axes[1, 0]
    ax.plot(t, noisy_signal, 'gray', alpha=0.5, label='Noisy', linewidth=1)
    ax.plot(t, clean_signal, 'g--', label='Clean', linewidth=2)
    ax.plot(t, sma_pandas, 'm-', label='SMA (pandas)', linewidth=2)
    ax.set_title('SMA (Pandas Rolling)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. EMA
    ax = axes[1, 1]
    ax.plot(t, noisy_signal, 'gray', alpha=0.5, label='Noisy', linewidth=1)
    ax.plot(t, clean_signal, 'g--', label='Clean', linewidth=2)
    ax.plot(t, ema, 'orange', label='EMA (α=0.3)', linewidth=2)
    ax.set_title('Exponential Moving Average')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Savitzky-Golay
    ax = axes[1, 2]
    ax.plot(t, noisy_signal, 'gray', alpha=0.5, label='Noisy', linewidth=1)
    ax.plot(t, clean_signal, 'g--', label='Clean', linewidth=2)
    ax.plot(t, savgol, 'purple', label='Savitzky-Golay', linewidth=2)
    ax.set_title('Savitzky-Golay Filter')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('smoothing_comparison.png', dpi=150)
    print("✅ 图表已保存为 'smoothing_comparison.png'")

    # 打印边界处的误差对比
    print("\n" + "=" * 60)
    print("边界误差分析（前5个点）")
    print("=" * 60)
    print(f"{'Method':<20} {'Point 0':<12} {'Point 1':<12} {'Point 2':<12}")
    print("-" * 60)

    for name, filtered in [
        ('Clean Signal', clean_signal),
        ('Noisy Signal', noisy_signal),
        ('SMA (old)', sma_old),
        ('SMA (new)', sma_new),
        ('SMA (scipy)', sma_vectorized),
        ('SMA (pandas)', sma_pandas),
        ('EMA', ema),
        ('Savitzky-Golay', savgol),
    ]:
        print(f"{name:<20} {filtered[0]:>11.4f} {filtered[1]:>11.4f} {filtered[2]:>11.4f}")

    # 计算边界误差（与干净信号对比）
    print("\n" + "=" * 60)
    print("边界绝对误差（与干净信号对比）")
    print("=" * 60)

    for name, filtered in [
        ('SMA (old)', sma_old),
        ('SMA (new)', sma_new),
        ('SMA (scipy)', sma_vectorized),
        ('SMA (pandas)', sma_pandas),
        ('EMA', ema),
        ('Savitzky-Golay', savgol),
    ]:
        errors = np.abs(filtered[:5] - clean_signal[:5])
        print(f"{name:<20} 前5点平均误差: {np.mean(errors):.4f}")