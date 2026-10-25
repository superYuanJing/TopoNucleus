"""
TopoNucleus - 分析工具函数
"""

import pandas as pd
import numpy as np
import json
import os


def load_experiment(csv_path):
    """
    加载原始 CSV + 对应的元数据 JSON

    返回：
      df: DataFrame (列 = 通道名 + time_s)
      metadata: dict
    """
    # 加载 CSV
    df = pd.read_csv(csv_path)

    # 加载元数据
    meta_path = csv_path.replace('.csv', '_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return df, metadata


def find_shutdown_index(current, t, estimated_time=None):
    """
    找到放电关断的时刻索引

    策略：
      1. 如果给了 estimated_time，在其附近 ±50ms 找电流跌落点
      2. 否则，找电流从 >10% 最大值跌落到 <1% 的拐点

    参数：
      current: 放电电流数组
      t: 时间数组
      estimated_time: 估计的关断时间（秒）

    返回：
      idx: 关断时刻的索引
    """
    if estimated_time is not None:
        # 在估计时间附近 ±50ms 搜索
        window = 0.05
        mask = (t >= estimated_time - window) & (t <= estimated_time + window)
        if np.any(mask):
            # 在这个窗口内找电流的最小值（关断点）
            window_current = current[mask]
            # 找电流下降到 5% 最大值的点
            threshold = 0.05 * window_current.max()
            below = window_current < threshold
            if np.any(below):
                # 第一个低于阈值的点
                local_idx = np.argmax(below)
                # 转回全局索引
                global_mask_idx = np.where(mask)[0]
                idx = global_mask_idx[local_idx]
            else:
                idx = np.where(mask)[0][0]  # fallback: 窗口起点
        else:
            idx = np.argmin(np.abs(t - estimated_time))
    else:
        # 自动检测：找电流从 >10% 跌落到 <1% 的拐点
        max_current = current.max()
        threshold_high = 0.10 * max_current
        threshold_low = 0.01 * max_current

        # 找到所有低于 low 的点
        below_low = current < threshold_low
        if np.any(below_low):
            first_below = np.argmax(below_low)
            # 往前找最后一个高于 high 的点
            before = current[:first_below]
            if np.any(before > threshold_high):
                idx = np.argmax(before > threshold_high)
            else:
                idx = first_below
        else:
            idx = len(current) // 2  # fallback

    return idx


def butter_bandpass(data, lowcut, highcut, fs, order=4):
    """
    带通滤波（Butterworth）

    参数：
      data: 信号数组
      lowcut: 低频截止（Hz）
      highcut: 高频截止（Hz）
      fs: 采样率（Hz）
      order: 滤波器阶数

    返回：
      filtered: 滤波后的信号
    """
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if high >= 1.0:
        high = 0.99
    if low <= 0:
        low = 0.01

    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, data)
    return filtered


def detect_oscillation(data, fs, freq_range=(1000, 100000)):
    """
    检测信号中是否有显著振荡，返回主频和功率

    参数：
      data: 信号数组（建议用关断后的片段）
      fs: 采样率
      freq_range: 频率范围 (low, high) Hz

    返回：
      peak_freq: 主频（Hz）
      peak_power: 主频功率
      has_oscillation: bool
    """
    fft_vals = np.abs(np.fft.rfft(data - data.mean()))
    fft_freq = np.fft.rfftfreq(len(data), 1/fs)

    low, high = freq_range
    mask = (fft_freq >= low) & (fft_freq <= high)

    if np.any(mask):
        peak_idx = np.argmax(fft_vals[mask])
        peak_freq = fft_freq[mask][peak_idx]
        peak_power = fft_vals[mask][peak_idx]

        # 判断是否有显著振荡：峰值功率 > 均值 + 3σ
        band_power = fft_vals[mask]
        threshold = band_power.mean() + 3 * band_power.std()
        has_oscillation = peak_power > threshold
    else:
        peak_freq = 0
        peak_power = 0
        has_oscillation = False

    return peak_freq, peak_power, has_oscillation


def compute_hold_time(signal, t, threshold_sigma=3.0, pre_samples=None):
    """
    计算信号关断后的维持时间

    参数：
      signal: 信号数组
      t: 时间数组
      threshold_sigma: 阈值 = sigma × 这个值
      pre_samples: 用前 pre_samples 个样本算噪声，默认前 10%

    返回：
      hold_time: 维持时间（秒）
      hold_samples: 维持样本数
    """
    if pre_samples is None:
        pre_samples = max(100, len(signal) // 10)

    noise = signal[:pre_samples]
    sigma = noise.std()
    threshold = threshold_sigma * sigma

    above = np.abs(signal) > threshold
    if np.any(above):
        # 找到最后一个高于阈值的点
        last_above = np.argmax(above[::-1])
        if last_above == 0 and above[-1]:
            # 全都在阈值以上
            hold_samples = len(signal)
        else:
            hold_samples = len(signal) - last_above
    else:
        hold_samples = 0

    hold_time = hold_samples / (1.0 / (t[1] - t[0])) if len(t) > 1 else 0
    # 更准确的计算
    if hold_samples > 0:
        hold_time = t[-hold_samples] - t[0] if hold_samples < len(t) else t[-1] - t[0]
    else:
        hold_time = 0

    return hold_time, hold_samples


def fit_exponential_decay(t, y):
    """
    拟合单指数衰减 y = A * exp(-t/tau)

    返回：
      A, tau, r2
    """
    from scipy.optimize import curve_fit

    # 确保 y 为正
    y = np.clip(y - y.min() + 1e-6, 1e-6, None)

    def exp_decay(t, A, tau):
        return A * np.exp(-t / tau)

    try:
        popt, pcov = curve_fit(exp_decay, t, y, p0=[y[0], t[-1]/5], maxfev=2000)
        A, tau = popt

        # R²
        y_pred = exp_decay(t, A, tau)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return A, tau, r2
    except Exception:
        return 0, np.inf, 0
