"""
TopoNucleus - 数据采集模块 (v0.2)
========================
功能：
  - 真实模式（已安装 nidaqmx）：从 NI-DAQmx 任务采集 6 路信号，按放电电流阈值触发
  - 模拟模式（默认）：生成可复现的模拟数据，用于算法开发与 CI
  - 输出：data/raw/<experiment_id>.csv + <experiment_id>_meta.json

通道分配（与 analyzer 默认配置一致）：
  Dev1/ai0  bdot_x
  Dev1/ai1  bdot_y
  Dev1/ai2  bdot_z
  Dev1/ai3  photodiode
  Dev1/ai4  discharge_current
  Dev1/ai5  discharge_voltage

输出 CSV 列：time_s, bdot_x, bdot_y, bdot_z, photodiode, discharge_current, discharge_voltage
保存前降采样 decimate 倍（默认 10，100k -> 10k S/s）。

元数据（v0.2 关键）：写入进化引擎要学的 8 个参数 + 理论 tau_R，
使"实验参数 <-> 基因组"可以互相重建，避免进化引擎读到 NaN 当 0 产生幻觉。

作者: TopoNucleus 开源社区
许可: CERN OHL v2
"""

import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType
    HAS_DAQ = True
except ImportError:
    HAS_DAQ = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TopoNucleusDAQ:
    def __init__(self, config=None):
        self.config = {
            'sample_rate': 100_000,
            'duration': 2.0,
            'pre_trigger': 0.1,
            'decimate': 10,
            'channels': {
                'bdot_x': 'Dev1/ai0',
                'bdot_y': 'Dev1/ai1',
                'bdot_z': 'Dev1/ai2',
                'photodiode': 'Dev1/ai3',
                'discharge_current': 'Dev1/ai4',
                'discharge_voltage': 'Dev1/ai5',
            },
            'voltage_range': (-5.0, 5.0),
            'output_dir': os.path.join(BASE_DIR, 'data', 'raw'),
            # 触发：电流绝对值超过该阈值（A）即认定放电开始
            'trigger_threshold_a': 0.01,
            # 理论电阻扩散时间（ms），按 THEORY §7.1 估算后填入，供 analyzer 判据
            'tau_r_ms': None,
        }
        if config:
            self.config.update(config)

        self.metadata = {
            'experiment_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'timestamp': datetime.now().isoformat(),
            'operator': 'anonymous',
            'hardware_version': '0.2',
            'software_version': '0.2.0',
            'seed_geometry': 'unknown',
            'windings': 3,
            'twist': 0.5,
            'ring_diameter_cm': 3.0,
            'wire_diameter_mm': 1.5,
            'seed_material': 'unknown',
            'seed_surface': 'unknown',
            'symmetry_break': 'none',
            'gas': 'Ar',
            'pressure_mtorr': 0.0,
            'discharge_power_w': 0.0,
            'discharge_type': 'DC',
            'sample_rate_hz': self.config['sample_rate'],
            'decimate': self.config['decimate'],
            'saved_sample_rate_hz': self.config['sample_rate'] // max(1, self.config['decimate']),
            'tau_r_ms': self.config['tau_r_ms'],
            'notes': '',
        }

    def set_metadata(self, **kwargs):
        self.metadata.update(kwargs)
        return self

    # ---------------------------------------------------------------
    # 采集主入口
    # ---------------------------------------------------------------
    def acquire(self):
        if HAS_DAQ:
            df, shutdown_s = self._acquire_ni()
        else:
            df, shutdown_s = self._acquire_sim()
        self.metadata['shutdown_time_s'] = float(shutdown_s)
        return df, self.metadata

    def _acquire_ni(self):
        """真实 NI-DAQmx 采集：有限采集 + 软件触发（电流过阈值）"""
        fs = self.config['sample_rate']
        dur = self.config['duration']
        n = int(fs * dur)
        chans = self.config['channels']
        lo, hi = self.config['voltage_range']
        th = self.config['trigger_threshold_a']

        with nidaqmx.Task() as task:
            for name, line in chans.items():
                task.ai_channels.add_ai_voltage_chan(line, min_val=lo, max_val=hi)
            task.timing.cfg_samp_clk_timing(fs, samps_per_chan=n,
                                            sample_mode=AcquisitionType.FINITE)
            print('[DAQ] 开始采集（有限 %d 样点 @ %.0f S/s）...' % (n, fs))
            data = task.read(number_of_samples_per_channel=n, timeout=dur + 5.0)
        df = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in
                          zip(chans.keys(), data)})
        t = np.arange(len(df)) / fs
        df.insert(0, 'time_s', t)

        # 关断时刻：放电期间电流最后一点（过阈值）
        i = df['discharge_current'].abs().values
        above = i > th
        if above.any():
            last = int(np.where(above)[0][-1])
            shutdown_s = float(t[last])
        else:
            shutdown_s = float(t[len(t) // 2])
        return df, shutdown_s

    def _acquire_sim(self):
        """模拟数据：一段放电窗口内注入 Bdot 振荡 + 关断后缓慢包络，可复现。"""
        fs = self.config['sample_rate']
        dur = self.config['duration']
        n = int(fs * dur)
        t = np.arange(n) / fs
        rng = np.random.default_rng(int(self.metadata['experiment_id'][:8] or 1, 36))

        t0 = 0.5
        t1 = t0 + 1.0
        env = np.exp(-((t - t0) / 0.2) ** 2) * ((t >= t0) & (t <= t1)).astype(float)

        # 关断后缓慢包络：慢衰 tau_e 随 ring 变化（给 BO 一个可学的目标）
        #   tau_e(ms) = 20 + 8 * ring_cm  -> ring 越大包络越慢
        ring = float(self.metadata.get('ring_diameter_cm', 3.0))
        tau_e = (20.0 + 8.0 * ring) / 1000.0
        post = np.maximum(t - t1, 0.0)
        slow = np.exp(-post / tau_e)
        fast = np.exp(-post / 0.002)
        tail = 0.5 * slow + 0.5 * fast
        envelope = env + tail * (t > t1)

        b = {}
        for i, (k, f) in enumerate([('bdot_x', 4000), ('bdot_y', 6000), ('bdot_z', 8000)]):
            phase = rng.uniform(0, 2 * np.pi)
            b[k] = (0.6 / (i + 2)) * np.sin(2 * np.pi * f * t + phase) * envelope

        pd_sig = (0.5 * env + 0.15 * np.sin(2 * np.pi * 5000 * t) * (t > t1)
                  + 0.05 * rng.normal(size=n))
        cur = (0.1 * ((t >= t0) & (t <= t1)).astype(float)
               + 0.005 * rng.normal(size=n))
        vol = 5.0 * ((t >= t0) & (t <= t1)).astype(float) + 0.1 * rng.normal(size=n)

        df = pd.DataFrame({**b, 'photodiode': pd_sig,
                           'discharge_current': cur,
                           'discharge_voltage': vol})
        df.insert(0, 'time_s', t)
        shutdown_s = t1
        return df, shutdown_s

    # ---------------------------------------------------------------
    # 保存
    # ---------------------------------------------------------------
    def save(self, df, output_dir=None):
        out = output_dir or self.config['output_dir']
        os.makedirs(out, exist_ok=True)
        d = int(self.config['decimate'])
        if d > 1:
            df = df.iloc[::d].reset_index(drop=True)
        eid = str(self.metadata['experiment_id'])
        csv_p = os.path.join(out, eid + '.csv')
        meta_p = os.path.join(out, eid + '_meta.json')
        df.to_csv(csv_p, index=False, float_format='%.6f')
        # 数值归一化，避免 NaN 被进化引擎误读为 0
        meta = {k: (None if (v is None or
                             (isinstance(v, float) and not np.isfinite(v)))
                    else v)
                for k, v in self.metadata.items()}
        for c in ['ring_diameter_cm', 'wire_diameter_mm', 'twist',
                  'pressure_mtorr', 'discharge_power_w']:
            if meta.get(c) is None:
                meta[c] = float('nan')
        Path(meta_p).write_text(json.dumps(meta, indent=2))
        print('[DAQ] saved: %s  (%d rows, %.0f S/s)' %
              (csv_p, len(df), 1.0 / (df['time_s'].iloc[1] - df['time_s'].iloc[0])))
        return csv_p, meta_p


def parse_args():
    p = argparse.ArgumentParser(description='TopoNucleus 数据采集')
    p.add_argument('--seed', type=str, default='trefoil', help='seed_geometry')
    p.add_argument('--material', type=str, default='iron', help='seed_material')
    p.add_argument('--surface', type=str, default='bare', help='seed_surface')
    p.add_argument('--ring', type=float, default=3.0, help='ring_diameter_cm')
    p.add_argument('--wire', type=float, default=1.5, help='wire_diameter_mm')
    p.add_argument('--twist', type=float, default=0.5, help='twist (0~1)')
    p.add_argument('--sym', type=str, default='none', help='symmetry_break')
    p.add_argument('--pressure', type=float, default=20.0, help='pressure_mtorr')
    p.add_argument('--power', type=float, default=100.0, help='discharge_power_w')
    p.add_argument('--tau-r-ms', type=float, default=None, help='理论 tau_R (ms)')
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--duration', type=float, default=2.0)
    return p.parse_args()


def main():
    a = parse_args()
    out = a.output_dir or os.path.join(BASE_DIR, 'data', 'raw')
    d = TopoNucleusDAQ({'output_dir': out, 'duration': a.duration,
                        'tau_r_ms': a.tau_r_ms})
    d.set_metadata(
        seed_geometry=a.seed, seed_material=a.material,
        seed_surface=a.surface, ring_diameter_cm=a.ring,
        wire_diameter_mm=a.wire, twist=a.twist,
        symmetry_break=a.sym, pressure_mtorr=a.pressure,
        discharge_power_w=a.power,
    )
    df, meta = d.acquire()
    d.save(df)


if __name__ == '__main__':
    main()
