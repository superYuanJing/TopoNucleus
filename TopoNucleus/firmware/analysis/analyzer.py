"""
TopoNucleus - 信号分析模块 (v0.2)
========================
功能：
  - 从 raw CSV 提取特征
  - 判断"平庸放电" vs "拓扑涌现候选"
  - 用 bootstrap 估计判据的假阳性率
  - 计算适应度 F
  - 输出 processed/ 下的特征文件

判据定义（Phase 0，v0.2 修订）：
  平庸放电（参考：普通 spheromak 关断后磁能 ~10ms 量级衰减，
  纯电阻扩散 tau_R 可估算；故 25ms 阈值并不等于"已证明自持"，
  只是进入"值得复核"区间）：
  1. 关断后 B 包络衰减时间常数 tau_e > 25 ms
  2. 且 tau_e 显著大于理论 tau_R（tau_e > 3 * tau_R）
  3. 且非指数检验 p < 0.05（bootstrap，校准过，见 THEORY §7）
  以上三条同时满足 => bdot_candidate（磁场候选）

  独立加分项（不用于判候选，避免误报）：
  - 发光关断后持续 > 2ms 且 > 2× 关断前基线
  - 阻抗关断前后比值 > 2 或 < 0.5

  综合候选 = 磁场候选 AND 至少一项加分项，以减少误报。

适应度：
  F = tau_e(ms) * V_pp / E_inj(J)   （无量纲参考量，非效率）

注意：本模块只负责统计判据，不宣称任何物理结论。
物理结论需要人工复核 + 重复实验。

作者: TopoNNucleus 开源社区
许可: CERN OHL v2
"""

import numpy as np
import pandas as pd
import json
import os
import argparse
import warnings
from pathlib import Path
from scipy import signal as sp_signal
from scipy.optimize import curve_fit, OptimizeWarning

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils import load_experiment, find_shutdown_index


class TopoNucleusAnalyzer:
    """拓扑晶核信号分析器"""

    def __init__(self, config=None):
        self.config = {
            'sample_rate': 10_000,        # 与 DAQ 降采样后的实际采样率一致
            'shutdown_time': 1.5,         # 放电关断时间（秒），默认由 DAQ 写入元数据
            'tau_e_threshold_ms': 25.0,   # 包络衰减时间常数下限（ms）
            'tau_ratio_threshold': 3.0,   # tau_e / tau_R 下限
            'bootstrap_n': 2000,          # bootstrap 次数
            'bootstrap_alpha': 0.05,      # 非指数检验显著性水平
            'bdot_channels': ['bdot_x', 'bdot_y', 'bdot_z'],
            'photodiode_channel': 'photodiode',
            'current_channel': 'discharge_current',
            'voltage_channel': 'discharge_voltage',
            'tau_r_ms': None,             # 理论 tau_R(ms)，为 None 时跳过比值判据
            'output_dir': os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'data', 'processed'),
        }
        if config:
            self.config.update(config)

        self.df = None
        self.metadata = {}
        self.features = {}
        self.tau_r_ms = None

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------
    def load(self, csv_path):
        """加载原始数据 + 元数据"""
        self.df, self.metadata = load_experiment(csv_path)
        # 元数据里若有关断时刻，覆盖默认
        if 'shutdown_time_s' in self.metadata:
            try:
                self.config['shutdown_time'] = float(self.metadata['shutdown_time_s'])
            except (TypeError, ValueError):
                pass
        if 'tau_r_ms' in self.metadata:
            try:
                self.tau_r_ms = float(self.metadata['tau_r_ms'])
            except (TypeError, ValueError):
                self.tau_r_ms = self.config['tau_r_ms']
        else:
            self.tau_r_ms = self.config['tau_r_ms']
        return self

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def analyze(self):
        t = self.df['time_s'].values
        idx = find_shutdown_index(
            self.df[self.config['current_channel']].values,
            t, self.config['shutdown_time'])
        self.features['shutdown_time'] = float(t[idx])
        self.features['shutdown_index'] = int(idx)

        self._analyze_bdot(idx)
        self._analyze_photodiode(idx)
        self._analyze_impedance(idx)
        self._compute_fitness()
        self._classify()
        return self

    # ------------------------------------------------------------------
    # Bdot：包络、指数拟合、bootstrap 非指数检验
    # ------------------------------------------------------------------
    def _analyze_bdot(self, idx):
        t = self.df['time_s'].values
        post_t = t[idx:] - t[idx]
        fs = 1.0 / (t[1] - t[0]) if len(t) > 1 else self.config['sample_rate']
        thr_tau = self.config['tau_e_threshold_ms'] / 1000.0
        results = {}

        for ch in self.config['bdot_channels']:
            y = self.df[ch].values.astype(float)
            pre = y[:idx]
            post = y[idx:]
            if len(pre) < 10:
                pre = y[max(0, int(len(y) * 0.4)):idx]

            # 关断后持续超过阈值的时间（3-sigma 噪声）
            sigma = max(float(pre.std()), 1e-9)
            thr = 3.0 * sigma
            above = np.abs(post) > thr
            if above.any():
                last = int(np.where(above)[0][-1])
                hold_ms = float(post_t[last]) * 1000.0
            else:
                hold_ms = 0.0

            # 包络（希尔伯特）+ 指数拟合
            env = _envelope(post, fs)
            tau_e = _fit_tau(post_t, env)

            # 与理论 tau_R 比较
            if self.tau_r_ms is not None and self.tau_r_ms > 0:
                tau_ratio = tau_e / self.tau_r_ms
                faster_than_diffusion = tau_ratio > self.config['tau_ratio_threshold']
            else:
                tau_ratio = None
                faster_than_diffusion = None

            # bootstrap 非指数检验（单尾）：
            #   H0: 残差与 exp 拟合不可区分；p 小 => 拒绝 H0
            p_ne = _bootstrap_non_exponential(
                post_t, env, n_boot=self.config['bootstrap_n'],
                alpha=self.config['bootstrap_alpha'])

            vpp = float(post.max() - post.min()) if len(post) else 0.0

            results[ch] = {
                'hold_time_ms': hold_ms,
                'envelope_tau_ms': tau_e,
                'tau_ratio_to_tauR': tau_ratio,
                'faster_than_diffusion': faster_than_diffusion,
                'non_exponential_p': p_ne,
                'is_non_exponential': bool(p_ne < self.config['bootstrap_alpha']),
                'v_pp': vpp,
                'noise_sigma': sigma,
                'long_decay': bool(tau_e > thr_tau),
            }

        # 取 tau 最大的通道作为代表
        best = max(results.values(), key=lambda r: r['envelope_tau_ms'])
        tau_e_max = best['envelope_tau_ms']
        any_long = any(r['long_decay'] for r in results.values())
        any_ne = any(r['is_non_exponential'] for r in results.values())
        any_fast = any(r['faster_than_diffusion'] is True for r in results.values())

        self.features['bdot'] = {
            'channels': results,
            'tau_e_ms_max': tau_e_max,
            'any_long_decay': any_long,
            'any_non_exponential': any_ne,
            'any_faster_than_diffusion': any_fast,
            'tau_r_ms': self.tau_r_ms,
        }

    # ------------------------------------------------------------------
    # 光电：修复基线（用关断前 20ms 窗口，避免放电期高基线）
    # ------------------------------------------------------------------
    def _analyze_photodiode(self, idx):
        t = self.df['time_s'].values
        ch = self.config['photodiode_channel']
        y = self.df[ch].values.astype(float)
        post = y[idx:]
        post_t = t[idx:] - t[idx]

        # 关断前基线：取 [t_off-20ms, t_off] 窗口均值
        win = (t >= t[idx] - 0.02) & (t < t[idx])
        pre_base = float(y[win].mean()) if win.any() else float(y[:idx].mean())
        pre_std = float(y[win].std()) if win.any() else max(float(y[:idx].std()), 1e-9)
        post_base = float(post.mean())

        # 持续：> 2ms 且 > 2× 基线（且至少基线+3σ）
        min_sustain = 0.002
        sustained = (len(post_t) > 0 and post_t[-1] > min_sustain and
                     post_base > 2.0 * pre_base and
                     post_base > pre_base + 3.0 * pre_std)

        # 振荡主频（1~50 kHz）
        fq, pw, has_osc = 0.0, 0.0, False
        if len(post) > 64:
            fs = 1.0 / (t[1] - t[0])
            f = np.fft.rfft(post - post.mean())
            fr = np.fft.rfftfreq(len(post), 1.0 / fs)
            m = (fr >= 1000.0) & (fr <= 50000.0)
            if m.any():
                band = np.abs(f[m])
                p_thr = band.mean() + 3.0 * band.std()
                j = int(np.argmax(band))
                if band[j] > p_thr:
                    fq, pw, has_osc = float(fr[m][j]), float(band[j]), True

        self.features['photodiode'] = {
            'pre_baseline': pre_base,
            'post_baseline': post_base,
            'ratio': post_base / pre_base if pre_base > 0 else float('inf'),
            'pre_std': pre_std,
            'sustained': bool(sustained),
            'hold_ms': float(post_t[-1]) * 1000.0 if len(post_t) else 0.0,
            'peak_freq_hz': fq,
            'peak_power': pw,
            'has_oscillation': has_osc,
        }

    # ------------------------------------------------------------------
    # 阻抗：关断前后比值（加分项）
    # ------------------------------------------------------------------
    def _analyze_impedance(self, idx):
        v = self.df[self.config['voltage_channel']].values.astype(float)
        i = self.df[self.config['current_channel']].values.astype(float)
        z = np.where(np.abs(i) > 1e-6, np.abs(v / i), np.nan)
        z_pre = np.nanmean(z[:idx])
        z_post = np.nanmean(z[idx:])
        if not np.isfinite(z_pre) or z_pre <= 0:
            z_pre = np.nan
        if not np.isfinite(z_post):
            z_post = np.nan
        ratio = z_post / z_pre if np.isfinite(z_pre) and np.isfinite(z_post) and z_pre != 0 else None
        has_jump = ratio is not None and (ratio > 2.0 or ratio < 0.5)

        self.features['impedance'] = {
            'z_pre': None if not np.isfinite(z_pre) else float(z_pre),
            'z_post': None if not np.isfinite(z_post) else float(z_post),
            'ratio': None if ratio is None or not np.isfinite(ratio) else float(ratio),
            'has_jump': bool(has_jump),
        }

    # ------------------------------------------------------------------
    # 适应度
    # ------------------------------------------------------------------
    def _compute_fitness(self):
        t_hold = self.features['bdot']['tau_e_ms_max']
        vpp = max((r['v_pp'] for r in self.features['bdot']['channels'].values()),
                  default=0.0)
        t = self.df['time_s'].values
        v = self.df[self.config['voltage_channel']].values.astype(float)
        i = self.df[self.config['current_channel']].values.astype(float)
        dt = float(t[1] - t[0]) if len(t) > 1 else 1.0 / self.config['sample_rate']
        m = np.abs(i) > 0.01 * (np.abs(i).max() or 1.0)
        e_inj = float(np.trapz((v * i)[m], dx=dt)) if m.any() else 0.0
        if not np.isfinite(e_inj) or e_inj <= 0:
            e_inj = 1.0

        self.features['fitness'] = {
            't_hold_ms': t_hold,
            'v_pp': vpp,
            'e_inj_j': e_inj,
            'F': t_hold * vpp / e_inj,
        }

    # ------------------------------------------------------------------
    # 分类（分层，避免单一弱判据直接判候选）
    # ------------------------------------------------------------------
    def _classify(self):
        b = self.features['bdot']
        pd_ok = self.features['photodiode']['sustained']
        z_ok = self.features['impedance']['has_jump']

        long_ok = b['any_long_decay']
        ne_ok = b['any_non_exponential']
        fd_ok = b['any_faster_than_diffusion']
        if self.tau_r_ms is None:
            # 无理论值时，用"长衰减 + 非指数"替代"快于扩散"
            field_candidate = long_ok and ne_ok
        else:
            field_candidate = long_ok and ne_ok and fd_ok

        self.features['classification'] = {
            'field_candidate': field_candidate,           # 磁场候选
            '加分项_发光持续': pd_ok,
            '加分项_阻抗跳变': z_ok,
            '综合候选': field_candidate and (pd_ok or z_ok),
            '判据明细': {
                'tau_e>25ms': long_ok,
                '非指数(p<0.05)': ne_ok,
                'tau_e>3*tau_R': fd_ok,
                '发光持续': pd_ok,
                '阻抗跳变': z_ok,
            },
        }

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------
    def is_candidate(self):
        return bool(self.features['classification']['综合候选'])

    def is_field_candidate(self):
        return bool(self.features['classification']['field_candidate'])

    def summary(self):
        lines = []
        lines.append('=' * 58)
        lines.append(f"实验 ID : {self.metadata.get('experiment_id', 'unknown')}")
        lines.append(f"种子    : {self.metadata.get('seed_geometry', '?')} "
                     f"({self.metadata.get('seed_material', '?')}, "
                     f"{self.metadata.get('seed_surface', '?')})")
        lines.append(f"参数    : {self.metadata.get('gas', '?')} @ "
                     f"{self.metadata.get('pressure_mtorr', '?')} mTorr, "
                     f"{self.metadata.get('discharge_power_w', '?')} W")
        if self.tau_r_ms:
            lines.append(f"理论tau_R : {self.tau_r_ms:.2f} ms")
        lines.append('-' * 58)

        for ch, r in self.features['bdot']['channels'].items():
            lines.append(f"[Bdot:{ch}] tau_e={r['envelope_tau_ms']:.2f}ms, "
                         f"hold={r['hold_time_ms']:.2f}ms, "
                         f"non_exp p={r['non_exponential_p']:.3f}, "
                         f"Vpp={r['v_pp']:.4f}V")
        b = self.features['bdot']
        lines.append(f"[Bdot] tau_e 最大 = {b['tau_e_ms_max']:.2f} ms; "
                     f"长衰减={b['any_long_decay']}; "
                     f"非指数={b['any_non_exponential']}; "
                     f"快于扩散={b['any_faster_than_diffusion']}")

        p = self.features['photodiode']
        lines.append(f"[PD] 关断前基线={p['pre_baseline']:.5f}, "
                     f"关断后={p['post_baseline']:.5f}, 比值={p['ratio']:.2f}, "
                     f"持续={p['sustained']}")
        lines.append(f"[PD] 振荡主频={p['peak_freq_hz']:.0f} Hz, "
                     f"有振荡={p['has_oscillation']}")

        z = self.features['impedance']
        lines.append(f"[Z] 关断前={z['z_pre']}, 关断后={z['z_post']}, "
                     f"比值={z['ratio']}, 跳变={z['has_jump']}")

        f = self.features['fitness']
        lines.append(f"[FIT] t_hold={f['t_hold_ms']:.2f}ms, Vpp={f['v_pp']:.4f}V, "
                     f"E_inj={f['e_inj_j']:.4f}J, F={f['F']:.6f}")

        c = self.features['classification']
        lines.append('-' * 58)
        lines.append(f"磁场候选 : {'是' if c['field_candidate'] else '否'}  "
                     f"(tau>25ms & 非指数{' & tau>3tau_R' if self.tau_r_ms else ''})")
        lines.append(f"加分     : 发光持续={c['加分项_发光持续']}, "
                     f"阻抗跳变={c['加分项_阻抗跳变']}")
        lines.append(f"★★★ 综合候选 : {'是' if c['综合候选'] else '否'} ★★★")
        lines.append('=' * 58)
        return '\n'.join(lines)

    def save(self, output_dir=None):
        if output_dir:
            self.config['output_dir'] = output_dir
        os.makedirs(self.config['output_dir'], exist_ok=True)
        exp_id = str(self.metadata.get('experiment_id', 'unknown'))
        out = os.path.join(self.config['output_dir'], f'{exp_id}_features.json')

        def convert(obj):
            if isinstance(obj, (np.floating, float)):
                return None if not np.isfinite(obj) else float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return convert(obj.tolist())
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        payload = {
            'metadata': self.metadata,
            'features': convert(self.features),
            'is_field_candidate': self.is_field_candidate(),
            'is_candidate': self.is_candidate(),
            'software_version': '0.2.0',
        }
        with open(out, 'w') as fp:
            json.dump(payload, fp, indent=2)
        print(f"[INFO] Features saved: {out}")
        return out


# ----------------------------------------------------------------------
# 工具函数（模块级）
# ----------------------------------------------------------------------
def _envelope(y, fs):
    """希尔伯特包络，并对 < 100 Hz 做低通，避免工频/慢漂干扰"""
    try:
        env = np.abs(sp_signal.hilbert(y))
    except Exception:
        env = np.abs(y)
    env = env - np.min(env)
    nyq = 0.5 * fs
    if nyq > 120.0:
        try:
            b, a = sp_signal.butter(4, min(100.0, nyq * 0.9) / nyq, btype='low')
            env = sp_signal.filtfilt(b, a, env)
        except Exception:
            pass
    env = np.clip(env, 0.0, None)
    return env


def _fit_tau(t, env):
    """对包络拟合 exp 衰减，返回 tau(ms)；失败返回 inf"""
    if len(env) < 10 or not np.any(env > 0):
        return float('inf')
    e = env - env.min() + 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', OptimizeWarning)
        try:
            popt, _ = curve_fit(
                lambda tt, a, tau: a * np.exp(-tt / tau),
                t, e, p0=[e[0], t[-1] / 5.0],
                bounds=([0.0, 1e-6], [np.inf, np.inf]), maxfev=3000)
            tau_s = float(popt[1])
        except Exception:
            return float('inf')
    if not np.isfinite(tau_s) or tau_s <= 0:
        return float('inf')
    return tau_s * 1000.0


def _bootstrap_non_exponential(t, env, n_boot=2000, alpha=0.05):
    """单尾 bootstrap 检验：真实残差是否显著大于纯指数噪声。
    返回 p 值（越小越支持非指数）。"""
    if len(env) < 10:
        return 1.0
    e = env - env.min() + 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', OptimizeWarning)
        try:
            popt, _ = curve_fit(lambda tt, a, tau: a * np.exp(-tt / tau),
                                t, e, p0=[e[0], t[-1] / 5.0],
                                bounds=([0.0, 1e-6], [np.inf, np.inf]),
                                maxfev=3000)
            fit = popt[0] * np.exp(-t / popt[1])
        except Exception:
            return 1.0
    resid = env - fit
    if not np.any(np.isfinite(resid)) or len(resid) < 10:
        return 1.0
    r = resid - resid.mean()
    real_stat = float(np.abs(r).mean())
    boot = np.random.default_rng().normal(0.0, max(float(np.std(r)), 1e-12),
                                          size=(n_boot, len(r)))
    boot_stat = boot.std(axis=1) * np.sqrt(2.0 / np.pi)  # 高斯 abs 均值近似
    p = float((boot_stat >= real_stat).mean())
    return min(p, 1.0)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='TopoNucleus 信号分析工具')
    p.add_argument('csv_path', type=str, help='原始 CSV 文件路径')
    p.add_argument('--output-dir', type=str, default=None, help='输出目录')
    p.add_argument('--shutdown-time', type=float, default=1.5,
                   help='放电关断时间（秒）')
    p.add_argument('--tau-r-ms', type=float, default=None,
                   help='理论电阻扩散时间 tau_R（ms），用于比值判据')
    return p.parse_args()


def main():
    args = parse_args()
    out = args.output_dir
    if out is None:  # 默认写到仓库根 data/processed，而不是 firmware/data/processed
        out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                           'data', 'processed')
    cfg = {'output_dir': out, 'shutdown_time': args.shutdown_time}
    if args.tau_r_ms is not None:
        cfg['tau_r_ms'] = args.tau_r_ms

    a = TopoNucleusAnalyzer(cfg)
    a.load(args.csv_path).analyze()
    print(a.summary())
    a.save()


if __name__ == '__main__':
    main()
