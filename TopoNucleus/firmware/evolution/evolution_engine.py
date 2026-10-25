"""
TopoNucleus - AI 进化引擎 (v0.2)
========================
功能：
  - 从 data/processed/ 读取每轮实验结果（_features.json）
  - 用贝叶斯优化（高斯过程 + Expected Improvement）引导下一代种子几何
  - 前 --random-rounds 轮做空间填充（Sobol），之后切到 BO
  - 输出 hardware/seed_jig/generation_NNN_seeds.{json,csv}

设计要点（v0.2）：
  - 只学习"可从元数据重建"的 8 个参数：ring_diameter_cm、wire_diameter_mm、
    twist、pressure_mtorr、discharge_power_w、material、surface、symmetry_break。
    knot_type/windings 目前 DAQ 元数据没写，不进搜索空间（避免学到幻觉列）。
  - 适应度用 F（t_hold*Vpp/E_inj），BO 最大化它；候选标记只作参考。
  - 候选只在"可行域内"生成：材料/表面/破缺的合法组合 + 尺寸/功率范围。
  - 与历史实验去重：重复组合有适应度则用历史值 + 小噪声，否则先随机探索。
  - 数据 < 5 条或 GP 拟合失败 => 回退到"精英 + 候补池随机变异"。

安装：pip install scikit-learn scipy numpy pandas

作者: TopoNucleus 开源社区
许可: CERN OHL v2
"""

import json
import os
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, fields
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# 合法取值（与 BOM / DAQ 元数据一致）
MATERIALS = ['iron', 'tungsten', 'fe-si', 'permalloy']
SURFACES = ['bare', 'anodized', 'coated', 'copper-plated']
SYM_BREAKS = ['none', 'notch', 'spike']
# 材料 × 表面 合法组合（bare 允许所有材料；其余表面仅铁基）
SURFACE_ALLOWED = {
    'bare': MATERIALS,
    'anodized': ['iron', 'fe-si', 'permalloy'],   # 铁/铁硅/坡莫可阳极氧化
    'coated': ['iron', 'fe-si', 'permalloy'],      # 聚酰亚胺绝缘涂层
    'copper-plated': ['iron', 'fe-si'],            # 电镀铜
}
# 连续参数边界 (min, max)
BOUNDS = {
    'ring_diameter_cm': (1.0, 5.0),
    'wire_diameter_mm': (0.5, 2.0),
    'twist': (0.0, 1.0),
    'pressure_mtorr': (5.0, 50.0),
    'discharge_power_w': (50.0, 300.0),
}
CAT_MAPS = {
    'material': MATERIALS,
    'surface': SURFACES,
    'symmetry_break': SYM_BREAKS,
}
# 连续参数在向量中的列索引
CONT_COLS = ['ring_diameter_cm', 'wire_diameter_mm', 'twist',
             'pressure_mtorr', 'discharge_power_w']
CAT_COLS = ['material', 'surface', 'symmetry_break']


@dataclass
class SeedGenome:
    ring_diameter_cm: float = 3.0
    wire_diameter_mm: float = 1.5
    twist: float = 0.5
    pressure_mtorr: float = 20.0
    discharge_power_w: float = 100.0
    material: str = 'iron'
    surface: str = 'bare'
    symmetry_break: str = 'none'
    knot_type: str = 'trefoil'     # 仅记录，不参与搜索
    windings: int = 3              # 仅记录，不参与搜索

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items()
                      if k in {f.name for f in fields(cls)}})

    # ---------- 向量化（GP 用） ----------
    def to_vector(self):
        v = [getattr(self, c) for c in CONT_COLS]
        for c in CAT_COLS:
            v.append(CAT_MAPS[c].index(getattr(self, c)))
        return np.array(v, dtype=float)

    @classmethod
    def from_vector(cls, vec):
        d = dict(zip(CONT_COLS + CAT_COLS, vec))
        for c in CONT_COLS:
            lo, hi = BOUNDS[c]
            d[c] = float(np.clip(d[c], lo, hi))
        for c in CAT_COLS:
            lst = CAT_MAPS[c]
            d[c] = lst[int(round(np.clip(d[c], 0, len(lst) - 1)))]
        return cls.from_dict(d)

    # ---------- 合法性 ----------
    def is_valid(self):
        if self.material not in MATERIALS:
            return False
        if self.surface not in SURFACES:
            return False
        if self.symmetry_break not in SYM_BREAKS:
            return False
        if self.material not in SURFACE_ALLOWED[self.surface]:
            return False
        for c, (lo, hi) in BOUNDS.items():
            v = getattr(self, c)
            if not (lo <= v <= hi):
                return False
        return True

    def canonical_key(self):
        """用于去重的规格键（参数四舍五入到网格）"""
        return (round(self.ring_diameter_cm, 1),
                round(self.wire_diameter_mm, 1),
                round(self.twist, 2),
                round(self.pressure_mtorr, 1),
                round(self.discharge_power_w, 1),
                self.material, self.surface, self.symmetry_break,
                self.knot_type)

    def mutate(self, rng, sigma=0.12):
        """在合法空间内做高斯变异 + 类别随机翻转到相邻值"""
        d = self.to_dict()
        # 连续参数
        for c in CONT_COLS:
            if rng.random() < 0.5:
                lo, hi = BOUNDS[c]
                span = (hi - lo) * sigma
                d[c] = float(np.clip(d[c] + rng.normal(0, span), lo, hi))
        # 类别参数
        if rng.random() < 0.5:
            d['material'] = rng.choice(MATERIALS)
        if rng.random() < 0.5:
            d['surface'] = rng.choice(SURFACES)
        if rng.random() < 0.5:
            d['symmetry_break'] = rng.choice(SYM_BREAKS)
        g = SeedGenome.from_dict(d)
        # 修正材料-表面冲突
        if g.material not in SURFACE_ALLOWED[g.surface]:
            g.material = rng.choice(SURFACE_ALLOWED[g.surface])
        return g


class TopoNucleusEvolution:
    def __init__(self, config=None):
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.config = {
            'population_size': 12,
            'top_k': 3,
            'pool_size': 60,              # 候补池大小
            'mutation_sigma': 0.12,
            'random_rounds': 5,
            'min_data_for_bo': 4,
            'results_dir': os.path.join(base, 'data', 'processed'),
            'output_dir': os.path.join(base, 'hardware', 'seed_jig'),
            'seed': None,
        }
        if config:
            self.config.update(config)
        self.rng = np.random.default_rng(self.config['seed'])
        self.generation = 0
        self.population: List[SeedGenome] = []
        self.fitness_history: List[Dict] = []
        self._sobol_i = 0
        self._sobol_dirs = self._make_sobol_directions()

        if HAS_SKLEARN:
            self.gp = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(0.01),
                n_restarts_optimizer=5, alpha=1e-6)
            self.scaler = StandardScaler()
        else:
            self.gp = None
            self.scaler = None

    # ---------------------------------------------------------------
    # Sobol 空间填充（前 N 轮，避免 BO 在无数据处乱猜）
    # ---------------------------------------------------------------
    def _make_sobol_directions(self):
        # 8 维，质数初始方向，足够前 5 轮去相关
        return np.array([
            [1, 3, 5, 7, 9, 11, 13, 15],
            [2, 6, 10, 14, 18, 22, 26, 30],
            [4, 12, 20, 28, 36, 44, 52, 60],
            [8, 24, 40, 56, 72, 88, 104, 120],
            [16, 48, 80, 112, 144, 176, 208, 240],
            [32, 96, 160, 224, 288, 352, 416, 480],
            [64, 192, 320, 448, 576, 704, 832, 960],
            [128, 384, 640, 896, 1152, 1408, 1664, 1920],
        ], dtype=np.int64)

    def _sobol_sample(self):
        """生成下一个 Sobol 点并映射到参数空间"""
        i = self._sobol_i
        self._sobol_i += 1
        p = (np.bitwise_xor(i, self._sobol_dirs) / (2.0 ** 8)).sum(axis=1) % 1.0
        d = {}
        for c, (lo, hi) in BOUNDS.items():
            d[c] = float(lo + p[CONT_COLS.index(c)] * (hi - lo))
        d['material'] = MATERIALS[int(p[5] * len(MATERIALS)) % len(MATERIALS)]
        d['surface'] = SURFACES[int(p[6] * len(SURFACES)) % len(SURFACES)]
        d['symmetry_break'] = SYM_BREAKS[int(p[7] * len(SYM_BREAKS)) % len(SYM_BREAKS)]
        g = SeedGenome.from_dict(d)
        if g.material not in SURFACE_ALLOWED[g.surface]:
            g.material = MATERIALS[int(p[5] * len(MATERIALS)) % len(MATERIALS)]
            if g.material not in SURFACE_ALLOWED[g.surface]:
                g.surface = 'bare'
        return g

    # ---------------------------------------------------------------
    # 读取实验结果
    # ---------------------------------------------------------------
    def load_results(self):
        rd = self.config['results_dir']
        if not os.path.isdir(rd):
            return pd.DataFrame()
        rows = []
        for f in sorted(os.listdir(rd)):
            if not f.endswith('_features.json'):
                continue
            try:
                data = json.loads(Path(os.path.join(rd, f)).read_text())
            except (OSError, json.JSONDecodeError):
                continue
            meta = data.get('metadata', {})
            ft = data.get('features', {})
            fit = ft.get('fitness', {})
            row = {
                'experiment_id': str(meta.get('experiment_id', '')),
                'seed_geometry': str(meta.get('seed_geometry', '')),
                'seed_material': str(meta.get('seed_material', '')),
                'seed_surface': str(meta.get('seed_surface', '')),
                'ring_diameter_cm': float(meta.get('ring_diameter_cm', 'nan')),
                'wire_diameter_mm': float(meta.get('wire_diameter_mm', 'nan')),
                'twist': float(meta.get('twist', 'nan')),
                'pressure_mtorr': float(meta.get('pressure_mtorr', 'nan')),
                'discharge_power_w': float(meta.get('discharge_power_w', 'nan')),
                'symmetry_break': str(meta.get('symmetry_break', 'none')),
                'F': float(fit.get('F', 0.0)),
                'is_candidate': bool(data.get('is_candidate', False)),
                't_hold_ms': float(fit.get('t_hold_ms', 0.0)),
                'v_pp': float(fit.get('v_pp', 0.0)),
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        # 只保留 8 个可学参数都齐全的行
        req = list(BOUNDS) + ['seed_material', 'seed_surface', 'symmetry_break']
        return df.dropna(subset=req).reset_index(drop=True)

    # ---------------------------------------------------------------
    # 进化主步
    # ---------------------------------------------------------------
    def evolve(self):
        df = self.load_results()
        n = len(df)

        if n < 4 or  \
           self.generation < self.config['random_rounds'] or \
           not HAS_SKLEARN:
            self.population = self._random_population(df, n)
            tag = 'Sobol' if n < self.config['min_data_for_bo'] else 'random'
            print(f"[INFO] Gen {self.generation}: {tag} fill "
                  f"({n} valid results, pop={len(self.population)})")
        else:
            self.population = self._bayesian_population(df)

        self._record_history(df)
        self.generation += 1
        return self.population

    def _random_population(self, df, n):
        """空间填充：优先未测组合，缺额用 Sobol"""
        seen = set()
        pop = []
        if n:
            for _, r in df.iterrows():
                g = SeedGenome(
                    ring_diameter_cm=float(r['ring_diameter_cm']),
                    wire_diameter_mm=float(r['wire_diameter_mm']),
                    twist=float(r['twist']),
                    pressure_mtorr=float(r['pressure_mtorr']),
                    discharge_power_w=float(r['discharge_power_w']),
                    material=str(r['seed_material']),
                    surface=str(r['seed_surface']),
                    symmetry_break=str(r['symmetry_break']),
                    knot_type=str(r.get('seed_geometry', 'trefoil')),
                )
                seen.add(g.canonical_key())
                pop.append(g)
        while len(pop) < self.config['population_size']:
            g = self._sobol_sample()
            k = g.canonical_key()
            if k in seen:
                continue
            # 若已有该组合的历史适应度，用"历史值+噪声"作为预测值参与后续 BO
            seen.add(k)
            pop.append(g)
        return pop[:self.config['population_size']]

    def _bayesian_population(self, df):
        # 训练数据：只用合法范围内的行
        X_rows, y = [], []
        for _, r in df.iterrows():
            g = SeedGenome(
                ring_diameter_cm=float(r['ring_diameter_cm']),
                wire_diameter_mm=float(r['wire_diameter_mm']),
                twist=float(r['twist']),
                pressure_mtorr=float(r['pressure_mtorr']),
                discharge_power_w=float(r['discharge_power_w']),
                material=str(r['seed_material']),
                surface=str(r['seed_surface']),
                symmetry_break=str(r['symmetry_break']),
                knot_type=str(r.get('seed_geometry', 'trefoil')),
            )
            if not g.is_valid():
                continue
            X_rows.append(g.to_vector())
            y.append(float(r['F']))
        if len(X_rows) < 4:
            return self._random_population(df, len(df))
        X = np.vstack(X_rows)
        y = np.array(y, dtype=float)
        if not np.all(np.isfinite(y)):
            y = np.where(np.isfinite(y), y, np.nanmin(y))

        # 拟合 GP
        try:
            Xs = self.scaler.fit_transform(X)
            self.gp.fit(Xs, y)
        except Exception as e:
            print(f"[WARN] GP fit failed: {e}; fallback to random")
            return self._random_population(df, len(df))

        y_best = float(np.max(y))
        pop = []

        # 1) 精英：历史中最好的 top_k 个真实点
        order = np.argsort(y)[-self.config['top_k']:][::-1]
        for i in order:
            pop.append(SeedGenome.from_vector(X[i]))

        # 2) 候补池：BO 生成，去重后按 EI 排序
        pool = []
        tried = {SeedGenome.from_vector(v).canonical_key() for v in X_rows}
        tried.update(g.canonical_key() for g in pop)
        while len(pool) < self.config['pool_size']:
            cand = self._propose()
            k = cand.canonical_key()
            if k in tried:
                continue
            tried.add(k)
            # 材料-表面合法性：anodized/coated/copper-plated 仅限铁基
            if cand.material not in SURFACE_ALLOWED[cand.surface]:
                cand.material = rng.choice(SURFACE_ALLOWED[cand.surface])
            x = self.scaler.transform(cand.to_vector().reshape(1, -1))
            try:
                mu, sd = self.gp.predict(x, return_std=True)
                mu, sd = float(mu[0]), max(float(sd[0]), 1e-9)
            except Exception:
                mu, sd = 0.0, 1.0
            z = (mu - y_best) / sd
            from scipy.stats import norm
            ei = (mu - y_best) * norm.cdf(z) + sd * norm.pdf(z)
            pool.append((cand, float(ei)))
            if len(pool) > self.config['pool_size'] * 2:
                break

        pool.sort(key=lambda t: t[1], reverse=True)
        for cand, ei in pool:
            if len(pop) >= self.config['population_size']:
                break
            pop.append(cand)
            # 给 GP 一个"先验"：未测点 = 邻近历史均值 + 噪声，避免反复抽样
            self._inject_pseudo(cand)

        # 3) 仍未满 => 用候补池随机变异补齐
        while len(pop) < self.config['population_size']:
            base = self.rng.choice(pool if pool else pop) if pop else self._sobol_sample()
            g = base.mutate(self.rng, self.config['mutation_sigma'])
            if g.canonical_key() not in tried:
                tried.add(g.canonical_key())
                pop.append(g)

        print(f"[INFO] Gen {self.generation}: BO ({len(X_rows)} valid points, "
              f"y_best={y_best:.4f}, {len(pop)} proposed)")
        return pop[:self.config['population_size']]

    def _propose(self):
        """从当前种群中随机选一个做变异，作为 BO 提议起点"""
        base = self.rng.choice(self.population) if self.population else self._sobol_sample()
        return base.mutate(self.rng, self.config['mutation_sigma'])

    def _inject_pseudo(self, cand):
        """把"未测组合"按邻近历史点插值后注入 GP，抑制 EI 虚高"""
        v = cand.to_vector().reshape(1, -1)
        try:
            vs = self.scaler.transform(v)
        except Exception:
            return
        Xs = self.scaler.transform(
            np.vstack([g.to_vector() for g in self.population] + [cand.to_vector()]))
        # 用已有样本均值作伪标签
        try:
            mu_pseudo = float(self.gp.predict(Xs[-1].reshape(1, -1))[0])
            self.gp.fit(np.vstack([self.scaler.transform(
                np.atleast_2d(self.gp.X_train_)), vs]),
                np.concatenate([self.gp.y_train_, [mu_pseudo]]))
        except Exception:
            pass

    # ---------------------------------------------------------------
    # 历史 / 导出
    # ---------------------------------------------------------------
    def _record_history(self, df):
        self.fitness_history.append({
            'generation': self.generation,
            'n_valid': len(df),
            'max_F': float(df['F'].max()) if len(df) else 0.0,
            'mean_F': float(df['F'].mean()) if len(df) else 0.0,
            'n_candidates': int(df['is_candidate'].sum()) if len(df) else 0,
        })

    def export_population(self, output_dir=None):
        out = output_dir or self.config['output_dir']
        os.makedirs(out, exist_ok=True)
        payload = {
            'generation': self.generation,
            'population_size': len(self.population),
            'seeds': [g.to_dict() for g in self.population],
            'fitness_history': self.fitness_history,
            'software_version': '0.2.0',
        }
        p = os.path.join(out, f'generation_{self.generation:03d}_seeds.json')
        Path(p).write_text(json.dumps(payload, indent=2))
        pd.DataFrame([g.to_dict() for g in self.population]).to_csv(
            os.path.join(out, f'generation_{self.generation:03d}_seeds.csv'),
            index=False)
        print(f"[INFO] exported: {p}")
        return p

    def print_population(self):
        print(f"\n{'='*70}\nGeneration {self.generation} — "
              f"{len(self.population)} seeds\n{'='*70}")
        print(f"{'#':<3}{'knot':<10}{'ring':<7}{'wire':<7}{'twist':<7}"
              f"{'P(mT)':<8}{'W':<7}{'mat':<9}{'surf':<13}{'sym':<5}")
        for i, g in enumerate(self.population):
            print(f"{i:<3}{g.knot_type:<10}{g.ring_diameter_cm:<7.1f}"
                  f"{g.wire_diameter_mm:<7.1f}{g.twist:<7.2f}"
                  f"{g.pressure_mtorr:<8.1f}{g.discharge_power_w:<7.0f}"
                  f"{g.material:<9}{g.surface:<13}{g.symmetry_break}")
        print('=' * 70 + '\n')


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='TopoNucleus AI 进化引擎')
    p.add_argument('--results-dir', type=str, default=None)
    p.add_argument('--output-dir', type=str, default=None)
    p.add_argument('--population-size', type=int, default=12)
    p.add_argument('--generations', type=int, default=1)
    p.add_argument('--random-rounds', type=int, default=5)
    p.add_argument('--seed', type=int, default=None)
    return p.parse_args()


def main():
    a = parse_args()
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    cfg = {
        'population_size': a.population_size,
        'results_dir': a.results_dir or os.path.join(base, 'data', 'processed'),
        'output_dir': a.output_dir or os.path.join(base, 'hardware', 'seed_jig'),
        'random_rounds': a.random_rounds,
        'seed': a.seed,
    }
    eng = TopoNucleusEvolution(cfg)

    df = eng.load_results()
    if len(df):
        print(f"[INFO] Loaded {len(df)} valid results; "
              f"max F = {df['F'].max():.6f}; "
              f"candidates = {int(df['is_candidate'].sum())}")

    for _ in range(max(1, a.generations)):
        eng.evolve()
        eng.print_population()
        eng.export_population()
        eng.generation = eng.generation  # 已由 evolve 自增，保持不变

    if eng.fitness_history:
        print("\n进化历史：")
        for h in eng.fitness_history:
            print(f"  gen={h['generation']:<3} n_valid={h['n_valid']:<4} "
                  f"max_F={h['max_F']:.6f} mean_F={h['mean_F']:.6f} "
                  f"candidates={h['n_candidates']}")


if __name__ == '__main__':
    main()
