# TopoNucleus — 拓扑晶核等离子体自组织实验

> 一个假设，一份清单，一个邀请。不是反应堆，不是永动机，是一个你可以自己搭的实验。

**核心假设**：在弱放电等离子体中放入扭结几何的导电骨架，粒子轨道被几何强制缠绕，可能涌现无需外部持续供能的相干磁结构。
**诚实标注**：L1（鞘层）已知；L2（几何引导）理论支持、无直接实验；**L3（拓扑涌现自持）是本项目要验证的核心赌注**；L4/L5 属远期假设。详见 [THEORY.md](THEORY.md)。

## 目录
| 路径 | 内容 |
|---|---|
| [`THEORY.md`](THEORY.md) | 原理分层、公式、参考文献、**判据的统计含义与已知漏洞** |
| [`hardware/BOM.md`](hardware/BOM.md) | 可下单加工清单（含规格/数量/来源/预算） |
| [`hardware/CERN_OHL_v2.txt`](hardware/CERN_OHL_v2.txt) | 硬件许可（CERN OHL v2） |
| [`firmware/`](firmware/) | 采集、分析、AI 进化代码 |
| [`results/log_template.md`](results/log_template.md) | 实验结果提交模板（含人工复核清单） |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 怎么参与 |
| [`docs/CC-BY-SA-4.0.txt`](docs/CC-BY-SA-4.0.txt) | 文档许可 |

## 三步跑起来（v0.2）

```bash
pip install -r firmware/requirements.txt

# 1) 生成模拟数据（有 NI-DAQmx 时自动切到真实采集，见下）
python3 firmware/daq/daq.py --seed trefoil --pressure 20 --power 100

# 2) 分析：判据 + 适应度；--tau-r-ms 填你按 THEORY §7.1 估算的 tau_R
python3 firmware/analysis/analyzer.py data/raw/<csv> --tau-r-ms 8.0

# 3) 进化：前 5 代空间填充，之后切贝叶斯优化（需要 ≥5 条有效结果）
python3 firmware/evolution/evolution_engine.py --population-size 12 --generations 3
```

**真实采集（有 NI 硬件）**：`pip install nidaqmx`，接线按 `firmware/daq/daq.py` 顶部通道表，
运行方式不变（模拟/真实由是否装了 nidaqmx 自动切换）。

**真实数据替换**：把 CSV + `_meta.json` 丢进 `data/raw/`，跑第 2 步，再跑第 3 步。

## 关键：怎么读结果（v0.2 修订）

| 标记 | 含义 | 别把它当成 |
|---|---|---|
| `field_candidate` | 磁场包络衰减 τ>25ms、非指数(p<0.05)、快于扩散 | **不等于"自持已证实"** |
| `is_candidate` | field_candidate +（发光持续 或 阻抗跳变） | 不等于能量增益 |
| 适应度 `F` | τ_e × V_pp / E_inj，仅用于搜索排序 | **不是效率、不是 Q 值** |

判据的历史漏洞与校准见 [THEORY.md §7](THEORY.md#7-判据的数学定义) 与 §11。
任何"候选"都必须按 `results/log_template.md` 第 4 节人工复核 + 重复 ≥5 次。

## 怎么参与

复现 / 改几何 / 证伪 / 改进，把结果按模板写进 `results/`。**证伪同样有价值。**
详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可
- 硬件（BOM、图纸、模具、腔体改造）：[CERN OHL v2](hardware/CERN_OHL_v2.txt)
- 文档（README、THEORY、结果记录）：[CC-BY-SA 4.0](docs/CC-BY-SA-4.0.txt)
- 数据（raw / processed）：[CC0](https://creativecommons.org/publicdomain/zero/1.0/)
