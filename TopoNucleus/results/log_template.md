# TopoNucleus 实验结果记录模板

一份实验结果 = 一份文件。存到 `results/<实验者>/<实验ID>.md`，原始数据放 `data/raw/`，特征放 `data/processed/`。

## 必填字段

```markdown
# 实验记录：<实验ID>

- 实验者：<名称 / 化名 / 机构>
- 日期：YYYY-MM-DD
- 硬件版本：<腔体/种子/诊断版本号，见 hardware/ 下的版本标签>
- 软件版本：firmware 提交号（git rev-parse HEAD）

## 1. 实验参数
| 参数 | 值 |
|---|---|
| 种子几何 (seed_geometry) | trefoil / figure8 / cinquefoil / random / ring / straight |
| 绕数 (windings) | 3 / 4 / 5 |
| 拧度 (twist, 0~1) | |
| 环径 (ring_diameter_cm) | 1.0 ~ 5.0 |
| 丝径 (wire_diameter_mm) | 0.5 ~ 2.0 |
| 材料 (seed_material) | iron / tungsten / fe-si / permalloy |
| 表面 (seed_surface) | bare / anodized / coated / copper-plated |
| 对称破缺 (symmetry_break) | none / notch / spike |
| 气体 | Ar / H2 / 混合 |
| 压力 (pressure_mtorr) | |
| 放电方式 | DC / RF |
| 放电功率 (discharge_power_w) | |
| 放电持续时间 (s) | |
| 关断时刻 (shutdown_time_s) | 写入 _meta.json，供 analyzer 自动定位 |
| 理论 tau_R (tau_r_ms) | 按腔体尺寸+电导率估算，写明算法/出处 |

## 2. 原始数据
- CSV：`<文件名>.csv`（列：time_s, bdot_x, bdot_y, bdot_z, photodiode, discharge_current, discharge_voltage）
- 元数据：`<文件名>_meta.json`
- 采集采样率：<Hz>；保存前是否降采样：<是/否，因子>

## 3. 分析输出（直接贴 analyzer 的 stdout）
```

把 `python3 firmware/analysis/analyzer.py data/raw/<csv> --tau-r-ms <值>` 的输出原样粘在下面。

## 4. 人工复核
- [ ] 是否看过原始波形？Bdot 关断后是否真的"慢慢掉"，而不是单次毛刺？
- [ ] 是否重复 ≥ 5 次？重复次数：__
- [ ] 是否有空腔基线对照（不放种子）？
- [ ] 是否排除外界干扰（手机、电机、电源开关瞬态）？
- [ ] 取能天线（Phase 1）是否接了？输出是持续的还是单次感应？

## 5. 结论
- 是否观察到磁场候选（field_candidate）：是 / 否
- 是否观察到综合候选（is_candidate）：是 / 否
- 备注 / 异常现象 / 照片或视频链接：
```

## 证伪同样有价值

如果你跑了若干参数、什么都没出来，请把"哪些组合试过、结果如何"写清楚。
这能让后来者直接跳过死路，把注意力放到新方向上。

## 数据格式约定（v0.2）

- `_meta.json` 必须包含 `experiment_id` 与上面"实验参数"表里的字段（字符串/数值均可），
  进化引擎只读取**已从实验参数记录过的 8 个参数**，不会把没记录的东西当搜索维度。
- 适应度 F = t_hold × V_pp / E_inj，仅用于搜索排序，**不是效率，也不是能量增益**。
- 任何"候选"都需要人工复核；本项目的统计判据只负责把值得复核的挑出来。
