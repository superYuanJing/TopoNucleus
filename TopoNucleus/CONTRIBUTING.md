# 参与 TopoNucleus

欢迎任何人复现、改几何、证伪、改进。不需要是等离子体物理学家。

## 四条路

| 方式 | 做什么 | 怎么提交 |
|---|---|---|
| 复现 | 按 `hardware/BOM.md` 搭一套，跑 Phase 0 | `results/<你的名字>/<实验ID>.md` + `data/raw/` + `data/processed/` |
| 变异 | 改种子几何/材料/表面，或改腔体参数 | 在 `hardware/seed_jig/` 或 `hardware/chamber_mods/` 提设计文件 + issue |
| 证伪 | 跑了一组参数什么都没出来 | 同上，写清楚试了哪些组合 |
| 改进 | 理论推导、分析判据、进化算法、新诊断 | Pull Request，说明改了什么、为什么 |

## 提交前检查

- [ ] 用的是哪个版本的硬件、哪个版本的 firmware（git 提交号）
- [ ] 原始数据 CSV + 元数据 `_meta.json` 齐全
- [ ] 元数据里写全了 8 个参数：ring_diameter_cm、wire_diameter_mm、twist、pressure_mtorr、discharge_power_w、seed_material、seed_surface、symmetry_break
- [ ] 按 `results/log_template.md` 第 4 节做过人工复核
- [ ] 没把"适应度 F"说成"能量增益"或"效率"

## 代码约定

- Python 3.9+，风格跟随现有代码（不引入新框架）
- 新依赖写进 `firmware/requirements.txt`，并标注"核心/可选"
- 分析模块的改动请附带一组 `data/raw/` 下的回归数据，跑 `analyzer.py` 确认输出无回退
- 进化引擎的改动请用 `--seed 42` 固定随机种子跑 5 代，确认结果可复现

## 不欢迎的

- 宣称"免费能源""永动机""已经成功"而没有原始数据
- 用"适应度 F"当效率宣传
- 不公开衍生设计（违反 CERN OHL v2）

## 沟通

- 科学问题 / 假设讨论：开 issue，标签 `theory`
- 硬件 / 加工：开 issue，标签 `hardware`
- 软件 / 判据：开 issue，标签 `software`
- 结果报告：直接提交 PR 到 `results/`
