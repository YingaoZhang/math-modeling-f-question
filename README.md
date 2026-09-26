# F 题数学建模项目

本项目包含数学建模 F 题前四问的可复现分析、结果表和论文。第一问评估语料质量并逐域建模配比与 The Pile Loss；第二问拟合 Pythia 标度律并分析质量效应；第三问在预算约束下优化配置；第四问分析模型效率前沿和规模—时间贡献。原始附件的数据边界和超大文件获取说明见 [`data/README.md`](data/README.md)。

## 环境与运行

推荐 Python 3.11。PyCharm 选择 **Open** 并打开项目根目录，再选择 Python 3.11 解释器。

```powershell
python -m pip install -e .
$env:PYTHONPATH='src'
python -m f_question.q1
python -m f_question.q2
python -m f_question.q3
python -m f_question.q4
```

运行四问后构建完整论文：

```powershell
python tools/build_complete_paper.py
```

命令行可通过 `--data` 指定附件目录。四问统一结果、方法、图表和附录仅交付 [`outputs/数学建模F题完整论文.docx`](outputs/数学建模F题完整论文.docx)。Q4 会审计 C8 JSON：4 个无法解析的文件标记为损坏并排除，其余 1,954 个可解析 JSON 进入官方七任务聚合；C8 任务规模回归与 C3 年度回归分开报告。

## 项目结构

```text
docs/                       题目资料、数据说明和方法评审稿
data/raw/real_attachments/  可公开提交的原始附件子集；超大附件见 data/README.md
src/f_question/             可复现分析代码
outputs/q1/ ... outputs/q4/    四问 CSV/JSON 可复核结果
outputs/figures/             统一存放论文插图
outputs/数学建模F题完整论文.docx  问题一至问题四统一论文（最终交付）
```

## 第一问结果摘要

- Q 被约束在 `(0, 1]`，比较等权、CRITIC 和熵权；各域均值含 bootstrap 95% CI。
- 17 域配比经单纯形闭合和乘性零值替换后转成 16 维 ILR；13 个 The Pile Loss 域分别建模。
- 1M 同尺度线性 Ridge 基线宏平均 R² 为 0.7641；训练集五折 CV 逐域选模型后，11 个 Loss 域选择有 `sum(t)=0` 识别约束的 Data Mixing Laws 指数式、2 个域选择二次交互 Ridge，独立测试逐域 R² 宏平均为 0.8896，13/13 为正。
- 绝对 Loss 跨到 60M/1B 不可迁移；A 与 B Loss、A Q 与 B `Q_score` 缺少配对样本，因此桥接和重标定列为不可识别，不虚构拟合式。

全部计算指标和敏感性表见 `outputs/q1/`，插图统一收录在 `outputs/figures/`。

第一至第四问的完整流程、模型形式、验证结果、图表和数据边界说明集中在唯一交付文件 `outputs/数学建模F题完整论文.docx`。构建时产生的各问 Markdown 中间稿会在 Word 保存后清理；重新构建前需依次运行四问。

## 分析模块

- `conflicts.py`：域内冲突判定与惩罚分数。
- `diagnostics.py`：A1/A2/A3 冲突复核、A17/A18 覆盖检查、教育—广告共现率。
- `extrapolation.py`：按配方 `index` 对齐 A12–A15 的逐域 Loss 排序诊断。
- `indicators.py`：22 项质量指标的方向、变换和纳入 Q 规范。
- `reporting.py`：论文补充表格、替代对比和解释。
- `q1.py`：命令行入口及主流程编排；原始附件不在计算中覆盖。

## 第二问结果摘要

第二问使用 B1 Pythia 日志拟合带不可约损失的标度律 `L=E+A N^{-alpha}+B D^{-beta}`，并以 B1 的 beta 作为两阶段先验；B2/B3 做模型族外与轨迹验证，B4/B5 做跨族和文献验证，B6--B8 单独识别原生 `Q_score` 效应，B9/B10 仅作百亿参数以上情景外推。第一问的 `Q_star` 和 ILR 配比通过 `outputs/q1/q2_input_interface_bundle.json` 接入，不与 The Pile Loss 直接拼接。

```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe -m f_question.q2
```

第二问结果表保存在 `outputs/q2/`；第三、四问分别由 `src/f_question/q3.py`、`src/f_question/q4.py` 运行。运行完四问后，使用 `tools/build_complete_paper.py` 重建唯一的完整 Word 论文，构建器会把插图归集到 `outputs/figures/`。

