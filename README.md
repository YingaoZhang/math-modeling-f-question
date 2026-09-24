# F 题数学建模项目

本项目围绕第一问完成语料质量评分、质量冲突分析和逐域配比—验证 Loss 建模。分析代码、论文报告、结果表格和插图均纳入仓库；原始附件的数据边界和超大文件获取说明见 [`data/README.md`](data/README.md)。

## 环境与运行

推荐 Python 3.11。PyCharm 选择 **Open** 并打开项目根目录，再选择 Python 3.11 解释器。

```powershell
python -m pip install -e .
python -m f_question.q1
```

从已有 CSV 结果重建论文报告：

```powershell
python -m f_question.q1 --report-only
```

命令行可通过 `--data` 指定附件目录。完整结果和口径边界见 [`outputs/q1/第一问_完整报告.docx`](outputs/q1/第一问_完整报告.docx)。

## 项目结构

```text
docs/                       题目资料、数据说明和方法评审稿
data/raw/real_attachments/  可公开提交的原始附件子集；超大附件见 data/README.md
src/f_question/             可复现分析代码
outputs/q1/                  CSV 表格、论文报告和插图
```

## 第一问结果摘要

- Q 被约束在 `(0, 1]`，比较等权、CRITIC 和熵权；各域均值含 bootstrap 95% CI。
- 17 域配比经单纯形闭合和乘性零值替换后转成 16 维 ILR；13 个 The Pile Loss 域分别建模。
- 1M 同尺度线性 Ridge 基线宏平均 R² 为 0.7641；训练集五折 CV 逐域选模型后，11 个 Loss 域选择有 `sum(t)=0` 识别约束的 Data Mixing Laws 指数式、2 个域选择二次交互 Ridge，独立测试逐域 R² 宏平均为 0.8896，13/13 为正。
- 绝对 Loss 跨到 60M/1B 不可迁移；A 与 B Loss、A Q 与 B `Q_score` 缺少配对样本，因此桥接和重标定列为不可识别，不虚构拟合式。

全部计算指标、敏感性表、数据边界和插图均见 `outputs/q1/`。

第一问的完整流程、模型形式、验证结果和图表集中在 [`outputs/q1/第一问_完整报告.docx`](outputs/q1/第一问_完整报告.docx)，避免多个版本造成口径分散。

## 分析模块

- `conflicts.py`：域内冲突判定与惩罚分数。
- `diagnostics.py`：A1/A2/A3 冲突复核、A17/A18 覆盖检查、教育—广告共现率。
- `extrapolation.py`：按配方 `index` 对齐 A12–A15 的逐域 Loss 排序诊断。
- `indicators.py`：22 项质量指标的方向、变换和纳入 Q 规范。
- `reporting.py`：论文补充表格、替代对比和解释。
- `q1.py`：命令行入口及主流程编排；原始附件不在计算中覆盖。
