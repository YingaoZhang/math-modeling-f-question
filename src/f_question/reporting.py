"""Paper-ready interpretation for the expanded first-question diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from itertools import combinations


def _table(frame: pd.DataFrame, digits: int = 4) -> str:
    view = frame.copy()
    for col in view.select_dtypes(include="number"):
        view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    columns = [str(c) for c in view.columns]
    rows = [[str(v).replace("|", "\\|").replace("\n", " ") for v in row]
            for row in view.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(columns) + " |",
                      "| " + " | ".join(["---"] * len(columns)) + " |"] +
                     ["| " + " | ".join(row) + " |" for row in rows])


def write_supplemental_report(out: Path) -> None:
    reversal = pd.read_csv(out / "estimated_loss_ranking_reversal_summary.csv")
    per_domain = pd.read_csv(out / "estimated_loss_1m_cross_scale_by_domain.csv")
    scale = pd.read_csv(out / "estimated_loss_10b_vs_70b_by_domain.csv")
    conflict = pd.read_csv(out / "conflict_threshold_by_population.csv")
    pairs = pd.read_csv(out / "conflict_pair_decomposition.csv")
    penalty = pd.read_csv(out / "conflict_penalty_sensitivity.csv")
    causes = pd.read_csv(out / "conflict_cause_edu_ad_cooccurrence.csv")
    coeff = pd.read_csv(out / "data_mixing_law_coefficients.csv")
    protocol = pd.read_csv(out / "quality_indicator_protocol_22.csv")
    a18 = pd.read_csv(out / "A18_text_sample_coverage.csv")
    examples = pd.read_csv(out / "A18_qualitative_text_examples.csv")
    joint_effects = pd.read_csv(out / "mixture_joint_transfer_effects.csv") if (out / "mixture_joint_transfer_effects.csv").exists() else pd.DataFrame()
    scaling = pd.read_csv(out / "scaling_est_reverse_engineering.csv") if (out / "scaling_est_reverse_engineering.csv").exists() else pd.DataFrame()
    shift = pd.read_csv(out / "conflict_classifier_domain_shift.csv") if (out / "conflict_classifier_domain_shift.csv").exists() else pd.DataFrame()

    avg = coeff.groupby("mixture_domain").agg(
        mean_t=("t_relative_substitution", "mean"), median_t=("t_relative_substitution", "median"),
        positive_loss_domains=("t_relative_substitution", lambda x: int((x > 0).sum())),
        negative_loss_domains=("t_relative_substitution", lambda x: int((x < 0).sum()))).reset_index()
    avg = avg.sort_values("mean_t")
    avg.to_csv(out / "data_mixing_law_domain_interpretation.csv", index=False)
    domains = coeff[["mixture_domain", "loss_domain", "t_relative_substitution"]]
    substitution_rows = []
    for loss_domain, group in domains.groupby("loss_domain"):
        values = group.set_index("mixture_domain").t_relative_substitution
        for gained, replaced in combinations(values.index, 2):
            substitution_rows.append({"loss_domain": loss_domain, "gain_domain": gained,
                                      "replaced_domain": replaced,
                                      "delta_log_excess_loss_per_unit_share": values[gained] - values[replaced],
                                      "interpretation": "increase gain_domain share by replacing replaced_domain share; conditional association"})
    substitution = pd.DataFrame(substitution_rows)
    substitution.to_csv(out / "data_mixing_law_pairwise_substitution.csv", index=False)
    extremes = pd.concat([
        substitution.loc[substitution.groupby("loss_domain").delta_log_excess_loss_per_unit_share.idxmin()].assign(direction="lower predicted excess Loss"),
        substitution.loc[substitution.groupby("loss_domain").delta_log_excess_loss_per_unit_share.idxmax()].assign(direction="higher predicted excess Loss")
    ], ignore_index=True)
    extremes.to_csv(out / "data_mixing_law_extreme_substitutions.csv", index=False)

    key_pairs = pairs.loc[pairs.upper_quantile == .75].groupby(["dataset", "group_a", "group_b"], as_index=False).agg(
        pair_count=("count", "sum"))
    key_pairs["share"] = key_pairs.pair_count / key_pairs.groupby("dataset").pair_count.transform("sum")
    key_pairs = key_pairs.sort_values(["dataset", "pair_count"], ascending=[True, False])
    threshold_a1 = conflict.loc[conflict.dataset == "A1"].groupby("upper_quantile").agg(
        total_conflicts=("conflict_n", "sum"),
        mean_domain_rate=("conflict_rate", "mean")).reset_index()
    threshold_a1["n"] = conflict.loc[(conflict.dataset == "A1") & (conflict.upper_quantile == .75), "n"].sum()
    threshold_a1["weighted_rate"] = threshold_a1.total_conflicts / threshold_a1.n
    expansion = conflict.loc[conflict.upper_quantile == .75].groupby("dataset").agg(
        records=("n", "sum"), mean_domain_conflict_rate=("conflict_rate", "mean")).reset_index()
    cause_summary = causes.groupby("dataset").agg(
        records=("n", "sum"), edu_high_ad_high_n=("edu_high_ad_high_n", "sum"),
        mean_domain_rate=("edu_high_ad_high_rate", "mean")).reset_index()
    cause_summary["weighted_rate"] = cause_summary.edu_high_ad_high_n / cause_summary.records
    line = []
    line += ["# 第一问补充实证：冲突机制、外推排序与组成效应", "",
             "## A12–A15：估算外推排序", "",
             "A12–A15 中 10B 与 70B 均有 63 个相同配方、13 个域完整 Loss。以下结果按 `index` 对齐；它们是估算数据诊断，不是独立实验。跨域合成先分别对每个域做 z 标准化，再对域 z 分数等权平均；不使用原始 Loss 算术均值作为结论。", "",
             _table(reversal), "",
             "逐域对照 1M 观测训练 Loss 与估算 Loss：", "", _table(per_domain), "",
             f"1M 与 10B、1M 与 70B 的逐域 Spearman 中位数分别为 {per_domain.loc[per_domain.scale == '10b', 'spearman_1m_vs_estimated'].median():.3f}、"
             f"{per_domain.loc[per_domain.scale == '70b', 'spearman_1m_vs_estimated'].median():.3f}；A12–A15 内部 10B/70B 域内排序 Spearman 中位数为 {scale.spearman_loss_10b_70b.median():.3f}。"
             f"经域内标准化后的 1M/10B、1M/70B 配方综合排序 Spearman 分别为 {reversal.loc[reversal.comparison.str.contains('1M vs 10b'), 'spearman'].iloc[0]:.3f}、"
             f"{reversal.loc[reversal.comparison.str.contains('1M vs 70b'), 'spearman'].iloc[0]:.3f}。这些估算结果仅说明跨尺度配方排序关联，不能解释为绝对 Loss 的跨尺度迁移。", "",
             "## A12–A15 标度生成机制反演", "",
             (f"逐域计算 gamma=-log(L_N/L_1M)/log(N/1M)，并对三尺度 log Loss 做乘性拟合。估算 gamma(10B) 与 gamma(70B) 的池化相关为 "
              f"{scaling.loc[scaling.dataset == 'estimated_pooled_domain_recipe', 'pearson_gamma10_gamma70'].iloc[0]:.4f}；估算域内中位 log-fit R² 为 "
              f"{scaling.loc[scaling.dataset == 'estimated', 'median_scaling_fit_r2'].median():.4f}。与 1M 初始 Loss 的相关未达到 r>0.90 的人工耦合判据，真实 1M→60M 对照也未复现统一高正相关。池化统计是域-配方描述性诊断，不是独立样本推断。" if len(scaling) else "标度反演结果见 scaling_est_reverse_engineering.csv。"), "",
             "## 冲突、成因与惩罚", "",
             "冲突在每个来源域内分别计算五组分数的经验秩：至少一组处于上分位且另一组处于下分位即定义为冲突。组对表统计高低分共现次数；这是描述性构成，不作因果解释。", "",
             "A1 阈值敏感性汇总：", "", _table(threshold_a1), "",
             "抽样集与扩展集在 P75/P25 下的复核：", "", _table(expansion), "",
             "冲突组对的前列组合（按数据集）：", "", _table(key_pairs.groupby("dataset", as_index=False, group_keys=False).head(5)), "",
             "题目示例的定向检验：域内教育价值组处于 P75 以上且原始广告概率处于 P75 以上（即清洁度方向不利）的共现率。", "", _table(cause_summary), "",
             "分类器域偏移诊断把 ModernBERT/FineWeb-Edu/Qurater 合成为 G_model，把结构统计合成为 G_struct，并以 G_noise=1-ad_en 表示清洁方向。目标域结果如下；无人工广告真值，因此这是假阳性风险证据而非误判率。", "",
             _table(shift.loc[shift.domain.isin(["arxiv", "stackexchange"]), ["domain", "G_model_mean", "G_noise_mean", "G_model_minus_global", "G_noise_minus_global", "model_high_ad_high_rate"]] if len(shift) else pd.DataFrame()), "",
             "综合评分敏感性定义为 `Q*=clip(Q_equal-λC, ε, 1)`，其中 C 是五组域内经验秩相对中位秩的平均绝对偏差（映射至 [0,1]）。", "",
             _table(penalty), "",
             "现有附件没有将文档级 Q 与配方级 13 域 Loss 配对的观测。因此 λ 表只评估 Q 的范围和排序变化；Loss CV 增益不可识别，不能据此宣称惩罚改善了模型。若以冲突率作为惩罚也属于定义型评分而非监督消解。论文主 Q 保留 λ=0，惩罚分作为敏感性评分。", "",
             "## 17 域配比效应解读", "",
             "指数式为 `L_d=c_d+k_d exp(Σ_j t_dj p_j)`，并约束 `Σ_j t_dj=0`。配比从域 k 转移到域 j 时，`log(L_d-c_d)` 的局部变化率是 `t_dj-t_dk`，是明确的相对替代对比。", "",
             "各 Loss 域中模型内最强和最弱的替代方向（关联解释）：", "", _table(extremes), "",
             "该指数模型的线性预测项没有配比交互项，所以可估计替代效应，不能检验域对协同/互补；声称互补需要另加交互并通过独立检验集验证。下面的 t 汇总仅作跨域描述，不能代替逐 Loss 域系数。", "",
             _table(avg), "",
             "`nih_exporter`、`enron_emails`、`europarl`、`philpapers` 无直接验证 Loss 列，其 t 仅表示它们进入混合比例后对其他有观测域 Loss 的替代关系，不是对该域自身质量的预测。", "",
             "## A17/A18 与指标规范", "",
             "A17 `regmix_domain_summary.csv` 已导出为域样本量、平均文本长度的先验表。A18 `regmix_domain_sample.jsonl.xz` 已做全量域覆盖和条数核验，可为论文提供原始文本定性例证；该文件不含 22 项质量信号，不能用来计算 Q 或量化冲突。", "",
             _table(a18), "",
             "A18 原始文本示例覆盖全部 17 域（仅示意领域文本差异，不作为评分或冲突证据）：", "", _table(examples), "",
             "22 个指标的方向、变换、分组及是否进入 Q 已规范化输出在 `quality_indicator_protocol_22.csv`。", "",
             "## 解释边界", "",
             "A12–A15 是共用配方的估算结果；A2/A3 与 A1 样本重叠部分须剔除后才可作扩展复核。跨域冲突组对不等于因果机制；配方系数只在给定混合比例空间、给定 Loss 域和模型形式下解释。", ""]
    if len(joint_effects):
        joint_summary = joint_effects.groupby("loss_domain", as_index=False).agg(
            evaluated_combinations=("mean_absolute_nonadditive_delta_loss", "size"),
            median_abs_nonadditive_delta=("mean_absolute_nonadditive_delta_loss", "median"),
            median_sign_consistency=("sign_consistency", "median"))
        top_idx = joint_effects.groupby("loss_domain").mean_absolute_nonadditive_delta_loss.idxmax()
        top_joint = joint_effects.loc[top_idx].sort_values("loss_domain")
        line += ["## 二阶组合效应", "",
                 "独立检验配方中，对来源域分别向两个目标域各转移 5 个百分点。分析限于训练集平均占比最高的 8 个组成域。非加性量为联合预测变化减去两项单独变化之和；它刻画二次 Ridge 的条件关联，不是因果协同。不同 Loss 域的绝对量纲不同，因此逐域汇总，不跨域按效应大小排序。", "",
                 _table(joint_summary), "",
                 "各 Loss 域内平均绝对非加性差值最大的组合：", "", _table(top_joint), ""]
    (out / "第一问_补充实证与解释.md").write_text("\n".join(line), encoding="utf-8")
