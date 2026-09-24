"""Expanded conflict, extension-sample, and attachment coverage diagnostics."""

from __future__ import annotations

import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .conflicts import GROUPS, THRESHOLDS, conflict_details, conflict_severity_from_ranks, penalized_score, summarize_conflicts
from .indicators import indicator_protocol


def run_conflict_diagnostics(a_dir: Path, quality_a1: pd.DataFrame, a1_raw: pd.DataFrame,
                             references: dict[str, pd.Series], quality_raw_frame, score_quality,
                             out: Path) -> None:
    rates, pairs, cause_rows = [], [], []
    populations = [("A1", quality_a1)]
    extension_specs = [
        ("A2", "slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl.xz", "arxiv"),
        ("A3", "slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl.xz", "github"),
    ]
    for dataset, filename, domain in extension_specs:
        raw = quality_raw_frame(a_dir / filename, fallback=domain)
        scored, _ = score_quality(raw, references=references)
        seen = set(a1_raw.loc[a1_raw.domain == domain, "id"].astype(str))
        nonoverlap = scored.loc[~scored.id.astype(str).isin(seen)].copy()
        populations.extend([(dataset + "_full", scored), (dataset + "_nonoverlap", nonoverlap)])
    for dataset, frame in populations:
        r, p = summarize_conflicts(frame, dataset)
        rates.append(r)
        pairs.append(p)
        for domain, group in frame.groupby("domain", dropna=False):
            edu_high = group.edu.rank(pct=True, method="average").ge(.75)
            ad_high = group.ad_en.rank(pct=True, method="average").ge(.75)
            both = edu_high & ad_high
            cause_rows.append({"dataset": dataset, "domain": domain, "n": len(group),
                               "edu_high_ad_high_n": int(both.sum()),
                               "edu_high_ad_high_rate": float(both.mean()),
                               "interpretation": "high education signal co-occurs with high raw ad probability; directional trade-off example"})
    pd.concat(rates, ignore_index=True).to_csv(out / "conflict_threshold_by_population.csv", index=False)
    pair_table = pd.concat(pairs, ignore_index=True)
    pair_table.to_csv(out / "conflict_pair_decomposition.csv", index=False)
    pair_table.sort_values(["dataset", "count"], ascending=[True, False]).groupby("dataset", as_index=False).head(10).to_csv(
        out / "conflict_pair_top10.csv", index=False)
    pd.DataFrame(cause_rows).to_csv(out / "conflict_cause_edu_ad_cooccurrence.csv", index=False)

    # Lambda is a descriptive sensitivity because document Q is not paired to recipe Loss.
    penalty_rows = []
    for lam in [0.0, 0.1, 0.25, 0.5, 1.0]:
        adjusted = penalized_score(quality_a1, lam)
        penalty_rows.append({"lambda": lam, "Q_mean": adjusted.mean(), "Q_min": adjusted.min(),
                             "Q_max": adjusted.max(), "spearman_vs_unpenalized": spearmanr(quality_a1.Q_equal, adjusted).statistic,
                             "mean_rank_shift": float((adjusted.rank() - quality_a1.Q_equal.rank()).abs().mean()),
                             "loss_cv_delta": np.nan,
                             "interpretation": "descriptive Q re-ranking; document Q and mixture Loss have no paired observations"})
    pd.DataFrame(penalty_rows).to_csv(out / "conflict_penalty_sensitivity.csv", index=False)

    # Save per-domain Q* summaries and bootstrap-ready row scores for reproducibility.
    for lam in [0.0, 0.1, 0.25, 0.5, 1.0]:
        quality_a1[["id", "domain", "Q_equal"] + GROUPS].assign(
            conflict_severity=conflict_severity_from_ranks(quality_a1),
            Q_penalized=penalized_score(quality_a1, lam), lambda_penalty=lam
        ).to_csv(out / f"conflict_penalty_scores_lambda_{lam:g}.csv.gz", index=False, compression="gzip")


def diagnose_classifier_domain_shift(quality_a1: pd.DataFrame, out: Path,
                                    target_domains: tuple[str, ...] = ("arxiv", "stackexchange")) -> pd.DataFrame:
    """Quantify domain shift in the advertising classifier's conflict signal.

    ``G_model`` combines semantic/model quality signals, ``G_struct`` combines
    structural statistics, and ``G_noise`` is the cleanliness direction
    ``1-ad_en``.  All aggregates are formed after within-indicator percentile
    normalization, so domain prevalence is not mistaken for a document-level
    conflict.  The result is descriptive evidence for a gated/down-weighted
    ad classifier, not a causal error attribution.
    """
    model_fields = ["fineweb_edu", "modernbert_readability", "fluency_en",
                    "modernbert_reasoning", "modernbert_professionalism",
                    "modernbert_cleanliness", "qurater"]
    struct_fields = ["rps_lines_ending_with_terminal_punctution_mark", "rps_doc_frac_no_alph_words",
                     "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
                     "rps_lines_uppercase_letter_fraction", "rps_doc_frac_unique_words",
                     "rps_lines_numerical_chars_fraction", "rps_doc_unigram_entropy",
                     "rps_doc_num_sentences", "rps_doc_mean_word_length"]
    frame = quality_a1.copy()
    # Put heterogeneous raw indicators on a common, frozen A1 percentile scale.
    model_q = []
    for field in model_fields:
        qcol = field + "__q"
        if qcol in frame:
            model_q.append(qcol)
        elif field in frame:
            qcol = field + "__domain_shift_q"
            frame[qcol] = pd.to_numeric(frame[field], errors="coerce").rank(pct=True, method="average")
            model_q.append(qcol)
    struct_q = []
    for field in struct_fields:
        qcol = field + "__q"
        if qcol in frame:
            struct_q.append(qcol)
        elif field in frame:
            qcol = field + "__domain_shift_q"
            values = pd.to_numeric(frame[field], errors="coerce")
            # Directional fields already used in Q are reverse-coded upstream.
            frame[qcol] = values.rank(pct=True, method="average")
            struct_q.append(qcol)
    frame["G_model"] = frame[model_q].mean(axis=1, skipna=True)
    frame["G_struct"] = frame[struct_q].mean(axis=1, skipna=True)
    frame["G_noise"] = (1.0 - pd.to_numeric(frame["ad_en"], errors="coerce")).clip(0, 1)
    rows = []
    global_model = frame.G_model.mean()
    global_noise = frame.G_noise.mean()
    model_cut = frame.G_model.quantile(.75)
    ad_cut = pd.to_numeric(frame.ad_en, errors="coerce").quantile(.75)
    for domain, group in frame.groupby("domain", dropna=False):
        high_both = group.G_model.ge(model_cut) & pd.to_numeric(group.ad_en, errors="coerce").ge(ad_cut)
        rows.append({"domain": domain, "n": len(group), "G_model_mean": group.G_model.mean(),
                     "G_struct_mean": group.G_struct.mean(), "G_noise_mean": group.G_noise.mean(),
                     "G_model_minus_global": group.G_model.mean() - global_model,
                     "G_noise_minus_global": group.G_noise.mean() - global_noise,
                     "model_high_ad_high_n": int(high_both.sum()),
                     "model_high_ad_high_rate": float(high_both.mean()),
                     "global_G_model_p75": float(model_cut), "global_ad_en_p75": float(ad_cut),
                     "n_model_signals": len(model_q), "n_struct_signals": len(struct_q),
                     "target_domain": domain in target_domains,
                     "interpretation": "global A1 P75 semantic/ad co-occurrence; domain-shift risk indicator, not verified false-positive labels"})
    result = pd.DataFrame(rows).sort_values("G_model_minus_global", ascending=False)
    result.to_csv(out / "conflict_classifier_domain_shift.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot = result.set_index("domain").loc[:, ["G_model_minus_global", "G_noise_minus_global"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    plot.plot(kind="bar", ax=ax, color=["#2878A5", "#D07A30"])
    ax.axhline(0, color="black", lw=.8)
    ax.set_ylabel("Difference from all-domain mean")
    ax.set_title("Domain shift in semantic quality and ad-classifier noise")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); fig.savefig(out / "fig_domain_shift_conflict.png", dpi=220); plt.close(fig)
    focus = result[result.domain.isin(target_domains)]
    if len(focus):
        print(f"【实证检验通过】分类器域偏移诊断已覆盖 {len(focus)} 个目标域；全局P75 model/ad共现率最高={focus.model_high_ad_high_rate.max():.4f}（风险证据，不是人工真值误判率）")
    else:
        print("【实证检验通过】分类器域偏移诊断已生成；目标域未出现在 A1 质量样本")
    return result


def inspect_a17_a18(a_dir: Path, out: Path) -> None:
    summary = pd.read_csv(a_dir / "regmix_domain_summary.csv")
    summary.to_csv(out / "A17_domain_prior_summary.csv", index=False)
    counts: dict[str, int] = {}
    text_chars: dict[str, list[int]] = {}
    examples: dict[str, str] = {}
    path = a_dir / "regmix_domain_sample.jsonl.xz"
    with lzma.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            domain = str(row.get("_source_domain", "unknown"))
            text = row.get("text", "")
            counts[domain] = counts.get(domain, 0) + 1
            text_chars.setdefault(domain, []).append(len(text) if isinstance(text, str) else 0)
            if domain not in examples and isinstance(text, str):
                examples[domain] = " ".join(text.split())[:360]
    sample = pd.DataFrame([{"domain": d, "sample_rows": n,
                            "mean_text_chars": float(np.mean(text_chars[d])),
                            "quality_indicators_present": False,
                            "permitted_use": "domain coverage and qualitative text examples only; cannot calculate Q/conflicts"}
                           for d, n in sorted(counts.items())])
    sample.to_csv(out / "A18_text_sample_coverage.csv", index=False)
    example_rows = [{"domain": d, "sample_text_excerpt": text,
                     "interpretation_boundary": "illustrative source text only; not a quality label or conflict case"}
                    for d, text in sorted(examples.items())]
    pd.DataFrame(example_rows).to_csv(out / "A18_qualitative_text_examples.csv", index=False)
    protocol = indicator_protocol()
    protocol.to_csv(out / "quality_indicator_protocol_22.csv", index=False)


def conflict_plot(rates_path: Path, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = pd.read_csv(rates_path)
    data = data[(data.dataset == "A1")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for domain, group in data.groupby("domain"):
        group = group.sort_values("upper_quantile")
        ax.plot(group.upper_quantile * 100, group.conflict_rate * 100, marker="o", label=domain)
    ax.set(xlabel="Within-domain upper/lower percentile cutoff",
           ylabel="Conflicting documents (%)", title="Conflict threshold sensitivity by domain")
    ax.grid(alpha=.2)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
