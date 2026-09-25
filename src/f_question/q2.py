"""Reproducible pipeline for Problem 2: physical scaling-law correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from .q2_scaling import (fit_classic_scaling, cross_validate_classic,
                         fit_quality_effect, fit_asymmetric_quality_effect,
                         b1_grid_diagnostics, asymmetric_quality_prediction,
                         bootstrap_classic_scaling)
from .q2_validation import (external_validation, trajectory_validation, large_scale_extrapolation,
                            b2_recalibration, affine_bridge_test, mrtS_table, summarize_mrts,
                            second_order_summary)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "real_attachments" / "B_scaling_laws"
OUT = PROJECT_ROOT / "outputs" / "q2"


def _savefig(path: Path, dpi: int = 300) -> None:
    plt.tight_layout(); plt.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close()


def _deidentified_a18() -> pd.DataFrame:
    """Create a 17-domain technical A18 table without source excerpts."""
    a_dir = PROJECT_ROOT / "data" / "raw" / "real_attachments" / "A_data_value"
    summary = pd.read_csv(a_dir / "regmix_domain_summary.csv")
    keywords = {
        "arxiv": ("abstract", "latex", "formula"), "dm_mathematics": ("problem", "equation", "remainder"),
        "enron_emails": ("email", "forwarded", "approval"), "europarl": ("debate", "motion", "translation"),
        "freelaw": ("statute", "court", "jurisdiction"), "github": ("repository", "code", "license"),
        "gutenberg_pg_19": ("novel", "chapter", "character"), "hackernews": ("technology", "comment", "startup"),
        "nih_exporter": ("research", "grant", "biomedical"), "philpapers": ("philosophy", "argument", "citation"),
        "pile_cc": ("web", "document", "crawl"), "pubmed_abstracts": ("abstract", "clinical", "study"),
        "pubmed_central": ("case", "method", "result"), "stackexchange": ("question", "answer", "code"),
        "ubuntu_irc": ("chat", "package", "debugging"), "uspto_backgrounds": ("patent", "claim", "invention"),
        "wikipedia_en": ("encyclopedia", "reference", "topic"),
    }
    out = []
    for _, row in summary.iterrows():
        d = str(row.domain); kw = keywords.get(d, ("domain", "document", "text"))
        out.append({"domain": d, "keyword_1": kw[0], "keyword_2": kw[1], "keyword_3": kw[2],
                    "sample_rows": int(row.sample_rows), "mean_text_chars": float(row.avg_text_chars),
                    "permitted_use": "technical domain example only; no continuous source text, no Q/conflict label"})
    return pd.DataFrame(out).sort_values("domain").reset_index(drop=True)


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    print("【实证检验】读取 B1--B10 附件并区分数据边界")
    b1 = pd.read_csv(DATA_ROOT / "pythia_training_log_existing.csv")
    model, diag = fit_classic_scaling(b1)
    cv = cross_validate_classic(b1)
    print("【实证检验】B1 参数 bootstrap（1000 次）")
    b1_boot = bootstrap_classic_scaling(b1, n_boot=1000)
    b1_boot.to_csv(OUT / "q2_b1_bootstrap_draws.csv", index=False)
    ci_rows = []
    for parameter in ["A", "B", "alpha", "beta"]:
        ci_rows.append({"parameter": parameter, "estimate": float(model.params[["A", "B", "alpha", "beta"].index(parameter)]),
                        "ci_low": float(b1_boot[parameter].quantile(.025)),
                        "ci_high": float(b1_boot[parameter].quantile(.975)),
                        "bootstrap_n": int(len(b1_boot))})
    b1_ci = pd.DataFrame(ci_rows)
    b1_ci.to_csv(OUT / "q2_b1_parameter_ci.csv", index=False)
    diag.to_csv(OUT / "q2_b1_predictions.csv", index=False)
    cv.to_csv(OUT / "q2_b1_cv_metrics.csv", index=False)
    ci_map = b1_ci.set_index("parameter").to_dict("index")
    coef_rows = [{"parameter": k, "value": v, "fit_stage": "stage2_beta_relaxed",
                  "ci_low": ci_map[k]["ci_low"], "ci_high": ci_map[k]["ci_high"]}
                 for k, v in zip(["A", "B", "alpha", "beta"], model.params)]
    coef_rows += [{"parameter": "beta_prior_B1", "value": model.beta_prior, "fit_stage": "stage1_fixed_and_stage2_penalty"},
                  {"parameter": "beta_penalty_lambda", "value": model.beta_penalty, "fit_stage": "stage2_penalty"}]
    pd.DataFrame(coef_rows).to_csv(OUT / "q2_b1_fit_coefficients.csv", index=False)
    grid, leave_n, pair_hold = b1_grid_diagnostics(b1, model)
    grid.to_csv(OUT / "q2_b1_grid_structure.csv", index=False)
    leave_n.to_csv(OUT / "q2_b1_leave_one_n.csv", index=False)
    pair_hold.to_csv(OUT / "q2_b1_group_holdout_metrics.csv", index=False)
    print(f"【实证检验通过】B1 解析网格诊断: {int(grid.n_rows.iloc[0])} 行, {int(grid.n_N.iloc[0])}x{int(grid.n_D.iloc[0])}; 无截距两阶段拟合完成")
    plt.figure(figsize=(6.2, 5)); plt.scatter(diag.val_loss, diag.predicted_loss, s=10, alpha=.35)
    lo, hi = float(diag.val_loss.min()), float(diag.val_loss.max()); plt.plot([lo, hi], [lo, hi], "k--")
    plt.xlabel("Observed B1 val_loss"); plt.ylabel("Predicted classic scaling loss"); plt.title("B1 observed versus predicted")
    _savefig(OUT / "fig_q2_b1_observed_predicted.png")
    plt.figure(figsize=(7, 4)); plt.scatter(np.log10(diag.N_params_B), diag.residual, s=9, alpha=.3)
    plt.axhline(0, color="k", lw=1); plt.xlabel("log10(parameter count B)"); plt.ylabel("Residual"); plt.title("B1 residual diagnostic"); _savefig(OUT / "fig_q2_b1_residuals.png")

    paths = {"B2_Cerebras": DATA_ROOT / "cerebras_training_log.csv",
             "B4_cross_family": DATA_ROOT / "scaling_baseline.csv",
             "B5_published": DATA_ROOT / "published_scaling_data.csv"}
    ext, ext_pred = external_validation(paths, model); ext.to_csv(OUT / "q2_scaling_validation_summary.csv", index=False)
    for name, frame in ext_pred.items(): frame.to_csv(OUT / f"q2_{name.lower()}_predictions.csv", index=False)
    b2cal = b2_recalibration(paths, model); b2cal.to_csv(OUT / "q2_b2_recalibration.csv", index=False)
    print("【实证检验通过】B2/B4/B5 外部验证与 B2 仿射重标定已完成")

    traj_paths = sorted((DATA_ROOT / "training_trajectories").glob("*.csv"))
    traj_metrics, traj_pred = trajectory_validation(traj_paths, model)
    traj_metrics.to_csv(OUT / "q2_b3_trajectory_validation.csv", index=False); traj_pred.to_csv(OUT / "q2_b3_trajectory_predictions.csv", index=False)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=False)
    for ax, (name, g) in zip(axes.ravel(), traj_pred.groupby("trajectory")):
        ax.plot(g.step, g.val_loss, lw=.8, label="observed"); ax.plot(g.step, g.predicted_loss, ls="--", lw=.8, label="predicted")
        ax.set_title(name.replace("trajectory_", ""), fontsize=8); ax.set_xlabel("step", fontsize=8); ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("val_loss"); axes[1, 0].set_ylabel("val_loss"); axes[0, 0].legend(fontsize=7)
    _savefig(OUT / "fig_q2_b3_trajectory_validation.png")
    print("【实证检验通过】B3 轨迹验证已重绘为 2x4 子图")

    q_frames = [pd.read_csv(DATA_ROOT / f) for f in ["supplementary_NQ_experiment.csv", "supplementary_NQ_experiment_expanded.csv", "supplementary_NQ_experiment_large.csv"]]
    q_all = pd.concat(q_frames, ignore_index=True).drop_duplicates(subset=["experiment_id"])
    linear_coef, linear_pred = fit_quality_effect(q_all, model)
    linear_coef.to_csv(OUT / "q2_quality_linear_diagnostic.csv", index=False)
    # Q_quality=1-Q_score and |Q_score-1| differ only by a sign convention.
    # Re-express the centered linear model in the mirrored coordinate so that
    # the identical fitted values and metrics are explicit in the deliverables.
    q0_linear = float(linear_coef["Q0_native"].iloc[0])
    coef_linear = linear_coef.set_index("coefficient")["value"]
    x_centered = q_all.Q_score.to_numpy(float) - q0_linear
    x_mirrored = (1.0 - q_all.Q_score.to_numpy(float)) - (1.0 - q0_linear)
    original_pred = (coef_linear["intercept"] + coef_linear["N_term"] * q_all.N_params_B.to_numpy(float) ** (-model.feature_exponents[0])
                     + coef_linear["D_term"] * q_all.D_tokens_B.to_numpy(float) ** (-model.feature_exponents[1])
                     + coef_linear["Q_centered"] * x_centered
                     + coef_linear["QxN_term"] * x_centered * q_all.N_params_B.to_numpy(float) ** (-model.feature_exponents[0])
                     + coef_linear["QxD_term"] * x_centered * q_all.D_tokens_B.to_numpy(float) ** (-model.feature_exponents[1]))
    mirrored_pred = (coef_linear["intercept"] + coef_linear["N_term"] * q_all.N_params_B.to_numpy(float) ** (-model.feature_exponents[0])
                     + coef_linear["D_term"] * q_all.D_tokens_B.to_numpy(float) ** (-model.feature_exponents[1])
                     - coef_linear["Q_centered"] * x_mirrored
                     - coef_linear["QxN_term"] * x_mirrored * q_all.N_params_B.to_numpy(float) ** (-model.feature_exponents[0])
                     - coef_linear["QxD_term"] * x_mirrored * q_all.D_tokens_B.to_numpy(float) ** (-model.feature_exponents[1]))
    mirror_metrics = pd.DataFrame([
        {"form": "centered Q_score", "r2": float(r2_score(q_all.val_loss, original_pred)),
         "mae": float(mean_absolute_error(q_all.val_loss, original_pred)),
         "max_abs_prediction_difference_vs_centered": 0.0},
        {"form": "mirrored |Q_score-1| = 1-Q_score", "r2": float(r2_score(q_all.val_loss, mirrored_pred)),
         "mae": float(mean_absolute_error(q_all.val_loss, mirrored_pred)),
         "max_abs_prediction_difference_vs_centered": float(np.max(np.abs(mirrored_pred - original_pred)))},
    ])
    mirror_metrics.to_csv(OUT / "q2_quality_mirror_equivalence.csv", index=False)
    q_sensitivity, q_pred, sign = fit_asymmetric_quality_effect(q_all, model)
    q_sensitivity.to_csv(OUT / "q2_quality_effect_coefficients.csv", index=False)
    q_pred.to_csv(OUT / "q2_quality_predictions.csv", index=False)
    from scipy.stats import pearsonr
    base_pred = model.params[0]*q_all.N_params_B.to_numpy(float)**(-model.params[2]) + model.params[1]*q_all.D_tokens_B.to_numpy(float)**(-model.params[3])
    lv = linear_coef.set_index("coefficient")["value"]
    old_derivative = (lv["Q_centered"] + lv["QxN_term"] * q_all.N_params_B.to_numpy(float) ** (-model.feature_exponents[0])
                      + lv["QxD_term"] * q_all.D_tokens_B.to_numpy(float) ** (-model.feature_exponents[1]))
    q_corr = pd.DataFrame([{"quantity": "raw Q_score vs val_loss", "pearson_r": pearsonr(q_all.Q_score, q_all.val_loss).statistic, "n": len(q_all)},
                           {"quantity": "Q_score vs loss residual after B1 N-D", "pearson_r": pearsonr(q_all.Q_score, q_all.val_loss-base_pred).statistic, "n": len(q_all)},
                           {"quantity": "dL/dQ_score grid minimum", "pearson_r": old_derivative.min(), "n": len(q_all)},
                           {"quantity": "dL/dQ_score grid maximum", "pearson_r": old_derivative.max(), "n": len(q_all)},
                           {"quantity": "dL/dQ_score positive share", "pearson_r": (old_derivative > 0).mean(), "n": len(q_all)}])
    q_corr.to_csv(OUT / "q2_quality_direction_diagnostics.csv", index=False)
    print(f"【实证检验通过】物理纠偏质量模型: residual r={q_corr.pearson_r.iloc[1]:.4f}, lambda=1 R2={q_sensitivity.loc[q_sensitivity['lambda']==1,'r2'].iloc[0]:.4f}")
    plt.figure(figsize=(6.5, 4.5))
    for q, g in q_pred.groupby("Q_score"):
        plt.scatter(g.N_params_B, g.val_loss, s=8, alpha=.25, label=f"Qscore={q:g}")
    plt.xscale("log"); plt.xlabel("N parameters (B)"); plt.ylabel("native B6-B8 val_loss"); plt.title("Qscore as defect rate: native quality data"); plt.legend(ncol=2, fontsize=7); _savefig(OUT / "fig_q2_quality_effect.png")
    # The low-end saturation check must use the paired large supplementary
    # table itself. Mixing one-sided Q=.1 rows from the base/expanded tables
    # with Q=.05 rows from the large table silently changes the matching set.
    low_source = pd.read_csv(DATA_ROOT / "supplementary_NQ_experiment_large.csv")
    low = (low_source[low_source.Q_score.isin([.05, .1])]
           .pivot_table(index=["N_params_B", "D_tokens_B"], columns="Q_score",
                        values="val_loss", aggfunc="first").dropna().reset_index())
    low["equal_loss"] = np.isclose(low[.05], low[.1]); low.to_csv(OUT / "q2_low_q_saturation_check.csv", index=False)

    large_metrics, large = large_scale_extrapolation(DATA_ROOT / "supplementary_large_models.csv", DATA_ROOT / "supplementary_large_baseline.csv", model)
    large_metrics.to_csv(OUT / "q2_large_scale_metrics.csv", index=False); large.to_csv(OUT / "q2_large_scale_extrapolation.csv", index=False)
    # B9/B10 are estimated scenarios.  The constant-bias correction below is
    # a same-table diagnostic, not an independently validated correction.
    residual = pd.to_numeric(large["residual"], errors="coerce").dropna().to_numpy(float)
    rng = np.random.default_rng(20260925)
    draws = np.array([rng.choice(residual, len(residual), replace=True).mean() for _ in range(4000)])
    bias = float(residual.mean())
    corrected = large.copy()
    corrected["bias_corrected_prediction_diagnostic"] = corrected["predicted_loss"] + bias
    corrected["bias_corrected_residual_diagnostic"] = corrected["val_loss"] - corrected["bias_corrected_prediction_diagnostic"]
    corrected[["family", "N_params_B", "D_tokens_B", "val_loss", "predicted_loss", "residual",
               "bias_corrected_prediction_diagnostic", "bias_corrected_residual_diagnostic"]].to_csv(
        OUT / "q2_large_scale_bias_sensitivity.csv", index=False, encoding="utf-8-sig")
    bias_summary = pd.DataFrame([{
        "n_estimated_rows": len(residual), "mean_residual_nats": bias,
        "bias_ci_low_nats": float(np.quantile(draws, .025)),
        "bias_ci_high_nats": float(np.quantile(draws, .975)),
        "mae_uncorrected_nats": float(np.mean(np.abs(residual))),
        "mae_bias_corrected_same_table_diagnostic_nats": float(np.mean(np.abs(residual-bias))),
        "r2_uncorrected": float(r2_score(large["val_loss"], large["predicted_loss"])),
        "r2_bias_corrected_same_table_diagnostic": float(r2_score(large["val_loss"], large["predicted_loss"]+bias)),
        "data_status": "estimated B9/B10 scenarios; same-table bias correction is descriptive, not independent validation"
    }])
    bias_summary.to_csv(OUT / "q2_large_scale_bias_summary.csv", index=False, encoding="utf-8-sig")
    print(f"【实证检验通过】B9/B10 估算表残差偏差={bias:.4f} nats, bootstrap 95% CI "
          f"[{np.quantile(draws, .025):.4f}, {np.quantile(draws, .975):.4f}]；同表校正仅作诊断")
    print("【边界声明】B9/B10 为超百亿规模外推情景工具，不作为独立实验验证")
    plt.figure(figsize=(7, 4)); plt.scatter(large.N_params_B, large.val_loss, s=12, alpha=.45, label="estimated observed"); plt.scatter(large.N_params_B, large.predicted_loss, s=12, alpha=.45, label="classic prediction"); plt.xscale("log"); plt.xlabel("N parameters (B)"); plt.ylabel("loss"); plt.title("Large-scale scenario extrapolation"); plt.legend(); _savefig(OUT / "fig_q2_large_scale_extrapolation.png")

    qp = q_sensitivity[q_sensitivity["lambda"] == 1].iloc[0][["A", "B", "gamma", "alpha", "beta", "theta"]].to_numpy(float)
    mrt = mrtS_table(model, qp, q_quality=.65, q_step=.1); mrt.to_csv(OUT / "q2_mrts_equivalence_table.csv", index=False)
    summarize_mrts(mrt).to_csv(OUT / "q2_mrts_summary.csv", index=False)
    mrt.to_csv(OUT / "q2_elasticities.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    grids = [("elasticity_N", "Parameter elasticity"), ("elasticity_D", "Data elasticity"), ("marginal_quality", "Quality marginal utility")]
    # Add data elasticity and positive quality utility fields for the plot/table.
    A,B,gamma,alpha,beta,theta=qp
    mrt["elasticity_D"] = (-beta*B*mrt.D_tokens_B.to_numpy(float)**(-beta))/mrt.loss.to_numpy(float)
    mrt["marginal_quality"] = -mrt.dL_dQ_quality
    mrt.to_csv(OUT / "q2_elasticities.csv", index=False)
    for ax, (col, title) in zip(axes, grids):
        pivot = mrt.assign(z=mrt[col]).pivot_table(index="N_params_B", columns="D_tokens_B", values="z")
        ax.imshow(pivot.values, aspect="auto", cmap="coolwarm"); ax.set_title(title, fontsize=9); ax.set_xlabel("D"); ax.set_ylabel("N"); ax.tick_params(labelsize=6)
    _savefig(OUT / "fig_q2_elasticity_heatmap.png")

    bridge_path = PROJECT_ROOT / "outputs" / "q1" / "q2_input_interface_bundle.json"
    bridge = json.loads(bridge_path.read_text(encoding="utf-8")) if bridge_path.exists() else {}
    qstar = bridge.get("Q_star_17", {}); tvec = bridge.get("first_order_substitution_vector_t", {})
    bridge_rows = pd.DataFrame({"domain": list(qstar), "Q_star": [qstar[k] for k in qstar], "t_first_order": [tvec.get(k, np.nan) for k in qstar], "Q_status": [bridge.get("Q_star_status", {}).get(k, "unknown") for k in qstar]})
    bridge_rows.to_csv(OUT / "q2_domain_substitution_effects.csv", index=False)
    second_order_summary(bridge).to_csv(OUT / "q2_second_order_nonadditivity_summary.csv", index=False)
    affine_bridge_test(PROJECT_ROOT / "data/raw/real_attachments/C_efficiency_evolution/loss_benchmark_bridge_expanded.csv").to_csv(OUT / "q2_affine_bridge_test_results.csv", index=False)
    _deidentified_a18().to_csv(OUT / "q2_A18_deidentified_domain_examples.csv", index=False)
    pd.DataFrame([{"item": "B1", "source": "real Pythia log", "used_as": "complete analytical N-D grid; structured diagnostics"}, {"item": "B2", "source": "Cerebras", "used_as": "external ranking plus affine recalibration"}, {"item": "B3", "source": "interpolated trajectories", "used_as": "same-family validation"}, {"item": "B4/B5", "source": "cross-family/literature", "used_as": "ranking validation"}, {"item": "B6-B8", "source": "native Q_score experiments", "used_as": "physical asymmetric defect-rate effect"}, {"item": "B9/B10", "source": "estimated large-scale tables", "used_as": "scenario sensitivity only"}, {"item": "A interface", "source": "Question 1 bridge", "used_as": "Q_bar(p), ILR substitution and conditional second-order effects"}]).to_csv(OUT / "q2_input_usage_boundary.csv", index=False)
    print("【实证检验通过】MRTS、仿射桥接、65条非加性摘要、A18脱敏与全部图表已输出")
    return {"classic": model, "quality_sensitivity": q_sensitivity, "external": ext, "trajectory": traj_metrics, "large": large_metrics}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(); run()


if __name__ == "__main__":
    main()
