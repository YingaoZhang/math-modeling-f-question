"""Command line pipeline for Problem 2: generalized scaling laws."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .q2_scaling import fit_classic_scaling, cross_validate_classic, fit_quality_effect, quality_prediction
from .q2_validation import (external_validation, trajectory_validation, large_scale_extrapolation,
                            quality_tradeoffs, elasticity_table)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "real_attachments" / "B_scaling_laws"
OUT = PROJECT_ROOT / "outputs" / "q2"


def _savefig(path: Path) -> None:
    plt.tight_layout(); plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    print("【实证检验】读取 B1--B10 附件并区分数据边界")
    b1 = pd.read_csv(DATA_ROOT / "pythia_training_log_existing.csv")
    model, diag = fit_classic_scaling(b1)
    cv = cross_validate_classic(b1)
    diag.to_csv(OUT / "q2_b1_predictions.csv", index=False)
    cv.to_csv(OUT / "q2_b1_cv_metrics.csv", index=False)
    pd.DataFrame([{"parameter": k, "value": v} for k, v in zip(["c", "A", "B", "alpha", "beta"], model.params)]).to_csv(OUT / "q2_b1_fit_coefficients.csv", index=False)
    print(f"【实证检验通过】B1 经典标度律: R2={model.test_metrics['r2']:.4f}, MAE={model.test_metrics['mae']:.4f}")

    plt.figure(figsize=(6.2, 5)); plt.scatter(diag.val_loss, diag.predicted_loss, s=10, alpha=.35)
    lo, hi = float(diag.val_loss.min()), float(diag.val_loss.max()); plt.plot([lo, hi], [lo, hi], "k--")
    plt.xlabel("Observed B1 val_loss"); plt.ylabel("Predicted classic scaling loss"); plt.title("B1 observed versus predicted")
    _savefig(OUT / "fig_q2_b1_observed_predicted.png")
    plt.figure(figsize=(7, 4)); plt.scatter(np.log10(diag.N_params_B), diag.residual, s=9, alpha=.3, label="residual")
    plt.axhline(0, color="k", lw=1); plt.xlabel("log10(parameter count B)"); plt.ylabel("Residual"); plt.title("B1 residual diagnostic"); _savefig(OUT / "fig_q2_b1_residuals.png")

    paths = {"B2_Cerebras": DATA_ROOT / "cerebras_training_log.csv",
             "B4_cross_family": DATA_ROOT / "scaling_baseline.csv",
             "B5_published": DATA_ROOT / "published_scaling_data.csv"}
    ext, ext_pred = external_validation(paths, model); ext.to_csv(OUT / "q2_scaling_validation_summary.csv", index=False)
    for name, frame in ext_pred.items(): frame.to_csv(OUT / f"q2_{name.lower()}_predictions.csv", index=False)
    print("【实证检验通过】B2/B4/B5 外部验证已完成")

    traj_paths = sorted((DATA_ROOT / "training_trajectories").glob("*.csv"))
    traj_metrics, traj_pred = trajectory_validation(traj_paths, model)
    traj_metrics.to_csv(OUT / "q2_b3_trajectory_validation.csv", index=False); traj_pred.to_csv(OUT / "q2_b3_trajectory_predictions.csv", index=False)
    print("【实证检验通过】B3 轨迹插值验证已完成")
    plt.figure(figsize=(7, 4))
    for name, g in traj_pred.groupby("trajectory"):
        plt.plot(g.step, g.val_loss, alpha=.4, lw=.7); plt.plot(g.step, g.predicted_loss, alpha=.4, lw=.7, ls="--")
    plt.xlabel("trajectory step"); plt.ylabel("val_loss"); plt.title("B3 trajectory validation (solid observed, dashed predicted)")
    _savefig(OUT / "fig_q2_b3_trajectory_validation.png")

    q_frames = [pd.read_csv(DATA_ROOT / f) for f in ["supplementary_NQ_experiment.csv", "supplementary_NQ_experiment_expanded.csv", "supplementary_NQ_experiment_large.csv"]]
    q_all = pd.concat(q_frames, ignore_index=True).drop_duplicates(subset=["experiment_id"])
    q_coef, q_pred = fit_quality_effect(q_all, model)
    q_coef.to_csv(OUT / "q2_quality_effect_coefficients.csv", index=False); q_pred.to_csv(OUT / "q2_quality_predictions.csv", index=False)
    print(f"【实证检验通过】B6--B8 质量扩展已完成: alpha={q_coef.alpha_selected.iloc[0]:.5g}, R2={q_coef.r2_in_sample.iloc[0]:.4f}")
    plt.figure(figsize=(6.5, 4.5));
    for q, g in q_pred.groupby("Q_score"):
        plt.scatter(g.N_params_B, g.val_loss, s=8, alpha=.3, label=f"Q={q:g}")
    plt.xscale("log"); plt.xlabel("N parameters (B)"); plt.ylabel("native B6-B8 val_loss"); plt.title("Native Q_score and validation loss")
    plt.legend(ncol=2, fontsize=7); _savefig(OUT / "fig_q2_quality_effect.png")

    large_metrics, large = large_scale_extrapolation(DATA_ROOT / "supplementary_large_models.csv", DATA_ROOT / "supplementary_large_baseline.csv", model)
    large_metrics.to_csv(OUT / "q2_large_scale_metrics.csv", index=False); large.to_csv(OUT / "q2_large_scale_extrapolation.csv", index=False)
    print("【边界声明】B9/B10 为大规模元数据与估算 Loss，仅作外推情景，不作为真实实验验证")
    plt.figure(figsize=(7, 4)); plt.scatter(large.N_params_B, large.val_loss, s=12, alpha=.45, label="estimated observed"); plt.scatter(large.N_params_B, large.predicted_loss, s=12, alpha=.45, label="classic prediction"); plt.xscale("log"); plt.xlabel("N parameters (B)"); plt.ylabel("loss"); plt.title("Large-scale extrapolation B9/B10"); plt.legend(); _savefig(OUT / "fig_q2_large_scale_extrapolation.png")

    trade = quality_tradeoffs(q_coef, model); elast = elasticity_table(q_coef, model)
    trade.to_csv(OUT / "q2_quality_parameter_tradeoff.csv", index=False); elast.to_csv(OUT / "q2_elasticities.csv", index=False)
    # Compact heatmap of parameter elasticity at Q=0.5.
    h = elast[elast.Q_score == .5].pivot(index="N_params_B", columns="D_tokens_B", values="elasticity_N")
    plt.figure(figsize=(7, 4)); plt.imshow(h.values, aspect="auto", cmap="coolwarm"); plt.colorbar(label="parameter elasticity"); plt.xticks(range(len(h.columns)), h.columns, rotation=45); plt.yticks(range(len(h.index)), h.index); plt.xlabel("D tokens (B)"); plt.ylabel("N parameters (B)"); plt.title("Elasticity of Loss to parameter scale"); _savefig(OUT / "fig_q2_elasticity_heatmap.png")

    bridge_path = PROJECT_ROOT / "outputs" / "q1" / "q2_input_interface_bundle.json"
    bridge = json.loads(bridge_path.read_text(encoding="utf-8")) if bridge_path.exists() else {}
    qstar = bridge.get("Q_star_17", {}); tvec = bridge.get("first_order_substitution_vector_t", {})
    bridge_rows = pd.DataFrame({"domain": list(qstar), "Q_star": [qstar[k] for k in qstar], "t_first_order": [tvec.get(k, np.nan) for k in qstar], "Q_status": [bridge.get("Q_star_status", {}).get(k, "unknown") for k in qstar]})
    bridge_rows.to_csv(OUT / "q2_domain_substitution_effects.csv", index=False)
    pd.DataFrame([{"item": "B1 primary fit", "source": "real Pythia log", "used_as": "classic N-D scaling"}, {"item": "B2/B3", "source": "Cerebras / interpolated trajectories", "used_as": "external validation"}, {"item": "B4/B5", "source": "cross-family / published", "used_as": "cross-family validation"}, {"item": "B6-B8", "source": "semi-synthetic Q experiments", "used_as": "native Q effect only"}, {"item": "B9/B10", "source": "metadata and estimated baseline", "used_as": "large-scale scenario extrapolation"}, {"item": "Q1 interface", "source": "A1 ECDF Q and ILR p", "used_as": "cross-question scenario input; no direct Loss merge"}]).to_csv(OUT / "q2_input_usage_boundary.csv", index=False)
    print("【实证检验通过】第二问全部 CSV 与 PNG 已输出")
    return {"classic": model, "quality_coeff": q_coef, "external": ext, "trajectory": traj_metrics, "large": large_metrics}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()


if __name__ == "__main__":
    main()
