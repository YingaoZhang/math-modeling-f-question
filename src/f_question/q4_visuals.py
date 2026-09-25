"""Publication figures for Question 4."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "savefig.dpi": 300})


def bridge_plot(residuals: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(residuals.Val_Loss, residuals.LB_Average, s=45, color="#2A6F97")
    lo, hi = residuals.Val_Loss.min(), residuals.Val_Loss.max()
    x = np.linspace(lo, hi, 100)
    b = np.polyfit(residuals.Val_Loss, residuals.LB_Average, 1)
    ax.plot(x, b[0]*x+b[1], color="#C1443E", lw=2)
    ax.set(xlabel="Pythia validation loss", ylabel="Leaderboard average", title="High-comparability loss benchmark bridge")
    fig.tight_layout(); path = out / "fig_q4_loss_benchmark_bridge.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); return path


def decomposition_plot(decomp: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(decomp.scale_contribution, decomp.time_contribution, s=16, alpha=.45, color="#386641")
    ax.axhline(0, color="0.6", lw=.8); ax.axvline(0, color="0.6", lw=.8)
    ax.set(xlabel="Scale contribution (benchmark points)", ylabel="Time contribution (benchmark points)",
           title="Pretrained leaderboard: scale versus non-scale contribution")
    fig.tight_layout(); path = out / "fig_q4_scale_nonscale_decomposition.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); return path


def forecast_plot(pred: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for h, g in pred.groupby("horizon_months"):
        g = g.sort_values("annual_compute_growth")
        ax.plot(g.annual_compute_growth, g.benchmark_predicted, marker="o", label=f"{h} months")
        ax.fill_between(g.annual_compute_growth, g.bridge_bootstrap_95ci_low,
                        g.bridge_bootstrap_95ci_high, alpha=.18)
    ax.set(xlabel="Annual compute growth factor", ylabel="Predicted benchmark average", title="Compute frontier scenario forecast")
    ax.legend(frameon=False); ax.grid(alpha=.2)
    fig.tight_layout(); path = out / "fig_q4_frontier_forecast.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); return path


def coverage_plot(epoch: pd.DataFrame, out: Path) -> Path:
    x = np.sort(epoch.compute_flop.to_numpy(float))
    y = np.arange(1, len(x)+1) / len(x)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, color="#5E548E")
    ax.set_xscale("log"); ax.set(xlabel="Training compute (FLOP)", ylabel="Empirical cumulative fraction", title="C4 compute coverage")
    ax.grid(alpha=.2)
    fig.tight_layout(); path = out / "fig_q4_data_coverage.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); return path


def task_heterogeneity_plot(task: pd.DataFrame, out: Path) -> Path:
    """Plot adjusted C3 annual slopes by evaluation task."""
    d = task.sort_values("year_slope_adjusted_points_per_year")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = np.where(d.year_slope_adjusted_points_per_year >= 0, "#2A7F62", "#C14B42")
    ax.barh(d.task, d.year_slope_adjusted_points_per_year, color=colors)
    ax.axvline(0, color="0.35", lw=.8)
    ax.set(xlabel="Year slope adjusted for log10(parameters) (points/year)",
           title="C3 task-level performance trends")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout(); path = out / "fig_q4_task_heterogeneity.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); return path


def envelope_plot(annual: pd.DataFrame, forecast: pd.DataFrame, out: Path) -> Path:
    """Plot observed envelopes and point trend scenarios.

    The current C3 annual fit has only three usable years (df=1). Its t
    interval is non-identifiable, so it is deliberately not filled as a
    confidence band.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(annual.Year, annual.p90, "o-", label="Observed p90", color="#2A6F97")
    ax.plot(annual.Year, annual.maximum, "o-", label="Observed max", color="#C1443E")
    for metric, color, label in [("p90", "#2A6F97", "p90 prediction"),
                                 ("maximum", "#C1443E", "max prediction")]:
        f = forecast[forecast.metric.eq(metric)].sort_values("forecast_year")
        if len(f):
            ax.plot(f.forecast_year, f.estimate, "--", color=color, label=label)
            low = pd.to_numeric(f["prediction_low_95"], errors="coerce")
            high = pd.to_numeric(f["prediction_high_95"], errors="coerce")
            if low.notna().all() and high.notna().all():
                ax.fill_between(f.forecast_year, low, high, color=color, alpha=.14)
    ax.text(0.02, 0.04, "C3 年度点仅 3 个，df=1 的 t 区间未作为有效预测带绘制",
            transform=ax.transAxes, fontsize=8, color="#555555")
    ax.set(xlabel="C3 publication year", ylabel="Leaderboard Average",
           title="Observed C3 frontier envelope and point trend scenarios", ylim=(0, 100))
    ax.legend(frameon=False); ax.grid(alpha=.2)
    fig.tight_layout(); path = out / "fig_q4_c3_empirical_envelope.png"; fig.savefig(path, bbox_inches="tight"); plt.close(fig); return path
