"""Parameter uncertainty propagation for the Question 3 scenario model.

The propagation is deliberately separated from the main grid: it fixes the
exponential cost and the shortest C7 context so that intervals describe
coefficient uncertainty rather than a mixture of scenario choices.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .q3_data import Q3Inputs
from .q3_models import COST_FUNCTIONS, compute_costs, predicted_loss


def _intervals(root: Path, inputs: Q3Inputs, source: str) -> dict[str, tuple[float, float]]:
    """Return conservative 95% parameter intervals for one paired source."""
    c = inputs.classic_by_source[source]
    out = {k: (float(v * .95), float(v * 1.05)) for k, v in c.items()}
    ci = root / "outputs" / "q2" / "q2_b1_parameter_ci.csv"
    if source == "b1" and ci.exists():
        d = pd.read_csv(ci).set_index("parameter")
        for k in c:
            if k in d.index:
                out[k] = (float(d.loc[k, "ci_low"]), float(d.loc[k, "ci_high"]))
    q = inputs.quality_by_source[source]
    out.update({k: (float(v * .95), float(v * 1.05)) for k, v in q.items()})
    quality_csv = root / "outputs" / "q2" / "q2_quality_effect_coefficients.csv"
    if quality_csv.exists():
        d = pd.read_csv(quality_csv)
        row = d.loc[np.isclose(d["lambda"], 10.0)].iloc[0]
        for k, lo, hi in [("gamma", "gamma_ci_low", "gamma_ci_high"),
                          ("theta", "theta_ci_low", "theta_ci_high")]:
            if pd.notna(row.get(lo)) and pd.notna(row.get(hi)):
                out[k] = (float(row[lo]), float(row[hi]))
    return out


def run_uncertainty(root: Path, inputs: Q3Inputs, out: Path,
                    n_draws: int = 200, budgets: tuple[float, ...] = (.01, 10.0, 1000.0),
                    seed: int = 20260924) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw paired six-parameter sets and propagate them to allocations."""
    rng = np.random.default_rng(seed)
    rows = []
    for source in ["b6_b8_lambda10", "b1"]:
        bounds = _intervals(root, inputs, source)
        for draw in range(int(n_draws)):
            classic = {k: float(max(1e-8, rng.uniform(*bounds[k]))) for k in inputs.classic_by_source[source]}
            quality = {k: float(max(1e-8, rng.uniform(*bounds[k]))) for k in inputs.quality_by_source[source]}
            local = replace(inputs, classic=classic, quality=quality)
            for budget in budgets:
                solved = _fixed_mix_solve(local, budget, int(min(inputs.context_values)), inputs.p0)
                solved.update({"coefficient_source": source, "bootstrap_draw": draw,
                               "context_tokens_fixed": int(min(inputs.context_values))})
                rows.append(solved)
    samples = pd.DataFrame(rows)
    value_cols = ["N_params_B", "D_tokens_B", "Q_target", "train_cost_share",
                  "quality_cost_share", "attention_cost_share", "loss"]
    grouped = samples.groupby(["coefficient_source", "budget_1e21"])
    interval_rows = []
    for keys, g in grouped:
        source, budget = keys
        row = {"coefficient_source": source, "budget_1e21": budget,
               "n_draws": int(g.bootstrap_draw.nunique())}
        for col in value_cols:
            row[f"{col}_median"] = float(g[col].median())
            row[f"{col}_ci_low"] = float(g[col].quantile(.025))
            row[f"{col}_ci_high"] = float(g[col].quantile(.975))
        interval_rows.append(row)
    intervals = pd.DataFrame(interval_rows)
    samples.to_csv(out / "q3_uncertainty_samples.csv", index=False, encoding="utf-8-sig")
    intervals.to_csv(out / "q3_uncertainty_intervals.csv", index=False, encoding="utf-8-sig")
    _plot_intervals(intervals, out / "fig_q3_uncertainty_bands.png")
    return samples, intervals


def _fixed_mix_solve(inputs: Q3Inputs, budget: float, context: int, p: np.ndarray) -> dict[str, float]:
    """Fast three-variable solve used only for parameter interval propagation."""
    n_lo, n_hi = inputs.n_bounds_b; d_lo, d_hi = inputs.d_bounds_b
    fn = COST_FUNCTIONS[0]
    def decode(x):
        return float(np.exp(x[0])), float(np.exp(x[1])), float(x[2])
    def obj(x):
        n, d, q = decode(x)
        return predicted_loss(n, d, q, p, inputs)["loss"]
    def feasible(x):
        n, d, q = decode(x)
        return budget - compute_costs(n, d, q, context, fn, inputs.q0)["total_cost_1e21"]
    x0 = np.array([np.log(np.sqrt(n_lo*n_hi)), np.log(np.sqrt(d_lo*d_hi)), min(.9, .5*(inputs.q0+.95))])
    res = minimize(obj, x0, method="SLSQP",
                   bounds=[(np.log(n_lo), np.log(n_hi)), (np.log(d_lo), np.log(d_hi)), (inputs.q0, .95)],
                   constraints=[{"type": "ineq", "fun": feasible}],
                   options={"maxiter": 160, "ftol": 1e-9, "disp": False})
    n, d, q = decode(res.x)
    pred = predicted_loss(n, d, q, p, inputs)
    costs = compute_costs(n, d, q, context, fn, inputs.q0)
    total = max(costs["total_cost_1e21"], 1e-12)
    return {"budget_1e21": budget, "N_params_B": n, "D_tokens_B": d, "Q_target": q,
            "loss": pred["loss"], **costs,
            "train_cost_share": costs["train_cost_1e21"]/total,
            "quality_cost_share": costs["quality_cost_1e21"]/total,
            "attention_cost_share": costs["attention_cost_1e21"]/total}


def _plot_intervals(intervals: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for source, g in intervals.groupby("coefficient_source"):
        g = g.sort_values("budget_1e21")
        x = np.arange(len(g))
        label = "B6-B8 λ=10" if source == "b6_b8_lambda10" else "B1 对照"
        for ax, col, title in zip(axes, ["N_params_B", "D_tokens_B", "Q_target"], ["参数量 N (B)", "数据量 D (B tokens)", "质量目标 Q"]):
            mid, lo, hi = [g[f"{col}_{s}"].to_numpy() for s in ("median", "ci_low", "ci_high")]
            ax.plot(x, mid, marker="o", label=label)
            ax.fill_between(x, lo, hi, alpha=.18)
            ax.set_xticks(x, [f"{v:g}" for v in g.budget_1e21])
            ax.set_title(title); ax.grid(alpha=.25)
    axes[0].set_ylabel("估计值与95%区间"); axes[1].set_ylabel("估计值与95%区间"); axes[2].set_ylabel("估计值与95%区间")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
