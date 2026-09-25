"""Cost functions, loss surrogate, and constrained optimization for Question 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .q3_data import Q3Inputs


ETA_ATTN = 2.0e-4
# D is represented in billion tokens and the reported budget in 1e21 FLOPs.
# Thus D_tokens * delta_g / 1e21 = D_billion * delta_g * 1e-12.
QUALITY_COST_SCALE = 1.0
MIX_EFFECT_SCALE = 0.08
MIX_KL_PENALTY = 0.03
Q_UPPER = 0.95
P_LOWER = 0.001


@dataclass(frozen=True)
class CostFunction:
    """Quality cost function from Appendix B with explicit units."""

    name: str
    gamma: float
    lam: float
    kind: str

    def raw(self, q: np.ndarray | float) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if self.kind == "exponential":
            return self.gamma * np.exp(self.lam * q)
        if self.kind == "power":
            return self.gamma * np.power(q, self.lam)
        if self.kind == "logarithmic":
            return self.gamma * np.log1p(self.lam * q)
        raise ValueError(self.kind)

    def relative_increment(self, q: np.ndarray | float, q0: float) -> np.ndarray:
        """Return [g(q)-g(q0)]/[g(1)-g(q0)], clipped at zero."""
        denom = float(self.raw(1.0) - self.raw(q0))
        value = (self.raw(q) - self.raw(q0)) / denom
        return np.maximum(np.asarray(value, dtype=float), 0.0)

    def absolute_increment_1e21(self, q: np.ndarray | float, q0: float) -> np.ndarray:
        """Convert the Appendix-B increment D[g(Q)-g(Q0)] to 1e21 FLOPs.

        The conversion uses D in billion tokens, so the factor is 1e-12.
        ``quality_cost_scale`` remains available for hardware calibration
        sensitivity around the literal Appendix-B unit conversion.
        """
        diff = self.raw(q) - self.raw(q0)
        return np.maximum(np.asarray(diff, dtype=float), 0.0) * 1.0e-12


COST_FUNCTIONS = (
    CostFunction("exponential", 1.0e7, 6.0, "exponential"),
    CostFunction("power", 5.0e9, 4.0, "power"),
    CostFunction("logarithmic", 2.0e9, 10.0, "logarithmic"),
)


def simplex_row(p: np.ndarray) -> np.ndarray:
    """Close a composition row to the simplex."""
    p = np.maximum(np.asarray(p, dtype=float), 1e-12)
    p = p / p.sum()
    # Keep every domain represented while preserving an exact unit sum.
    return P_LOWER + (1.0 - len(p) * P_LOWER) * p


def softmax_logits(z: np.ndarray) -> np.ndarray:
    """Map 16 free logits plus a zero reference to a 17-part composition."""
    full = np.r_[np.asarray(z, dtype=float), 0.0]
    full -= np.max(full)
    e = np.exp(full)
    return P_LOWER + (1.0 - (len(full)) * P_LOWER) * (e / e.sum())


def decode_x(x: np.ndarray, inputs: Q3Inputs) -> tuple[float, float, float, np.ndarray]:
    """Decode an optimizer vector into N (B), D (B), Q and p."""
    n_b, d_b = np.exp(x[0]), np.exp(x[1])
    return float(n_b), float(d_b), float(x[2]), softmax_logits(x[3:])


def effective_quality(q_target: float, q_bar: float, q0: float) -> float:
    """Combine the global processing target with the Q1 composition quality.

    The multiplicative defect adjustment equals the target when p has the
    baseline quality q0.  It is deliberately clipped because Q1's 14 median
    imputations make Q_bar a weak, uncertainty-bearing interface.
    """
    q = 1.0 - (1.0 - q_target) * q0 / max(q_bar, 1e-8)
    return float(np.clip(q, 0.05, 0.99))


def compute_costs(n_b: float, d_b: float, q_target: float, context: int,
                  cost_fn: CostFunction, q0: float,
                  quality_cost_scale: float = QUALITY_COST_SCALE) -> dict[str, float]:
    """Compute costs in units of 1e21 FLOPs.

    With N and D measured in billions, 6ND gives 0.006ND in 1e21 FLOPs.
    The quality term uses Appendix B's absolute increment.  The factor 1e-12
    inside ``absolute_increment_1e21`` converts billion-token D to the
    reported 1e21-FLOP unit; ``quality_cost_scale`` is a sensitivity multiplier.
    """
    train = 0.006 * n_b * d_b
    attn = train * ETA_ATTN * context / 6.0
    quality = quality_cost_scale * d_b * float(cost_fn.absolute_increment_1e21(q_target, q0))
    return {"train_cost_1e21": float(train), "quality_cost_1e21": float(quality),
            "attention_cost_1e21": float(attn), "total_cost_1e21": float(train + quality + attn)}


def predicted_loss(n_b: float, d_b: float, q_target: float, p: np.ndarray,
                   inputs: Q3Inputs, mix_effect_scale: float = MIX_EFFECT_SCALE,
                   mix_kl_penalty: float = MIX_KL_PENALTY) -> dict[str, float]:
    """Evaluate the Q2-derived loss surrogate and its composition terms."""
    c = inputs.classic
    q_bar = float(np.dot(p, inputs.q_star))
    q_eff = effective_quality(q_target, q_bar, inputs.q0)
    classic = c["A"] * n_b ** (-c["alpha"]) + c["B"] * d_b ** (-c["beta"])
    qterm = inputs.quality["gamma"] * (1.0 - q_eff) ** inputs.quality["theta"]
    mix_linear = mix_effect_scale * float(np.dot(inputs.t_vector, p - inputs.p0))
    kl = float(np.sum(p * np.log(np.maximum(p, 1e-12) / inputs.p0)))
    mix_regularized = mix_linear + mix_kl_penalty * kl
    return {"loss": float(classic + qterm + mix_regularized),
            "classic_loss": float(classic), "quality_loss": float(qterm),
            "mix_linear_loss": float(mix_linear), "mix_kl_penalty": float(mix_kl_penalty * kl),
            "Q_bar": q_bar, "Q_effective": q_eff, "composition_KL": kl}


def _initial_vector(inputs: Q3Inputs, n_b: float, d_b: float, q: float,
                    p: np.ndarray | None = None) -> np.ndarray:
    """Create a stable SLSQP initial point."""
    p = inputs.p0 if p is None else simplex_row(p)
    z = np.log(np.maximum(p[:-1], 1e-10) / max(p[-1], 1e-10))
    z = np.clip(z, -7.5, 7.5)
    return np.r_[np.log(n_b), np.log(d_b), q, z]


def optimize_scenario(inputs: Q3Inputs, budget_1e21: float, context: int,
                      cost_fn: CostFunction, seed: int = 42,
                      quality_cost_scale: float = QUALITY_COST_SCALE,
                      random_starts: int = 2, maxiter: int = 700,
                      mix_effect_scale: float = MIX_EFFECT_SCALE,
                      mix_kl_penalty: float = MIX_KL_PENALTY) -> tuple[dict[str, float], np.ndarray]:
    """Minimize surrogate Loss subject to a compute budget."""
    rng = np.random.default_rng(seed + int(context) + int(budget_1e21 * 100))
    n_lo, n_hi = inputs.n_bounds_b
    d_lo, d_hi = inputs.d_bounds_b
    logit0 = np.log(np.maximum(inputs.p0[:-1], 1e-10) / max(inputs.p0[-1], 1e-10))
    bounds = [(np.log(n_lo), np.log(n_hi)), (np.log(d_lo), np.log(d_hi)),
              (inputs.q0, Q_UPPER)] + [(-8.0, 8.0)] * 16

    def objective(x: np.ndarray) -> float:
        n_b, d_b, q, p = decode_x(x, inputs)
        return predicted_loss(n_b, d_b, q, p, inputs, mix_effect_scale, mix_kl_penalty)["loss"]

    def feasible(x: np.ndarray) -> float:
        n_b, d_b, q, _ = decode_x(x, inputs)
        return budget_1e21 - compute_costs(n_b, d_b, q, context, cost_fn, inputs.q0,
                                           quality_cost_scale)["total_cost_1e21"]

    starts = [
        _initial_vector(inputs, n_lo * 1.5, d_lo * 1.5, inputs.q0),
        _initial_vector(inputs, min(1.0, n_hi), min(100.0, d_hi), min(.8, Q_UPPER)),
        _initial_vector(inputs, min(3.0, n_hi), min(300.0, d_hi), min(.9, Q_UPPER)),
    ]
    for _ in range(int(random_starts)):
        p = simplex_row(inputs.p0 * np.exp(rng.normal(0, .6, len(inputs.p0))))
        starts.append(_initial_vector(inputs, np.exp(rng.uniform(np.log(n_lo), np.log(min(n_hi, 30)))),
                                     np.exp(rng.uniform(np.log(d_lo), np.log(min(d_hi, 1000)))),
                                     float(rng.uniform(inputs.q0, Q_UPPER)), p))

    solutions = []
    for x0 in starts:
        result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                          constraints=[{"type": "ineq", "fun": feasible}],
                          options={"maxiter": maxiter, "ftol": 1e-10, "disp": False})
        n_b, d_b, q, p = decode_x(result.x, inputs)
        costs = compute_costs(n_b, d_b, q, context, cost_fn, inputs.q0, quality_cost_scale)
        pred = predicted_loss(n_b, d_b, q, p, inputs, mix_effect_scale, mix_kl_penalty)
        violation = max(0.0, costs["total_cost_1e21"] - budget_1e21)
        solutions.append((violation, pred["loss"], result, n_b, d_b, q, p, costs, pred))
    feasible_solutions = [s for s in solutions if s[0] <= max(1e-7, budget_1e21 * 1e-5)]
    chosen = min(feasible_solutions or solutions, key=lambda s: (s[0], s[1]))
    violation, _, result, n_b, d_b, q, p, costs, pred = chosen
    total = max(costs["total_cost_1e21"], 1e-12)
    row = {"budget_1e21": budget_1e21, "budget_flops": budget_1e21 * 1e21,
           "context_tokens": context, "cost_function": cost_fn.name,
           "quality_cost_scale": quality_cost_scale,
           "optimizer_success": bool(result.success), "constraint_violation_1e21": violation,
           "N_params_B": n_b, "D_tokens_B": d_b, "Q_target": q,
           **pred, **costs,
           "train_cost_share": costs["train_cost_1e21"] / total,
           "quality_cost_share": costs["quality_cost_1e21"] / total,
           "attention_cost_share": costs["attention_cost_1e21"] / total,
           "N_to_D_ratio": n_b / d_b,
           "p_top_domain": inputs.domain_order[int(np.argmax(p))],
           "p_top_share": float(np.max(p)),
           "p_min": float(np.min(p)),
           "mix_effect_scale": mix_effect_scale,
           "mix_kl_weight": mix_kl_penalty,
           "message": str(result.message)}
    return row, p


def run_optimization_grid(inputs: Q3Inputs, budgets_1e21: tuple[float, ...] = (.01, 10.0, 1000.0),
                          source: str = "b6_b8_lambda10", boundary_scheme: str = "legacy",
                          random_starts: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Solve all budget/context/cost-function scenarios."""
    rows, mix_rows = [], []
    for fn in COST_FUNCTIONS:
        for budget in budgets_1e21:
            for context in inputs.context_values:
                row, p = optimize_scenario(inputs, budget, context, fn, random_starts=random_starts)
                row["coefficient_source"] = source
                row["boundary_scheme"] = boundary_scheme
                rows.append(row)
                mix_rows.extend({"budget_1e21": budget, "budget_flops": budget * 1e21,
                                 "context_tokens": context, "cost_function": fn.name,
                                 "domain": d, "share": float(v), "coefficient_source": source,
                                 "boundary_scheme": boundary_scheme,
                                 "p0": float(inputs.p0[i]), "share_multiple": float(v / inputs.p0[i])}
                                for i, (d, v) in enumerate(zip(inputs.domain_order, p)))
    all_results = pd.DataFrame(rows)
    # For each budget and cost function, the context is selected by predicted Loss.
    best = (all_results.sort_values(["budget_1e21", "cost_function", "loss"])
            .groupby(["budget_1e21", "cost_function"], as_index=False).first())
    best["selected_context_reason"] = "lowest surrogate Loss over C7-supported context scenarios"
    return all_results, pd.DataFrame(mix_rows).merge(
        best[["budget_1e21", "cost_function", "context_tokens"]].assign(selected=True),
        on=["budget_1e21", "cost_function", "context_tokens"], how="left").fillna({"selected": False})


def quality_cost_curve(q0: float, n_points: int = 101) -> pd.DataFrame:
    """Tabulate the three normalized Appendix B quality cost functions."""
    q = np.linspace(q0, Q_UPPER, n_points)
    out = pd.DataFrame({"Q": q})
    for fn in COST_FUNCTIONS:
        out[f"g_relative_{fn.name}"] = fn.relative_increment(q, q0)
    return out


def critical_context() -> float:
    """Analytic context length at which attention equals training cost."""
    return 6.0 / ETA_ATTN
