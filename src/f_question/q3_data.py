"""Data loading and boundary definitions for Question 3.

The third question consumes the fitted outputs of Questions 1 and 2.  This
module keeps that interface explicit and records the unit conversions used by
the compute budget model.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Q3Inputs:
    """Validated inputs passed to the optimization layer."""

    domain_order: tuple[str, ...]
    q_star: np.ndarray
    t_vector: np.ndarray
    p0: np.ndarray
    q0: float
    n_bounds_b: tuple[float, float]
    d_bounds_b: tuple[float, float]
    context_values: tuple[int, ...]
    classic: dict[str, float]
    quality: dict[str, float]
    classic_by_source: dict[str, dict[str, float]]
    quality_by_source: dict[str, dict[str, float]]
    architecture: pd.DataFrame
    second_order: pd.DataFrame


def _read_coefficients(q2_dir: Path) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Read two complete, internally paired Q2 coefficient conventions.

    ``b6_b8_lambda10`` is the native quality-extension convention. ``b1``
    keeps B1's jointly fitted classic coefficients as a Pythia-compatible
    sensitivity baseline.  B1 has no native quality coefficient, so its
    quality term is explicitly marked as borrowed from B6--B8; this is a
    mixed convention and is never presented as a fully jointly observed fit.
    """
    classic_path = q2_dir / "q2_b1_fit_coefficients.csv"
    quality_path = q2_dir / "q2_quality_effect_coefficients.csv"
    b1 = {"E": 1.6897677882, "A": 0.3540135531, "B": 1.2402868555,
          "alpha": 0.3399571273, "beta": 0.2798761245}
    b68 = {"E": 1.60617, "A": 0.8111223285, "B": 0.5641070701,
           "alpha": 0.2334249279, "beta": 0.2625033577}
    quality10 = {"gamma": 1.5164923304, "theta": 1.0868284874}
    if classic_path.exists():
        c = pd.read_csv(classic_path)
        for name in b1:
            row = c.loc[c.parameter.eq(name), "value"]
            if len(row):
                b1[name] = float(row.iloc[0])
    if quality_path.exists():
        q = pd.read_csv(quality_path)
        row = q.loc[np.isclose(q["lambda"], 10.0)]
        if len(row):
            for name in quality10:
                quality10[name] = float(row.iloc[0][name])
            for name in b68:
                if name in row.columns:
                    b68[name] = float(row.iloc[0][name])
    return {"b6_b8_lambda10": b68, "b1": b1}, {"b6_b8_lambda10": quality10, "b1": quality10.copy()}


def load_q3_inputs(project_root: Path) -> Q3Inputs:
    """Load C7 and the Q1/Q2 bridge without reading the large C8 directory.

    C7 supplies the empirical context-window support.  The 17-domain baseline
    recipe is the mean of A4 after row closure, and the quality vector is read
    from the explicit Q1-to-Q2 JSON interface.
    """
    root = Path(project_root)
    real = root / "data" / "raw" / "real_attachments"
    c7 = pd.read_csv(real / "C_efficiency_evolution" / "model_architecture_metadata.csv")
    c7["max_position_embeddings"] = pd.to_numeric(c7["max_position_embeddings"], errors="coerce")
    contexts = tuple(sorted(c7["max_position_embeddings"].dropna().astype(int).unique()))

    bridge_path = root / "outputs" / "q1" / "q2_input_interface_bundle.json"
    if not bridge_path.exists():
        raise FileNotFoundError(f"Missing Q1 interface: {bridge_path}")
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    domains = tuple(bridge["domain_order"])
    q_star = np.array([float(bridge["Q_star_17"][d]) for d in domains], dtype=float)
    t_vector = np.array([float(bridge["first_order_substitution_vector_t"][d]) for d in domains], dtype=float)
    p0_df = pd.read_csv(real / "A_data_value" / "regmix_tables" / "train_mixture_1m.csv")
    p_cols = [f"train_the_pile_{d}" for d in domains]
    missing = sorted(set(p_cols) - set(p0_df.columns))
    if missing:
        raise ValueError(f"A4 is missing expected domains: {missing}")
    recipe = p0_df[p_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    recipe = np.maximum(recipe, 1e-12)
    recipe /= recipe.sum(axis=1, keepdims=True)
    p0 = recipe.mean(axis=0)
    p0 = np.maximum(p0, 1e-6)
    p0 /= p0.sum()
    q0 = float(bridge.get("Q0_train_recipe_mean", np.dot(p0, q_star)))

    classic_by_source, quality_by_source = _read_coefficients(root / "outputs" / "q2")
    primary = "b6_b8_lambda10"
    second_order = pd.DataFrame(bridge.get("second_order_top5_pairs_by_loss_domain", []))
    return Q3Inputs(
        domain_order=domains,
        q_star=q_star,
        t_vector=t_vector,
        p0=p0,
        q0=q0,
        # The upper bound is a scenario boundary chosen to cover the C4
        # frontier, not a physical claim about maximum model size.
        n_bounds_b=(0.07, 5000.0),
        d_bounds_b=(0.134, 4000.0),
        context_values=contexts,
        classic=classic_by_source[primary],
        quality=quality_by_source[primary],
        classic_by_source=classic_by_source,
        quality_by_source=quality_by_source,
        architecture=c7,
        second_order=second_order,
    )


def with_source(inputs: Q3Inputs, source: str,
                n_bounds_b: tuple[float, float] | None = None,
                d_bounds_b: tuple[float, float] | None = None) -> Q3Inputs:
    """Return an immutable input view for one complete coefficient source."""
    if source not in inputs.classic_by_source:
        raise KeyError(f"unknown coefficient source: {source}")
    return Q3Inputs(
        domain_order=inputs.domain_order, q_star=inputs.q_star, t_vector=inputs.t_vector,
        p0=inputs.p0, q0=inputs.q0,
        n_bounds_b=n_bounds_b or inputs.n_bounds_b,
        d_bounds_b=d_bounds_b or inputs.d_bounds_b,
        context_values=inputs.context_values,
        classic=inputs.classic_by_source[source], quality=inputs.quality_by_source[source],
        classic_by_source=inputs.classic_by_source, quality_by_source=inputs.quality_by_source,
        architecture=inputs.architecture, second_order=inputs.second_order)


def input_boundary_table(inputs: Q3Inputs) -> pd.DataFrame:
    """Return a compact machine-readable description of Q3 data boundaries."""
    c8_dir = (Path(__file__).resolve().parents[2] / "data" / "raw" / "real_attachments"
              / "C_efficiency_evolution" / "detailed_results")
    c8_count = sum(1 for _ in c8_dir.rglob("*.json")) if c8_dir.exists() else 0
    return pd.DataFrame([
        {"item": "C7 architecture metadata", "role": "context support and architecture context",
         "n": len(inputs.architecture), "status": "real; 45 rows"},
        {"item": "Q1 q2_input_interface_bundle", "role": "Q_star_17 and first-order substitution vector",
         "n": 17, "status": "derived interface; 14 Q values are median-imputed"},
        {"item": "Q2 B1 fit coefficients", "role": "Pythia-compatible classic N-D loss proxy",
         "n": 1, "status": "Pythia fit; B1 classic + B6-B8 quality is explicitly mixed"},
        {"item": "A4 training recipe mean", "role": "baseline simplex composition p0",
         "n": 512, "status": "real recipe table; row-closed before averaging"},
        {"item": "C8 detailed_results", "role": "not used in Q3 or Q4",
         "n": c8_count, "status": "all JSON excluded from modelling; CSV summaries only"},
    ])
