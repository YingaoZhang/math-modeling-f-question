"""Explicit, machine-readable bridge from Problem 1 outputs to Problem 2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .conflicts import conflict_severity_from_ranks


def export_q2_bridge_interface(out: Path, data_root: Path, quality: pd.DataFrame,
                               loss_coefficients: pd.DataFrame,
                               lambda_penalty: float = 0.0,
                               joint_effects: pd.DataFrame | None = None) -> dict:
    """Export a bounded 17-domain quality functional and composition effects.

    The document-level penalized score is aggregated by source domain. Domains
    without A1 quality records receive the observed-domain median and are marked
    as imputed. This makes the interface complete without claiming measurements
    that are absent. The mapping table enumerates arbitrary recipe rows from the
    training mixture and is directly loadable by Q2.
    """
    out.mkdir(parents=True, exist_ok=True)
    domain_order = [c.replace("train_the_pile_", "") for c in
                    pd.read_csv(data_root / "A_data_value" / "regmix_tables" / "train_mixture_1m.csv", nrows=1)
                    if c.startswith("train_the_pile_")]
    # Use the requested conflict-adjusted score when lambda > 0.  The bundle
    # records the lambda so Q2 can distinguish this normative adjustment from
    # a supervised calibration.
    document_score = (quality["Q_equal"] - lambda_penalty * conflict_severity_from_ranks(quality)).clip(1e-9, 1.0)
    source_q = document_score.groupby(quality["domain"]).mean().to_dict()
    observed = {d: float(source_q[d]) for d in source_q}
    fallback = float(np.median(list(observed.values()))) if observed else 0.5
    q_star = {d: float(np.clip(observed.get(d, fallback), 1e-9, 1.0)) for d in domain_order}
    q_status = {d: ("observed" if d in observed else "median_imputed_no_document_quality") for d in domain_order}

    mix = pd.read_csv(data_root / "A_data_value" / "regmix_tables" / "train_mixture_1m.csv")
    proportions = mix[[f"train_the_pile_{d}" for d in domain_order]].to_numpy(float)
    row_sum = proportions.sum(axis=1, keepdims=True)
    proportions = np.divide(proportions, np.where(row_sum == 0, 1.0, row_sum))
    q_vec = np.array([q_star[d] for d in domain_order])
    mapping = pd.DataFrame({"recipe_id": mix["index"], "Q_bar": proportions @ q_vec})
    mapping.to_csv(out / "q2_recipe_quality_mapping.csv", index=False)
    baseline_q0 = float(np.average(mapping.Q_bar, weights=np.ones(len(mapping))))

    # Quantify how much macro quality variation is lost when domains without
    # document-level measurements are filled with the observed-domain median.
    observed_values = [observed[d] for d in observed]
    median_value = float(np.median(observed_values)) if observed_values else 0.5
    q_observed_only = np.array([observed.get(d, np.nan) for d in domain_order], dtype=float)
    q_imputed = np.where(np.isfinite(q_observed_only), q_observed_only, median_value)
    # For this diagnostic, unknown domains are removed and the remaining
    # composition is reclosed, producing the quality functional supported by
    # actual document evidence only.
    known_mask = np.isfinite(q_observed_only)
    known_prop = proportions[:, known_mask]
    known_prop = known_prop / np.where(known_prop.sum(axis=1, keepdims=True) == 0,
                                       1.0, known_prop.sum(axis=1, keepdims=True))
    q_bar_observed = known_prop @ q_observed_only[known_mask]
    q_bar_imputed = proportions @ q_imputed
    var_full = float(np.var(q_bar_imputed, ddof=1))
    var_observed = float(np.var(q_bar_observed, ddof=1))
    pd.DataFrame([{
        "n_domains": len(domain_order),
        "n_observed_domains": int(known_mask.sum()),
        "n_median_imputed_domains": int((~known_mask).sum()),
        "median_imputation_value": median_value,
        "Qbar_variance_with_imputation": var_full,
        "Qbar_variance_observed_only_reclosed": var_observed,
        "variance_loss_fraction_vs_observed_only": float(1.0 - var_full / var_observed) if var_observed > 0 else np.nan,
        "interpretation": "Q_bar is weak evidence because most domains lack document-level quality observations"
    }]).to_csv(out / "qbar_imputation_variance_loss.csv", index=False)

    coeff = loss_coefficients.copy()
    t_summary = coeff.groupby("mixture_domain").t_relative_substitution.mean().reindex(domain_order).fillna(0.0)
    t_vec = (t_summary - t_summary.mean()).to_numpy(float)
    top_pairs = []
    for loss_domain, group in coeff.groupby("loss_domain"):
        s = group.set_index("mixture_domain").t_relative_substitution.reindex(domain_order).fillna(0.0)
        vals = []
        for i, a in enumerate(domain_order):
            for b in domain_order[i + 1:]:
                vals.append((abs(float(s[a] - s[b])), a, b, float(s[a] - s[b])))
        for magnitude, a, b, delta in sorted(vals, reverse=True)[:5]:
            top_pairs.append({"loss_domain": loss_domain, "domain_a": a, "domain_b": b,
                              "relative_substitution_delta": delta, "absolute_delta": magnitude})

    # Prefer independently evaluated quadratic non-additivity for the second-
    # order interface.  The coefficient-difference fallback is retained for
    # backwards compatibility when older callers do not provide test results.
    if joint_effects is not None and len(joint_effects):
        top_pairs = []
        for loss_domain, group in joint_effects.groupby("loss_domain"):
            ranked = group.assign(_abs=group["mean_absolute_nonadditive_delta_loss"].abs()) \
                         .sort_values("_abs", ascending=False).head(5)
            for _, row in ranked.iterrows():
                top_pairs.append({"loss_domain": loss_domain,
                                  "domain_a": row["recipient_1"],
                                  "domain_b": row["recipient_2"],
                                  "donor_domain": row["donor_domain"],
                                  "transfer_each": float(row["transfer_each"]),
                                  "nonadditive_delta_loss": float(row["mean_nonadditive_delta_loss"]),
                                  "absolute_nonadditive_delta_loss": float(row["mean_absolute_nonadditive_delta_loss"]),
                                  "effect_source": "independent test quadratic model"})
    else:
        for row in top_pairs:
            row["effect_source"] = "first-order substitution contrast fallback; no quadratic test table supplied"

    bundle = {
        "schema_version": "q1-q2-bridge-v1",
        "quality_scale": "A1 ECDF; Q_star in (0,1]; lambda penalty is descriptive unless paired validation is supplied",
        "conflict_penalty_lambda": float(lambda_penalty),
        "domain_order": domain_order,
        "Q_star_17": q_star,
        "Q_star_status": q_status,
        "Q0_train_recipe_mean": baseline_q0,
        "Qbar_imputation_diagnostic": "qbar_imputation_variance_loss.csv",
        "first_order_substitution_vector_t": {d: float(v) for d, v in zip(domain_order, t_vec)},
        "second_order_top5_pairs_by_loss_domain": top_pairs,
        "second_order_effect_definition": "joint transfer change minus sum of two single transfers, evaluated per Loss domain on independent test recipes",
        "recipe_mapping_csv": "q2_recipe_quality_mapping.csv",
        "mapping_formula": "Q_bar(p)=sum_i p_i Q_star_i after row closure",
        "loss_scale_note": "The Pile validation cross entropy only; do not mix directly with Pythia val_loss",
    }
    (out / "q2_input_interface_bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"【实证检验通过】Q2 bridge exported: {len(domain_order)} domains, Q0={baseline_q0:.6f}, mapped recipes={len(mapping)}")
    return bundle
