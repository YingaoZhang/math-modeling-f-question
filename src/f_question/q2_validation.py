"""External validation, extrapolation and sensitivity calculations for Q2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, f as f_dist, t as t_dist
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .q2_scaling import ClassicFit, classic_prediction, asymmetric_quality_prediction


def metric_row(name: str, y: np.ndarray, pred: np.ndarray, data_status: str) -> dict:
    return {"dataset": name, "data_status": data_status, "n": int(len(y)),
            "r2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "spearman": float(spearmanr(y, pred).statistic) if len(y) > 2 else np.nan}


def external_validation(paths: dict[str, Path], classic: ClassicFit) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows, predictions = [], {}
    for name, status in [("B2_Cerebras", "semi_external_real_family"),
                         ("B4_cross_family", "real_cross_family"),
                         ("B5_published", "real_literature")]:
        df = pd.read_csv(paths[name])
        df = df[["N_params_B", "D_tokens_B", "val_loss"]].apply(pd.to_numeric, errors="coerce").dropna()
        df = df[(df.N_params_B > 0) & (df.D_tokens_B > 0) & (df.val_loss > 0)]
        pred = classic_prediction(df[["N_params_B", "D_tokens_B"]].to_numpy(float), classic.params)
        predictions[name] = df.assign(predicted_loss=pred, residual=df.val_loss.to_numpy(float) - pred)
        rows.append(metric_row(name, df.val_loss.to_numpy(float), pred, status))
    return pd.DataFrame(rows), predictions


def trajectory_validation(paths: list[Path], classic: ClassicFit) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, all_pred = [], []
    for p in paths:
        df = pd.read_csv(p)
        df = df[["N_params_B", "D_tokens_B", "val_loss", "step", "interpolated"]].apply(pd.to_numeric, errors="coerce").dropna()
        pred = classic_prediction(df[["N_params_B", "D_tokens_B"]].to_numpy(float), classic.params)
        all_pred.append(df.assign(trajectory=p.stem, predicted_loss=pred, residual=df.val_loss.to_numpy(float) - pred))
        rows.append({"trajectory": p.stem, **metric_row(p.stem, df.val_loss.to_numpy(float), pred, "interpolated_trajectory")})
    return pd.DataFrame(rows), pd.concat(all_pred, ignore_index=True)


def large_scale_extrapolation(path_models: Path, path_loss: Path, classic: ClassicFit) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(path_models)
    loss = pd.read_csv(path_loss)
    merged = loss.merge(meta[["model_name", "N_params_B", "D_tokens_B"]],
                        left_on=["family", "N_params_B", "D_tokens_B"],
                        right_on=["model_name", "N_params_B", "D_tokens_B"], how="left")
    # Some rows have no model-name match; the family/scale loss is still valid.
    merged = merged.dropna(subset=["N_params_B", "D_tokens_B", "val_loss"])
    pred = classic_prediction(merged[["N_params_B", "D_tokens_B"]].to_numpy(float), classic.params)
    merged["predicted_loss"] = pred
    merged["residual"] = merged.val_loss.to_numpy(float) - pred
    metrics = pd.DataFrame([metric_row("B9_B10_large", merged.val_loss.to_numpy(float), pred, "estimated_large_scale")])
    return metrics, merged


def b2_recalibration(paths: dict[str, Path], classic: ClassicFit, folds: int = 5) -> pd.DataFrame:
    """Fit a within-B2 affine calibration and report cross-validated metrics."""
    df = pd.read_csv(paths["B2_Cerebras"])[["N_params_B", "D_tokens_B", "val_loss"]].apply(pd.to_numeric, errors="coerce").dropna()
    df = df[(df.N_params_B > 0) & (df.D_tokens_B > 0) & (df.val_loss > 0)].reset_index(drop=True)
    raw = classic_prediction(df[["N_params_B", "D_tokens_B"]].to_numpy(float), classic.params)
    rng = np.random.default_rng(42); order = rng.permutation(len(df)); chunks = np.array_split(order, folds)
    rows = []
    for i, te in enumerate(chunks, 1):
        tr = np.setdiff1d(order, te); x = raw[tr]; y = df.val_loss.to_numpy(float)[tr]
        a, b = np.polyfit(x, y, 1)
        pred = a * raw[te] + b
        rows.append({"fold": i, "n": len(te), "slope": a, "intercept": b,
                     **metric_row("B2_affine_calibrated", df.val_loss.to_numpy(float)[te], pred, "5fold_affine_recalibration")})
    out = pd.DataFrame(rows)
    out.attrs["raw_spearman"] = float(spearmanr(df.val_loss, raw).statistic)
    out.attrs["raw_r2"] = float(r2_score(df.val_loss, raw))
    return out


def affine_bridge_test(path: Path) -> pd.DataFrame:
    """Test the proposed ``L_Pile=u L_Pythia+v`` bridge by comparability tier."""
    df = pd.read_csv(path)
    rows = []
    for tier, g in [("all", df), ("High", df[df.Loss_Comparability.str.startswith("High")]),
                    ("Medium", df[df.Loss_Comparability.str.startswith("Medium")])]:
        x = g.Val_Loss.to_numpy(float); y = g.LB_Average.to_numpy(float); n = len(g)
        if n < 3:
            rows.append({"tier": tier, "n": n, "u": np.nan, "v": np.nan, "r2": np.nan, "F": np.nan,
                         "p_value": np.nan, "u_ci_low": np.nan, "u_ci_high": np.nan,
                         "v_ci_low": np.nan, "v_ci_high": np.nan, "ci_width_u": np.nan,
                         "identifiability": "insufficient observations"}); continue
        X = np.column_stack([np.ones(n), x]); beta = np.linalg.lstsq(X, y, rcond=None)[0]; pred = X @ beta
        resid = y - pred; sse = float(np.sum(resid**2)); sst = float(np.sum((y-y.mean())**2)); r2 = 1-sse/sst
        df1, df2 = 1, n-2; F = ((sst-sse)/df1)/(sse/df2) if sse > 0 else np.inf
        p = float(f_dist.sf(F, df1, df2))
        cov = (sse/df2) * np.linalg.inv(X.T @ X); se = np.sqrt(np.diag(cov)); crit = float(t_dist.ppf(.975, df2))
        rows.append({"tier": tier, "n": n, "u": beta[1], "v": beta[0], "r2": r2, "F": F, "p_value": p,
                     "u_ci_low": beta[1]-crit*se[1], "u_ci_high": beta[1]+crit*se[1],
                     "v_ci_low": beta[0]-crit*se[0], "v_ci_high": beta[0]+crit*se[0],
                     "ci_width_u": 2*crit*se[1], "identifiability": "weak; High tier is 7/75" if tier == "High" else "descriptive"})
    return pd.DataFrame(rows)


def mrtS_table(classic: ClassicFit, quality_params: np.ndarray,
               n_values: np.ndarray | None = None, d_values: np.ndarray | None = None,
               q_quality: float = 0.65, q_step: float = 0.1) -> pd.DataFrame:
    """Compute the quality/parameter MRTS on a complete N,D grid.

    ``q_quality`` increases when defect rate falls.  The reported multiplier is
    therefore below one for a quality improvement; this is the paper-facing
    positive statement of the same MRTS calculation.
    """
    if n_values is None: n_values = np.array([0.070542, .162405, .409009, 1.040867, 1.416184, 2.782831, 6.86104, 11.965825])
    if d_values is None: d_values = np.linspace(5, 300, 147)
    rows = []
    p = np.asarray(quality_params, float)
    for n in n_values:
        for d in d_values:
            frame = pd.DataFrame({"N_params_B": [n], "D_tokens_B": [d], "Q_score": [1-q_quality]})
            L = float(asymmetric_quality_prediction(frame, p)[0])
            A, B, gamma, alpha, beta, theta = p
            defect = 1-q_quality
            dLdqdef = gamma*theta*max(defect, 1e-8)**(theta-1)
            dLdqquality = -dLdqdef
            dLdlnN = -alpha*A*n**(-alpha)
            delta_ln_n = -(dLdqquality*q_step)/dLdlnN
            rows.append({"N_params_B": n, "D_tokens_B": d, "Q_quality": q_quality,
                         "loss": L, "dL_dQ_quality": dLdqquality, "dL_dlnN": dLdlnN,
                         "elasticity_N": dLdlnN/L, "delta_Q_quality": q_step,
                         "delta_ln_N": delta_ln_n, "delta_N_percent": (np.exp(delta_ln_n)-1)*100,
                         "parameter_multiplier_for_quality_plus_0_1": np.exp(delta_ln_n),
                         "parameter_reduction_percent": (1-np.exp(delta_ln_n))*100,
                         "parameter_increase_multiplier_for_same_gain": np.exp(-delta_ln_n),
                         "N_equiv_B": n*np.exp(delta_ln_n)})
    return pd.DataFrame(rows)


def summarize_mrts(table: pd.DataFrame) -> pd.DataFrame:
    """Return min, quartiles, median and max for MRTS quantities."""
    cols = ["dL_dQ_quality", "dL_dlnN", "elasticity_N", "delta_ln_N", "delta_N_percent",
            "parameter_multiplier_for_quality_plus_0_1", "parameter_reduction_percent",
            "parameter_increase_multiplier_for_same_gain", "N_equiv_B"]
    rows = []
    for c in cols:
        s = table[c].dropna()
        rows.append({"quantity": c, "min": s.min(), "Q1": s.quantile(.25), "median": s.median(), "Q3": s.quantile(.75), "max": s.max()})
    return pd.DataFrame(rows)


def second_order_summary(bridge: dict) -> pd.DataFrame:
    """Summarize the 65 interface effects as model-conditional nonadditivity."""
    rows = bridge.get("second_order_top5_pairs_by_loss_domain", [])
    df = pd.DataFrame(rows)
    if df.empty: return pd.DataFrame()
    v = pd.to_numeric(df.nonadditive_delta_loss, errors="coerce")
    return pd.DataFrame([{"n_effects": len(df), "positive_share": float((v > 0).mean()),
                          "min": v.min(), "median": v.median(), "max": v.max(),
                          "effect_source": str(df.effect_source.iloc[0]),
                          "interpretation": "model-conditional nonadditive increment; not an experimental causal interaction"}])
