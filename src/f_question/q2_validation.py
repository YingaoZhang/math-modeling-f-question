"""External validation, extrapolation and sensitivity calculations for Q2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .q2_scaling import ClassicFit, classic_prediction, quality_prediction


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


def quality_tradeoffs(coef: pd.DataFrame, classic: ClassicFit) -> pd.DataFrame:
    vals = coef.set_index("coefficient")["value"]
    q0 = float(coef.Q0_native.iloc[0])
    rows = []
    for n in [0.07, 1.0, 10.0, 100.0, 700.0]:
        for d in [10, 100, 1000, 10000]:
            for q in [0.2, 0.5, 0.8, 1.0]:
                frame = pd.DataFrame({"N_params_B": [n], "D_tokens_B": [d], "Q_score": [q]})
                pred = float(quality_prediction(frame, coef, classic)[0])
                rows.append({"N_params_B": n, "D_tokens_B": d, "Q_score": q, "predicted_loss": pred})
    return pd.DataFrame(rows)


def elasticity_table(coef: pd.DataFrame, classic: ClassicFit) -> pd.DataFrame:
    vals = coef.set_index("coefficient")["value"]
    q0 = float(coef.Q0_native.iloc[0]); alpha, beta = classic.feature_exponents
    rows = []
    for n in [0.07, 1, 10, 100, 700]:
        for d in [10, 100, 1000, 10000]:
            for q in [0.2, 0.5, 0.8]:
                f = pd.DataFrame({"N_params_B": [n], "D_tokens_B": [d], "Q_score": [q]})
                L = float(quality_prediction(f, coef, classic)[0])
                nt, dt = n ** -alpha, d ** -beta
                dLdlnN = -alpha * (vals["N_term"] + (q-q0)*vals["QxN_term"]) * nt
                dLdlnD = -beta * (vals["D_term"] + (q-q0)*vals["QxD_term"]) * dt
                dLdQ = vals["Q_centered"] + vals["QxN_term"]*nt + vals["QxD_term"]*dt
                rows.append({"N_params_B": n, "D_tokens_B": d, "Q_score": q,
                             "loss": L, "elasticity_N": dLdlnN/L, "elasticity_D": dLdlnD/L,
                             "marginal_quality": dLdQ, "quality_improvement_per_0.1": -0.1*dLdQ})
    return pd.DataFrame(rows)
