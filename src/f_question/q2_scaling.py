"""Scaling-law models for Problem 2.

The module deliberately keeps the validation-loss scale of Appendix B
separate from the The Pile loss scale used in Question 1.  B1 is the
primary Pythia fit; B6--B8 are used only for the native ``Q_score`` effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


@dataclass
class ClassicFit:
    params: np.ndarray
    feature_exponents: tuple[float, float]
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "n": int(len(y)),
    }


def classic_prediction(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    n, d = np.asarray(x)[:, 0], np.asarray(x)[:, 1]
    c, a, b, alpha, beta = params
    return c + a * np.power(np.maximum(n, 1e-9), -alpha) + b * np.power(np.maximum(d, 1e-9), -beta)


def fit_classic_scaling(df: pd.DataFrame, random_state: int = 42) -> tuple[ClassicFit, pd.DataFrame]:
    """Fit ``L=c+A N^-alpha+B D^-beta`` on B1 and return diagnostics."""
    work = df[["N_params_B", "D_tokens_B", "val_loss"]].apply(pd.to_numeric, errors="coerce").dropna()
    work = work[(work.N_params_B > 0) & (work.D_tokens_B > 0) & (work.val_loss > 0)].reset_index(drop=True)
    train, test = train_test_split(work, test_size=0.2, random_state=random_state)
    xtr = train[["N_params_B", "D_tokens_B"]].to_numpy(float)
    ytr = train.val_loss.to_numpy(float)
    xte = test[["N_params_B", "D_tokens_B"]].to_numpy(float)
    yte = test.val_loss.to_numpy(float)
    def residual(z: np.ndarray) -> np.ndarray:
        return classic_prediction(xtr, z) - ytr
    y0 = float(np.median(ytr))
    init = np.array([max(0.5, y0 * 0.7), 1.0, 1.0, 0.3, 0.3])
    fit = least_squares(residual, init, bounds=([-10, 0, 0, 0.01, 0.01], [20, 100, 100, 2.0, 2.0]), max_nfev=50000)
    params = fit.x
    train_metrics = _metrics(ytr, classic_prediction(xtr, params))
    test_metrics = _metrics(yte, classic_prediction(xte, params))
    model = ClassicFit(params, (float(params[3]), float(params[4])), train_metrics, test_metrics)
    pred = classic_prediction(work[["N_params_B", "D_tokens_B"]].to_numpy(float), params)
    diagnostics = work.assign(predicted_loss=pred, residual=work.val_loss - pred,
                              split=np.where(work.index.isin(train.index), "train", "test"))
    return model, diagnostics


def _fit_classic_all(df: pd.DataFrame) -> ClassicFit:
    """Fit the nonlinear model on every row in ``df`` (used inside CV folds)."""
    work = df[["N_params_B", "D_tokens_B", "val_loss"]].apply(pd.to_numeric, errors="coerce").dropna()
    work = work[(work.N_params_B > 0) & (work.D_tokens_B > 0) & (work.val_loss > 0)]
    x = work[["N_params_B", "D_tokens_B"]].to_numpy(float); y = work.val_loss.to_numpy(float)
    def residual(z: np.ndarray) -> np.ndarray: return classic_prediction(x, z) - y
    init = np.array([max(0.5, float(np.median(y)) * .7), 1.0, 1.0, .3, .3])
    fit = least_squares(residual, init, bounds=([-10, 0, 0, .01, .01], [20, 100, 100, 2, 2]), max_nfev=50000)
    return ClassicFit(fit.x, (float(fit.x[3]), float(fit.x[4])), _metrics(y, classic_prediction(x, fit.x)), {})


def fit_quality_effect(b6_b8: pd.DataFrame, classic: ClassicFit) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a regularized Q extension on native B6--B8 ``Q_score``.

    The fixed exponents from B1 make the quality coefficient identifiable and
    keep the half-synthetic extension from freely re-fitting all scale laws.
    """
    d = b6_b8[["N_params_B", "D_tokens_B", "Q_score", "val_loss"]].apply(pd.to_numeric, errors="coerce").dropna()
    d = d[(d.N_params_B > 0) & (d.D_tokens_B > 0) & (d.Q_score > 0) & (d.val_loss > 0)].copy()
    alpha, beta = classic.feature_exponents
    nterm = np.power(d.N_params_B.to_numpy(float), -alpha)
    dterm = np.power(d.D_tokens_B.to_numpy(float), -beta)
    q = d.Q_score.to_numpy(float)
    q0 = float(np.median(q))
    qc = q - q0
    X = np.column_stack([np.ones(len(d)), nterm, dterm, qc, qc * nterm, qc * dterm])
    y = d.val_loss.to_numpy(float)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model = RidgeCV(alphas=np.logspace(-5, 3, 25), cv=cv).fit(X, y)
    pred = model.predict(X)
    coef_names = ["intercept", "N_term", "D_term", "Q_centered", "QxN_term", "QxD_term"]
    coef = pd.DataFrame({"coefficient": coef_names, "value": np.r_[model.intercept_, model.coef_[1:]],
                         "alpha_selected": float(model.alpha_), "Q0_native": q0,
                         "r2_in_sample": float(r2_score(y, pred)), "mae_in_sample": float(mean_absolute_error(y, pred))})
    d["predicted_loss"] = pred
    d["residual"] = y - pred
    d["source_group"] = "B6_B7_B8"
    return coef, d


def quality_prediction(df: pd.DataFrame, coef: pd.DataFrame, classic: ClassicFit) -> np.ndarray:
    """Predict B-native quality-extended Loss from a coefficient table."""
    alpha, beta = classic.feature_exponents
    vals = coef.set_index("coefficient")["value"]
    q0 = float(coef.Q0_native.iloc[0])
    n = pd.to_numeric(df.N_params_B).to_numpy(float)
    d = pd.to_numeric(df.D_tokens_B).to_numpy(float)
    q = pd.to_numeric(df.Q_score).to_numpy(float)
    nt = n ** -alpha
    dt = d ** -beta
    return (vals["intercept"] + vals["N_term"] * nt + vals["D_term"] * dt
            + vals["Q_centered"] * (q - q0) + vals["QxN_term"] * (q - q0) * nt
            + vals["QxD_term"] * (q - q0) * dt)


def cross_validate_classic(df: pd.DataFrame) -> pd.DataFrame:
    """Five-fold CV summary for the nonlinear classic model."""
    work = df[["N_params_B", "D_tokens_B", "val_loss"]].apply(pd.to_numeric, errors="coerce").dropna()
    work = work[(work.N_params_B > 0) & (work.D_tokens_B > 0) & (work.val_loss > 0)].reset_index(drop=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for fold, (tr, te) in enumerate(kf.split(work), 1):
        sub = work.iloc[tr].copy()
        model = _fit_classic_all(sub)
        x = work.iloc[te][["N_params_B", "D_tokens_B"]].to_numpy(float)
        y = work.iloc[te].val_loss.to_numpy(float)
        rows.append({"fold": fold, **_metrics(y, classic_prediction(x, model.params)),
                     "alpha": model.params[3], "beta": model.params[4]})
    return pd.DataFrame(rows)
