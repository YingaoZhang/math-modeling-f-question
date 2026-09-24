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


def asymmetric_quality_prediction(df: pd.DataFrame, params: np.ndarray) -> np.ndarray:
    """Predict native B6--B8 loss with a physically signed quality term.

    ``Q_score`` is treated as a defect rate.  With ``q_quality=1-Q_score``
    the term is ``gamma*(1-q_quality)**theta = gamma*Q_score**theta``;
    therefore ``dL/dq_quality < 0`` whenever ``gamma, theta > 0``.
    ``params`` is ``(c, A, B, gamma, alpha, beta, theta)``.
    """
    n = pd.to_numeric(df.N_params_B).to_numpy(float)
    d = pd.to_numeric(df.D_tokens_B).to_numpy(float)
    q = np.clip(pd.to_numeric(df.Q_score).to_numpy(float), 1e-8, 1.0)
    c, a, b, gamma, alpha, beta, theta = params
    return c + a * n ** (-alpha) + b * d ** (-beta) + gamma * q ** theta


def fit_asymmetric_quality_effect(b6_b8: pd.DataFrame, classic: ClassicFit,
                                  lambdas: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0)) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit ``L=c+A N^-alpha+B D^-beta+gamma Q_score^theta``.

    The B1 exponents are soft priors.  The penalty is appended as two
    pseudo-residuals, so lambda is selected as a sensitivity parameter while
    preserving the physical sign restriction and positive power parameters.
    """
    d = b6_b8[["N_params_B", "D_tokens_B", "Q_score", "val_loss"]].apply(pd.to_numeric, errors="coerce").dropna()
    d = d[(d.N_params_B > 0) & (d.D_tokens_B > 0) & (d.Q_score >= 0) & (d.val_loss > 0)].copy()
    x = d[["N_params_B", "D_tokens_B", "Q_score"]].to_numpy(float)
    y = d.val_loss.to_numpy(float)
    alpha0, beta0 = classic.feature_exponents
    # Start from a stable positive fit; the bounds enforce gamma, theta > 0.
    init = np.array([0.6, 0.6, 1.0, 2.0, alpha0, beta0, 1.0], dtype=float)
    rows, pred_rows = [], []
    for lam in lambdas:
        def residual(z: np.ndarray) -> np.ndarray:
            pred = asymmetric_quality_prediction(pd.DataFrame(x, columns=["N_params_B", "D_tokens_B", "Q_score"]), z)
            penalty = np.sqrt(lam) * np.array([z[4] - alpha0, z[5] - beta0])
            return np.r_[pred - y, penalty]
        fit = least_squares(residual, init,
                            bounds=([-10, 0, 0, 1e-8, 0.01, 0.01, 0.05],
                                    [20, 100, 100, 100, 2.0, 2.0, 4.0]),
                            max_nfev=100000)
        init = fit.x
        pred = asymmetric_quality_prediction(d, fit.x)
        rows.append({
            "lambda": float(lam), "c": fit.x[0], "A": fit.x[1], "B": fit.x[2],
            "gamma": fit.x[3], "alpha": fit.x[4], "beta": fit.x[5], "theta": fit.x[6],
            "alpha_minus_B1": fit.x[4] - alpha0, "beta_minus_B1": fit.x[5] - beta0,
            "r2": float(r2_score(y, pred)), "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))), "n": int(len(y)),
            "physical_sign": "dL/dQ_quality<0; dL/dQ_defect>0",
        })
        if lam == 1.0:
            pred_rows = d.assign(predicted_loss=pred, residual=y - pred,
                                 q_quality=1.0 - d.Q_score.to_numpy(float),
                                 model="asymmetric_power_lambda_1")
    summary = pd.DataFrame(rows)
    if not len(pred_rows):
        best = summary.iloc[(summary.r2 - summary.r2.max()).abs().argmin()]
        # This branch is only defensive when a caller omits lambda=1.
        z = best[["c", "A", "B", "gamma", "alpha", "beta", "theta"]].to_numpy(float)
        pred = asymmetric_quality_prediction(d, z)
        pred_rows = d.assign(predicted_loss=pred, residual=y-pred, q_quality=1.0-d.Q_score, model="asymmetric_power")
    return summary, pred_rows, pd.DataFrame([{"derivative": "dL/dQ_quality", "minimum": float(np.min(-summary.gamma*summary.theta)), "sign": "strictly negative by construction"}])


def b1_grid_diagnostics(df: pd.DataFrame, classic: ClassicFit) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Quantify the complete B1 grid and structured holdout behaviour."""
    work = df[["N_params_B", "D_tokens_B", "val_loss"]].apply(pd.to_numeric, errors="coerce").dropna()
    keys = work[["N_params_B", "D_tokens_B"]].drop_duplicates()
    rows = [{"n_rows": len(work), "n_N": work.N_params_B.nunique(), "n_D": work.D_tokens_B.nunique(),
             "expected_cartesian": work.N_params_B.nunique()*work.D_tokens_B.nunique(),
             "unique_pairs": len(keys), "complete_unique_grid": bool(len(keys) == len(work))}]
    grid = pd.DataFrame(rows)
    # Group means reconstruct the generated grid exactly; this is a structure check,
    # not an independent generalisation claim.
    gm = work.groupby(["N_params_B", "D_tokens_B"], as_index=False).val_loss.mean()
    grid_r2 = r2_score(work.val_loss, gm.val_loss)
    grid["group_mean_r2"] = float(grid_r2)
    hold_rows = []
    for n in sorted(work.N_params_B.unique()):
        test = work[work.N_params_B == n]; train = work[work.N_params_B != n]
        # Fit on the remaining grid with the same nonlinear form.
        m = _fit_classic_all(train)
        pred = classic_prediction(test[["N_params_B", "D_tokens_B"]].to_numpy(float), m.params)
        hold_rows.append({"holdout_N": n, **_metrics(test.val_loss.to_numpy(float), pred)})
    # (N,D) pairs are unique; report the actual in-sample fit error rather than
    # copying a rounded value from the narrative. This remains a structured
    # grid diagnostic, not an independent external validation.
    full_pred = classic_prediction(work[["N_params_B", "D_tokens_B"]].to_numpy(float), classic.params)
    pair_hold = pd.DataFrame([{"holdout_scheme": "group_by_(N,D)",
                               **_metrics(work.val_loss.to_numpy(float), full_pred),
                               "note": "reported deterministic grid diagnostic; no random external data"}])
    return grid, pd.DataFrame(hold_rows), pair_hold


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
