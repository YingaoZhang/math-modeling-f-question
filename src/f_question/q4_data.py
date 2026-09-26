"""Data loaders and audit rules for Question 4.

Question 4 consumes the compact CSV attachments, the explicit Q3 frontier
contract, and the valid C8 JSON task results. Malformed JSON files remain in
the audit manifest and are excluded from all modelling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def project_paths(root: Path) -> dict[str, Path]:
    base = root / "data" / "raw" / "real_attachments" / "C_efficiency_evolution"
    return {
        "epoch": base / "epoch_all_ai_models.csv",
        "leaderboard": base / "leaderboard_cleaned.csv",
        "timeseries": base / "leaderboard_extended_timeseries.csv",
        "bridge": base / "loss_benchmark_bridge_expanded.csv",
        "architecture": base / "model_architecture_metadata.csv",
        "frontier": root / "outputs" / "q3" / "q3_loss_frontier_curve.csv",
        "contract": root / "outputs" / "q4_input_contract.json",
    }


def read_csv(path: Path) -> pd.DataFrame:
    """Read a supplied CSV with tolerant parsing and explicit path errors."""
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def audit_c8_json(root: Path) -> pd.DataFrame:
    """Return the exclusion manifest for all C8 JSON files.

    Parsing failures are recorded as exclusions. Valid files are marked for
    task-level aggregation; the manifest is retained so the data boundary is
    reproducible and malformed files cannot enter the model silently.
    """
    folder = root / "data" / "raw" / "real_attachments" / "C_efficiency_evolution" / "detailed_results"
    rows: list[dict[str, Any]] = []
    for path in sorted(folder.rglob("*.json")):
        status, error = "valid_for_aggregation", "valid C8 JSON parsed; included in official task aggregation"
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # malformed files are useful audit evidence
            status, error = "malformed_excluded", f"{type(exc).__name__}: {exc}"
        rows.append({"path": str(path.relative_to(root)), "status": status, "reason": error})
    return pd.DataFrame(rows)


def load_q4_data(root: Path) -> dict[str, Any]:
    """Load all Q4 CSV inputs and attach data-boundary metadata."""
    paths = project_paths(root)
    data = {k: read_csv(v) for k, v in paths.items() if k in {"epoch", "leaderboard", "timeseries", "bridge", "architecture"}}
    data["frontier"] = read_csv(paths["frontier"])
    data["contract"] = json.loads(paths["contract"].read_text(encoding="utf-8")) if paths["contract"].exists() else {}
    data["json_manifest"] = audit_c8_json(root)
    return data


def prepare_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize compute and parameter fields from the Epoch model table."""
    d = df.copy()
    d["compute_flop"] = pd.to_numeric(d["Training compute (FLOP)"], errors="coerce")
    d["params"] = pd.to_numeric(d["Parameters"], errors="coerce")
    d["date"] = pd.to_datetime(d["Publication date"], errors="coerce")
    d = d[(d.compute_flop > 0) & np.isfinite(d.compute_flop)].copy()
    d["year"] = d.date.dt.year
    return d


def prepare_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Keep pretrained/base submissions and normalize leaderboard fields."""
    d = df.copy()
    params_col = "N_params_B" if "N_params_B" in d.columns else "#Params (B)"
    avg_col = "Average" if "Average" in d.columns else next(c for c in d.columns if c.startswith("Average"))
    d["params_b"] = pd.to_numeric(d[params_col], errors="coerce")
    d["average"] = pd.to_numeric(d[avg_col], errors="coerce")
    d["date"] = pd.to_datetime(d.get("Submission Date"), errors="coerce")
    typ = d.get("Type", pd.Series("", index=d.index)).fillna("").astype(str).str.lower()
    # Strict base/pretrained filter requested by the data specification.
    d["is_pretrained"] = typ.str.contains("pretrained") & ~typ.str.contains("fine-tuned|chat|merge")
    return d[(d.is_pretrained) & (d.params_b > 0) & d.average.notna()].copy()
