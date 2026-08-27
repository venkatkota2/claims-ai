"""Model stability and segment monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) == 0 or len(current) == 0 or bins < 2:
        raise ValueError("PSI requires non-empty samples and at least two bins")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_share = np.histogram(reference, bins=edges)[0] / len(reference)
    current_share = np.histogram(current, bins=edges)[0] / len(current)
    reference_share = np.clip(reference_share, 1e-6, None)
    current_share = np.clip(current_share, 1e-6, None)
    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def segment_performance(
    claims: pd.DataFrame,
    probability: np.ndarray,
    *,
    segment: str,
    threshold: float = 0.70,
) -> pd.DataFrame:
    if segment not in claims:
        raise ValueError(f"unknown segment: {segment}")
    frame = claims[[segment, "delay_flag"]].copy()
    frame["probability"] = np.asarray(probability)
    rows = []
    for name, group in frame.groupby(segment, dropna=False):
        auc = (
            float(roc_auc_score(group["delay_flag"], group["probability"]))
            if group["delay_flag"].nunique() == 2
            else float("nan")
        )
        rows.append(
            {
                segment: name,
                "claims": len(group),
                "delay_rate": float(group["delay_flag"].mean()),
                "mean_score": float(group["probability"].mean()),
                "review_rate": float((group["probability"] >= threshold).mean()),
                "roc_auc": auc,
            }
        )
    return pd.DataFrame(rows).sort_values(segment).reset_index(drop=True)
