"""Synthetic claims portfolio generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def generate_claims(count: int = 10_000, *, seed: int = 42) -> pd.DataFrame:
    """Create a reproducible claim-level table with an operational delay target."""
    if count < 100:
        raise ValueError("count must be at least 100")
    rng = np.random.default_rng(seed)

    lines = rng.choice(["auto", "property", "liability"], count, p=[0.52, 0.31, 0.17])
    channels = rng.choice(["broker", "direct", "digital"], count, p=[0.45, 0.32, 0.23])
    regions = rng.choice(["west", "ontario", "quebec", "atlantic"], count, p=[0.24, 0.43, 0.23, 0.10])
    report_offsets = rng.integers(0, 730, count)
    reported_at = pd.Timestamp("2024-01-01") + pd.to_timedelta(report_offsets, unit="D")

    complexity = np.clip(
        rng.beta(2.2, 3.2, count)
        + 0.18 * (lines == "liability")
        + 0.08 * (lines == "property"),
        0,
        1,
    )
    coverage_limit = np.where(
        lines == "auto",
        rng.choice([50_000, 100_000, 200_000], count),
        np.where(
            lines == "property",
            rng.choice([250_000, 500_000, 1_000_000], count),
            rng.choice([1_000_000, 2_000_000, 5_000_000], count),
        ),
    ).astype(float)
    severity_estimate = np.minimum(
        coverage_limit,
        coverage_limit * rng.lognormal(mean=-2.7 + 1.2 * complexity, sigma=0.7, size=count),
    )
    missing_documents = rng.binomial(3, np.clip(0.10 + 0.35 * complexity, 0, 0.8), count)
    days_since_last_activity = np.clip(
        rng.gamma(1.8, 2.8, count) + 7 * (missing_documents > 1),
        0,
        45,
    ).round().astype(int)
    activity_count_7d = np.maximum(
        0,
        rng.poisson(np.clip(4.5 - 2.0 * complexity - 0.12 * days_since_last_activity, 0.3, None)),
    )
    adjuster_open_load = np.clip(rng.normal(38, 12, count) + 12 * (regions == "ontario"), 5, 95).round().astype(int)
    fraud_indicator = rng.binomial(1, np.clip(0.015 + 0.06 * complexity, 0, 0.2), count)

    logit = (
        -3.6
        + 0.055 * days_since_last_activity
        + 0.030 * adjuster_open_load
        + 0.65 * missing_documents
        + 1.7 * complexity
        - 0.16 * activity_count_7d
        + 0.35 * (channels == "broker")
        + 0.40 * (lines == "liability")
        + rng.normal(0, 0.65, count)
    )
    delay_probability = _sigmoid(logit)
    delay_flag = rng.binomial(1, delay_probability)
    target_days = np.select(
        [lines == "auto", lines == "property"],
        [30, 45],
        default=60,
    )
    cycle_time_days = np.maximum(
        1,
        rng.normal(target_days * (0.65 + 0.75 * delay_flag), target_days * 0.22, count),
    ).round().astype(int)
    closed_flag = rng.binomial(1, np.clip(0.88 - 0.28 * delay_flag, 0.2, 0.95), count)
    closed_at = reported_at + pd.to_timedelta(cycle_time_days, unit="D")
    closed_at = pd.Series(closed_at).where(closed_flag == 1, pd.NaT)
    settlement_amount = np.where(
        closed_flag == 1,
        np.minimum(coverage_limit, severity_estimate * rng.uniform(0.72, 1.08, count)),
        0.0,
    ).round(2)

    return pd.DataFrame(
        {
            "claim_id": [f"CLM-{index:07d}" for index in range(1, count + 1)],
            "policy_id": [f"POL-{value:06d}" for value in rng.integers(1, max(2, count // 2), count)],
            "reported_at": reported_at,
            "closed_at": pd.to_datetime(closed_at),
            "line_of_business": lines,
            "channel": channels,
            "region": regions,
            "coverage_limit": coverage_limit,
            "severity_estimate": severity_estimate.round(2),
            "settlement_amount": settlement_amount,
            "activity_count_7d": activity_count_7d,
            "days_since_last_activity": days_since_last_activity,
            "missing_documents": missing_documents,
            "adjuster_open_load": adjuster_open_load,
            "fraud_indicator": fraud_indicator,
            "complexity_score": complexity.round(6),
            "target_days": target_days,
            "cycle_time_days": cycle_time_days,
            "closed_flag": closed_flag,
            "delay_flag": delay_flag,
        }
    )

