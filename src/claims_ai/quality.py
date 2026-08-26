"""Claim-level data-quality controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class QualityCheck:
    rule: str
    passed: bool
    failure_count: int
    failure_rate: float


def run_quality_checks(claims: pd.DataFrame) -> list[QualityCheck]:
    required = {
        "claim_id",
        "policy_id",
        "reported_at",
        "closed_at",
        "coverage_limit",
        "severity_estimate",
        "settlement_amount",
        "closed_flag",
    }
    missing_columns = sorted(required - set(claims.columns))
    if missing_columns:
        raise ValueError(f"missing required columns: {', '.join(missing_columns)}")
    if claims.empty:
        raise ValueError("claims table must not be empty")

    rules: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = [
        ("claim_id_present", lambda frame: frame["claim_id"].notna()),
        ("claim_id_unique", lambda frame: ~frame["claim_id"].duplicated(keep=False)),
        ("policy_id_present", lambda frame: frame["policy_id"].notna()),
        ("positive_coverage", lambda frame: frame["coverage_limit"] > 0),
        ("severity_within_coverage", lambda frame: frame["severity_estimate"].between(0, frame["coverage_limit"])),
        ("settlement_within_coverage", lambda frame: frame["settlement_amount"].between(0, frame["coverage_limit"])),
        (
            "closure_date_consistent",
            lambda frame: (
                ((frame["closed_flag"] == 1) & frame["closed_at"].notna() & (frame["closed_at"] >= frame["reported_at"]))
                | ((frame["closed_flag"] == 0) & frame["closed_at"].isna())
            ),
        ),
    ]
    results = []
    for name, rule in rules:
        valid = rule(claims).fillna(False)
        failures = int((~valid).sum())
        results.append(QualityCheck(name, failures == 0, failures, failures / len(claims)))
    return results

