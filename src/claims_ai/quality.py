"""Relational and model-feature data-quality controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .synthetic import SyntheticClaimsData, build_feature_table


@dataclass(frozen=True)
class QualityCheck:
    rule: str
    passed: bool
    failure_count: int
    failure_rate: float


def _check(rule: str, valid: pd.Series | np.ndarray, total: int) -> QualityCheck:
    valid_array = pd.Series(valid, dtype="boolean").fillna(False)
    failures = int((~valid_array).sum())
    return QualityCheck(rule, failures == 0, failures, failures / max(1, total))


def _require_columns(frame: pd.DataFrame, required: set[str], table: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{table} is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{table} must not be empty")


def run_quality_checks(data: SyntheticClaimsData) -> list[QualityCheck]:
    """Validate relational integrity, operational dates, amounts, and features."""
    policies, claims = data.policies, data.claims
    activities, payments = data.activities, data.payments
    _require_columns(
        policies,
        {"policy_id", "line_of_business", "region", "coverage_limit"},
        "policies",
    )
    _require_columns(
        claims,
        {
            "claim_id",
            "policy_id",
            "reported_at",
            "closed_at",
            "snapshot_at",
            "severity_estimate",
            "settlement_amount",
            "target_days",
            "cycle_time_days",
            "closed_flag",
            "delay_flag",
        },
        "claims",
    )
    _require_columns(activities, {"activity_id", "claim_id", "activity_at"}, "activities")
    _require_columns(payments, {"payment_id", "claim_id", "paid_at", "amount"}, "payments")

    results = [
        _check("policy_id_unique", ~policies["policy_id"].duplicated(keep=False), len(policies)),
        _check("claim_id_unique", ~claims["claim_id"].duplicated(keep=False), len(claims)),
        _check(
            "activity_id_unique",
            ~activities["activity_id"].duplicated(keep=False),
            len(activities),
        ),
        _check("payment_id_unique", ~payments["payment_id"].duplicated(keep=False), len(payments)),
        _check(
            "claim_policy_foreign_key",
            claims["policy_id"].isin(policies["policy_id"]),
            len(claims),
        ),
        _check(
            "activity_claim_foreign_key",
            activities["claim_id"].isin(claims["claim_id"]),
            len(activities),
        ),
        _check(
            "payment_claim_foreign_key",
            payments["claim_id"].isin(claims["claim_id"]),
            len(payments),
        ),
        _check(
            "positive_coverage",
            np.isfinite(policies["coverage_limit"]) & (policies["coverage_limit"] > 0),
            len(policies),
        ),
    ]

    claim_policy = claims.merge(
        policies[["policy_id", "coverage_limit"]],
        on="policy_id",
        how="left",
        validate="many_to_one",
    )
    results.extend(
        [
            _check(
                "severity_within_coverage",
                np.isfinite(claim_policy["severity_estimate"])
                & claim_policy["severity_estimate"].between(
                    0, claim_policy["coverage_limit"], inclusive="both"
                ),
                len(claims),
            ),
            _check(
                "settlement_within_coverage",
                np.isfinite(claim_policy["settlement_amount"])
                & claim_policy["settlement_amount"].between(
                    0, claim_policy["coverage_limit"], inclusive="both"
                ),
                len(claims),
            ),
            _check(
                "closure_consistent",
                (
                    (claims["closed_flag"] == 1)
                    & claims["closed_at"].notna()
                    & (claims["closed_at"] >= claims["reported_at"])
                    & claims["cycle_time_days"].notna()
                    & claims["delay_flag"].notna()
                )
                | (
                    (claims["closed_flag"] == 0)
                    & claims["closed_at"].isna()
                    & claims["cycle_time_days"].isna()
                    & claims["delay_flag"].isna()
                ),
                len(claims),
            ),
            _check(
                "delay_target_consistent",
                (claims["closed_flag"] == 0)
                | (
                    claims["delay_flag"].astype("Float64")
                    == (claims["cycle_time_days"] > claims["target_days"]).astype(float)
                ),
                len(claims),
            ),
            _check(
                "snapshot_date_consistent",
                (claims["snapshot_at"] >= claims["reported_at"])
                & (claims["snapshot_at"] <= data.as_of_date),
                len(claims),
            ),
        ]
    )

    activity_dates = activities.merge(
        claims[["claim_id", "reported_at", "snapshot_at"]],
        on="claim_id",
        how="left",
        validate="many_to_one",
    )
    payment_dates = payments.merge(
        claims[["claim_id", "reported_at", "closed_at", "closed_flag"]],
        on="claim_id",
        how="left",
        validate="many_to_one",
    )
    payment_dates["latest_valid_payment_date"] = payment_dates["closed_at"].where(
        payment_dates["closed_flag"] == 1, data.as_of_date
    )
    results.extend(
        [
            _check(
                "activity_date_consistent",
                activity_dates["activity_at"].between(
                    activity_dates["reported_at"], activity_dates["snapshot_at"], inclusive="both"
                ),
                len(activities),
            ),
            _check(
                "payment_date_consistent",
                payment_dates["paid_at"].between(
                    payment_dates["reported_at"],
                    payment_dates["latest_valid_payment_date"],
                    inclusive="both",
                ),
                len(payments),
            ),
            _check(
                "payment_amount_non_negative",
                np.isfinite(payments["amount"]) & (payments["amount"] >= 0),
                len(payments),
            ),
        ]
    )

    closed = claims[claims["closed_flag"] == 1]
    paid = payments.groupby("claim_id")["amount"].sum()
    closed_paid = closed["claim_id"].map(paid).fillna(0.0)
    results.append(
        _check(
            "closed_payments_match_settlement",
            np.isclose(closed_paid, closed["settlement_amount"], atol=0.01),
            len(closed),
        )
    )

    features = build_feature_table(data)
    numeric_features = [
        "coverage_limit",
        "severity_estimate",
        "activity_count_7d",
        "days_since_last_activity",
        "payments_to_date",
        "claim_age_days",
        "missing_documents",
        "adjuster_open_load",
        "fraud_indicator",
        "complexity_score",
    ]
    results.append(
        _check(
            "model_features_finite",
            np.isfinite(features[numeric_features].to_numpy(dtype=float)).all(axis=1),
            len(features),
        )
    )
    return results
