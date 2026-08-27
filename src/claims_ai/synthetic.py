"""Relational synthetic claims data and leakage-aware feature engineering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticClaimsData:
    """Synthetic source tables observed at one operational as-of date."""

    policies: pd.DataFrame
    claims: pd.DataFrame
    activities: pd.DataFrame
    payments: pd.DataFrame
    as_of_date: pd.Timestamp


def _require_columns(frame: pd.DataFrame, columns: set[str], table: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{table} is missing columns: {', '.join(missing)}")


def generate_policies(count: int, *, seed: int = 42) -> pd.DataFrame:
    """Generate stable policy-level attributes."""
    if count <= 0:
        raise ValueError("policy count must be positive")
    rng = np.random.default_rng(seed)
    lines = rng.choice(["auto", "property", "liability"], count, p=[0.52, 0.31, 0.17])
    regions = rng.choice(
        ["west", "ontario", "quebec", "atlantic"], count, p=[0.24, 0.43, 0.23, 0.10]
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
    return pd.DataFrame(
        {
            "policy_id": [f"POL-{index:06d}" for index in range(1, count + 1)],
            "line_of_business": lines,
            "region": regions,
            "coverage_limit": coverage_limit,
        }
    )


def generate_claims(
    policies: pd.DataFrame,
    count: int,
    *,
    as_of_date: str | pd.Timestamp = "2026-01-01",
    seed: int = 43,
) -> pd.DataFrame:
    """Generate claim records whose labels are derived from realized cycle time."""
    _require_columns(
        policies,
        {"policy_id", "line_of_business", "region", "coverage_limit"},
        "policies",
    )
    if count < 100:
        raise ValueError("claim count must be at least 100")
    if policies.empty or not policies["policy_id"].is_unique:
        raise ValueError("policies must contain unique rows")

    rng = np.random.default_rng(seed)
    as_of = pd.Timestamp(as_of_date)
    selected = policies.iloc[rng.integers(0, len(policies), count)].reset_index(drop=True)
    lines = selected["line_of_business"].to_numpy()
    regions = selected["region"].to_numpy()
    coverage_limit = selected["coverage_limit"].to_numpy(dtype=float)
    channels = rng.choice(["broker", "direct", "digital"], count, p=[0.45, 0.32, 0.23])
    reported_at = pd.Series(
        as_of - pd.to_timedelta(rng.integers(1, 731, count), unit="D"), dtype="datetime64[ns]"
    )

    complexity = np.clip(
        rng.beta(2.2, 3.2, count) + 0.18 * (lines == "liability") + 0.08 * (lines == "property"),
        0.0,
        1.0,
    )
    severity_estimate = np.minimum(
        coverage_limit,
        coverage_limit * rng.lognormal(mean=-2.7 + 1.2 * complexity, sigma=0.7, size=count),
    )
    missing_documents = rng.binomial(3, np.clip(0.08 + 0.34 * complexity, 0.0, 0.8), count)
    adjuster_open_load = (
        np.clip(rng.normal(36, 11, count) + 11 * (regions == "ontario"), 5, 95).round().astype(int)
    )
    fraud_indicator = rng.binomial(1, np.clip(0.012 + 0.055 * complexity, 0.0, 0.2), count)
    target_days = np.select([lines == "auto", lines == "property"], [30, 45], default=60)

    log_cycle_ratio = (
        -0.95
        + 0.90 * complexity
        + 0.18 * missing_documents
        + 0.012 * (adjuster_open_load - 35)
        + 0.30 * (channels == "broker")
        + 0.25 * (lines == "liability")
        + 0.18 * fraud_indicator
        + rng.normal(0.0, 0.30, count)
    )
    realized_cycle_days = np.clip(np.rint(target_days * np.exp(log_cycle_ratio)), 1, 365).astype(
        int
    )
    realized_closed_at = reported_at + pd.to_timedelta(realized_cycle_days, unit="D")
    closed_flag = (realized_closed_at <= as_of).astype(int)
    closed_at = realized_closed_at.where(closed_flag == 1, pd.NaT)

    cycle_time_days = pd.Series(realized_cycle_days, dtype="Float64").where(closed_flag == 1)
    delay_flag = pd.Series(realized_cycle_days > target_days, dtype="boolean").where(
        closed_flag == 1
    )
    settlement_amount = np.where(
        closed_flag == 1,
        np.minimum(coverage_limit, severity_estimate * rng.uniform(0.72, 1.08, count)),
        0.0,
    ).round(2)

    planned_snapshot = reported_at + pd.to_timedelta(np.maximum(1, target_days // 2), unit="D")
    last_pre_close = realized_closed_at - pd.Timedelta(days=1)
    historical_snapshot = planned_snapshot.where(planned_snapshot <= last_pre_close, last_pre_close)
    historical_snapshot = historical_snapshot.where(historical_snapshot >= reported_at, reported_at)
    snapshot_at = historical_snapshot.where(closed_flag == 1, as_of)

    return pd.DataFrame(
        {
            "claim_id": [f"CLM-{index:07d}" for index in range(1, count + 1)],
            "policy_id": selected["policy_id"].to_numpy(),
            "reported_at": reported_at,
            "closed_at": pd.to_datetime(closed_at),
            "snapshot_at": pd.to_datetime(snapshot_at),
            "channel": channels,
            "severity_estimate": severity_estimate.round(2),
            "settlement_amount": settlement_amount,
            "missing_documents": missing_documents,
            "adjuster_open_load": adjuster_open_load,
            "fraud_indicator": fraud_indicator,
            "complexity_score": complexity.round(6),
            "target_days": target_days,
            "cycle_time_days": cycle_time_days,
            "closed_flag": closed_flag,
            "delay_flag": delay_flag.astype("Int64"),
        }
    )


def generate_activities(claims: pd.DataFrame, *, seed: int = 44) -> pd.DataFrame:
    """Generate activities available no later than each claim snapshot."""
    _require_columns(
        claims,
        {
            "claim_id",
            "reported_at",
            "snapshot_at",
            "missing_documents",
            "complexity_score",
        },
        "claims",
    )
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    activity_id = 1
    for claim in claims.itertuples(index=False):
        age_days = max(0, int((claim.snapshot_at - claim.reported_at).days))
        gap_days = min(
            age_days,
            round(
                rng.gamma(1.5, 1.8) + 3.0 * claim.missing_documents + 5.0 * claim.complexity_score
            ),
        )
        last_activity = claim.snapshot_at - pd.Timedelta(days=gap_days)
        activity_count = max(
            1,
            int(
                rng.poisson(
                    2.0 + min(age_days, 180) / 20.0 * max(0.35, 1.2 - 0.5 * claim.complexity_score)
                )
            ),
        )
        available_days = max(0, int((last_activity - claim.reported_at).days))
        offsets = (
            rng.integers(0, available_days + 1, activity_count - 1)
            if activity_count > 1
            else np.array([], dtype=int)
        )
        timestamps = [claim.reported_at + pd.Timedelta(days=int(offset)) for offset in offsets]
        timestamps.append(last_activity)
        for activity_at in sorted(timestamps):
            records.append(
                {
                    "activity_id": f"ACT-{activity_id:09d}",
                    "claim_id": claim.claim_id,
                    "activity_at": activity_at,
                    "activity_type": rng.choice(
                        ["note", "document_request", "customer_contact", "assessment"]
                    ),
                    "actor_type": rng.choice(["adjuster", "customer", "vendor"]),
                }
            )
            activity_id += 1
    return pd.DataFrame.from_records(records)


def generate_payments(claims: pd.DataFrame, *, seed: int = 45) -> pd.DataFrame:
    """Generate claim payments with stable claim foreign keys."""
    _require_columns(
        claims,
        {
            "claim_id",
            "reported_at",
            "closed_at",
            "snapshot_at",
            "closed_flag",
            "settlement_amount",
            "severity_estimate",
        },
        "claims",
    )
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    payment_id = 1
    for claim in claims.itertuples(index=False):
        if claim.closed_flag == 1:
            total = float(claim.settlement_amount)
            payment_count = int(rng.integers(1, 4)) if total > 0 else 0
            final_date = claim.closed_at
        elif rng.random() < 0.35:
            total = float(claim.severity_estimate * rng.uniform(0.05, 0.35))
            payment_count = int(rng.integers(1, 3))
            final_date = claim.snapshot_at
        else:
            continue
        if payment_count == 0:
            continue

        weights = rng.dirichlet(np.ones(payment_count))
        amounts = np.round(total * weights, 2)
        amounts[-1] = round(total - float(np.sum(amounts[:-1])), 2)
        available_days = max(0, int((final_date - claim.reported_at).days))
        offsets = np.sort(rng.integers(0, available_days + 1, payment_count))
        for amount, offset in zip(amounts, offsets, strict=True):
            records.append(
                {
                    "payment_id": f"PAY-{payment_id:09d}",
                    "claim_id": claim.claim_id,
                    "paid_at": claim.reported_at + pd.Timedelta(days=int(offset)),
                    "amount": max(0.0, float(amount)),
                }
            )
            payment_id += 1
    return pd.DataFrame.from_records(
        records, columns=["payment_id", "claim_id", "paid_at", "amount"]
    )


def build_feature_table(data: SyntheticClaimsData) -> pd.DataFrame:
    """Build as-of model features from relational activity and payment records."""
    frame = data.claims.merge(data.policies, on="policy_id", how="left", validate="many_to_one")
    snapshots = frame[["claim_id", "snapshot_at", "reported_at"]]

    activities = data.activities.merge(snapshots, on="claim_id", how="left", validate="many_to_one")
    observed_activities = activities[activities["activity_at"] <= activities["snapshot_at"]]
    recent = observed_activities[
        observed_activities["activity_at"]
        >= observed_activities["snapshot_at"] - pd.Timedelta(days=7)
    ]
    recent_count = recent.groupby("claim_id").size().rename("activity_count_7d")
    last_activity = observed_activities.groupby("claim_id")["activity_at"].max()

    payments = data.payments.merge(
        snapshots[["claim_id", "snapshot_at"]],
        on="claim_id",
        how="left",
        validate="many_to_one",
    )
    observed_payments = payments[payments["paid_at"] <= payments["snapshot_at"]]
    paid_to_date = observed_payments.groupby("claim_id")["amount"].sum().rename("payments_to_date")

    frame = frame.join(recent_count, on="claim_id").join(last_activity, on="claim_id")
    frame = frame.join(paid_to_date, on="claim_id")
    frame["claim_age_days"] = (frame["snapshot_at"] - frame["reported_at"]).dt.days
    frame["days_since_last_activity"] = (frame["snapshot_at"] - frame["activity_at"]).dt.days
    frame["activity_count_7d"] = frame["activity_count_7d"].fillna(0).astype(int)
    frame["days_since_last_activity"] = frame["days_since_last_activity"].fillna(
        frame["claim_age_days"]
    )
    frame["payments_to_date"] = frame["payments_to_date"].fillna(0.0)
    return frame.drop(columns=["activity_at"]).sort_values("claim_id").reset_index(drop=True)


def generate_synthetic_data(
    claim_count: int = 10_000,
    *,
    seed: int = 42,
    as_of_date: str | pd.Timestamp = "2026-01-01",
) -> SyntheticClaimsData:
    """Generate a reproducible relational portfolio and operational snapshot."""
    policy_count = max(50, claim_count // 2)
    policies = generate_policies(policy_count, seed=seed)
    claims = generate_claims(policies, claim_count, as_of_date=as_of_date, seed=seed + 1)
    activities = generate_activities(claims, seed=seed + 2)
    payments = generate_payments(claims, seed=seed + 3)
    return SyntheticClaimsData(
        policies=policies,
        claims=claims,
        activities=activities,
        payments=payments,
        as_of_date=pd.Timestamp(as_of_date),
    )
