from dataclasses import replace

import pandas as pd

from claims_ai import build_feature_table, generate_synthetic_data, run_quality_checks


def test_synthetic_relational_data_are_reproducible_and_valid():
    first = generate_synthetic_data(500, seed=9)
    second = generate_synthetic_data(500, seed=9)

    pd.testing.assert_frame_equal(first.policies, second.policies)
    pd.testing.assert_frame_equal(first.claims, second.claims)
    pd.testing.assert_frame_equal(first.activities, second.activities)
    pd.testing.assert_frame_equal(first.payments, second.payments)
    assert first.policies["policy_id"].is_unique
    assert first.claims["claim_id"].is_unique
    assert first.activities["activity_id"].is_unique
    assert first.payments["payment_id"].is_unique
    assert first.claims["policy_id"].isin(first.policies["policy_id"]).all()
    assert first.activities["claim_id"].isin(first.claims["claim_id"]).all()
    assert first.payments["claim_id"].isin(first.claims["claim_id"]).all()
    assert all(check.passed for check in run_quality_checks(first))


def test_delay_target_is_derived_from_realized_cycle_time():
    data = generate_synthetic_data(500, seed=3)
    labeled = data.claims[data.claims["closed_flag"] == 1]
    open_claims = data.claims[data.claims["closed_flag"] == 0]

    expected = (labeled["cycle_time_days"] > labeled["target_days"]).astype(int)
    assert (labeled["delay_flag"].astype(int) == expected).all()
    assert open_claims["delay_flag"].isna().all()
    assert open_claims["cycle_time_days"].isna().all()


def test_feature_table_derives_activity_payment_and_age_features():
    data = generate_synthetic_data(300, seed=17)
    features = build_feature_table(data)
    claim = features.iloc[0]
    claim_activities = data.activities[data.activities["claim_id"] == claim["claim_id"]]
    recent = claim_activities[
        claim_activities["activity_at"] >= claim["snapshot_at"] - pd.Timedelta(days=7)
    ]
    claim_payments = data.payments[
        (data.payments["claim_id"] == claim["claim_id"])
        & (data.payments["paid_at"] <= claim["snapshot_at"])
    ]

    assert claim["activity_count_7d"] == len(recent)
    assert claim["payments_to_date"] == claim_payments["amount"].sum()
    assert claim["claim_age_days"] == (claim["snapshot_at"] - claim["reported_at"]).days


def test_quality_checks_detect_foreign_key_and_financial_exceptions():
    data = generate_synthetic_data(200, seed=5)
    bad_payments = data.payments.copy()
    bad_payments.loc[0, "claim_id"] = "CLM-DOES-NOT-EXIST"
    bad_claims = data.claims.copy()
    bad_claims.loc[0, "settlement_amount"] = 10_000_000_000.0
    corrupted = replace(data, payments=bad_payments, claims=bad_claims)
    checks = {check.rule: check for check in run_quality_checks(corrupted)}

    assert not checks["payment_claim_foreign_key"].passed
    assert checks["payment_claim_foreign_key"].failure_count == 1
    assert not checks["settlement_within_coverage"].passed
    assert checks["settlement_within_coverage"].failure_count == 1
