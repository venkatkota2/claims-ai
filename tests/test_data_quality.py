import pandas as pd

from claims_ai import generate_claims, run_quality_checks


def test_synthetic_claims_are_reproducible_and_valid():
    first = generate_claims(500, seed=9)
    second = generate_claims(500, seed=9)

    pd.testing.assert_frame_equal(first, second)
    assert first["claim_id"].is_unique
    assert set(first["delay_flag"].unique()) == {0, 1}
    assert all(check.passed for check in run_quality_checks(first))


def test_quality_check_detects_financial_exception():
    claims = generate_claims(200, seed=3)
    claims.loc[0, "settlement_amount"] = claims.loc[0, "coverage_limit"] + 1
    checks = {check.rule: check for check in run_quality_checks(claims)}

    assert not checks["settlement_within_coverage"].passed
    assert checks["settlement_within_coverage"].failure_count == 1

