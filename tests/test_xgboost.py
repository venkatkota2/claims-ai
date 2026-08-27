import pytest

from claims_ai import generate_synthetic_data, run_workflow

pytest.importorskip("xgboost")


def test_optional_xgboost_workflow_smoke():
    result = run_workflow(generate_synthetic_data(700, seed=31), estimator="xgboost")

    assert 0.0 <= result.metrics.brier_score <= 1.0
    assert result.monitoring["open_scored_rows"] > 0
    assert (result.review_queue["closed_flag"] == 0).all()
