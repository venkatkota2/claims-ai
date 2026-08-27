import numpy as np
import pandas as pd
import pytest

from claims_ai import (
    build_feature_table,
    build_review_queue,
    evaluate_predictions,
    generate_synthetic_data,
    run_workflow,
)
from claims_ai.monitoring import population_stability_index, segment_performance


def test_delay_risk_workflow_separates_evaluation_from_open_scoring():
    data = generate_synthetic_data(2_000, seed=21)
    result = run_workflow(data)

    assert result.metrics.roc_auc > 0.65
    assert result.metrics.average_precision > result.metrics.positive_rate
    assert result.monitoring["train_to_open_score_psi"] >= 0
    assert result.monitoring["train_to_open_score_psi_status"] in {
        "stable",
        "monitor",
        "investigate",
    }
    assert result.monitoring["open_scored_rows"] > 0
    assert len(result.review_queue) > 0
    assert (result.review_queue["closed_flag"] == 0).all()
    assert "automatic claim denial" in result.model_card["prohibited_uses"]
    assert "not causal" in result.model_card["explanation_note"].lower()
    assert "association" in result.model.feature_importance()["interpretation"].iloc[0]


def test_closed_claims_never_enter_operational_review_queue():
    data = generate_synthetic_data(300, seed=5)
    features = build_feature_table(data)
    probabilities = np.full(len(features), 0.99)
    queue = build_review_queue(features, probabilities)
    open_ids = set(features.loc[features["closed_flag"] == 0, "claim_id"])
    closed_ids = set(features.loc[features["closed_flag"] == 1, "claim_id"])

    assert set(queue["claim_id"]) == open_ids
    assert set(queue["claim_id"]).isdisjoint(closed_ids)
    assert queue["review_reasons"].str.contains("high_delay_risk").all()


def test_claim_already_past_target_is_reviewed_without_a_high_model_score():
    claim = pd.DataFrame(
        {
            "claim_id": ["CLM-1"],
            "closed_flag": [0],
            "line_of_business": ["auto"],
            "region": ["ontario"],
            "severity_estimate": [10_000.0],
            "claim_age_days": [31],
            "target_days": [30],
            "days_since_last_activity": [2],
            "missing_documents": [0],
            "fraud_indicator": [0],
        }
    )

    queue = build_review_queue(claim, np.array([0.10]))

    assert queue["claim_id"].tolist() == ["CLM-1"]
    assert queue["review_reasons"].iloc[0] == "past_service_target"


def test_review_and_monitoring_controls_reject_invalid_inputs():
    data = generate_synthetic_data(300, seed=12)
    features = build_feature_table(data)
    open_claims = features[features["closed_flag"] == 0]
    with pytest.raises(ValueError, match="severity_threshold"):
        build_review_queue(
            open_claims,
            np.full(len(open_claims), 0.5),
            severity_threshold=float("nan"),
        )
    with pytest.raises(ValueError, match="PSI"):
        population_stability_index(np.array([0.1, np.nan]), np.array([0.2]))

    labeled = features[features["closed_flag"] == 1].head(3)
    with pytest.raises(ValueError, match="probabilities"):
        segment_performance(labeled, np.array([0.5]), segment="region")


@pytest.mark.parametrize(
    ("actual", "probability", "threshold", "message"),
    [
        ([], [], 0.7, "non-zero"),
        ([0, 1], [0.2], 0.7, "same non-zero"),
        ([0, 1], [0.2, np.nan], 0.7, "finite"),
        ([0, 1], [0.2, np.inf], 0.7, "finite"),
        ([0, 1], [0.2, 1.2], 0.7, "invalid probability"),
        ([0, 2], [0.2, 0.8], 0.7, "binary"),
        ([1, 1], [0.2, 0.8], 0.7, "both target classes"),
        ([0, 1], [0.2, 0.8], 1.0, "invalid probability"),
    ],
)
def test_prediction_evaluation_rejects_invalid_inputs(actual, probability, threshold, message):
    with pytest.raises(ValueError, match=message):
        evaluate_predictions(actual, np.asarray(probability), threshold=threshold)
