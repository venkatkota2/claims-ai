from claims_ai import build_review_queue, generate_claims, run_workflow


def test_delay_risk_workflow_produces_controls_and_useful_signal():
    claims = generate_claims(2_000, seed=21)
    result = run_workflow(claims)

    assert result.metrics.roc_auc > 0.65
    assert result.metrics.average_precision > result.metrics.positive_rate
    assert 0 <= result.monitoring["score_psi"]
    assert len(result.review_queue) > 0
    assert "automatic claim denial" in result.model_card["prohibited_uses"]
    assert not result.model.feature_importance().empty


def test_human_review_reasons_are_explicit():
    claims = generate_claims(200, seed=5)
    probabilities = [0.99] * len(claims)
    queue = build_review_queue(claims, probabilities)

    assert len(queue) == len(claims)
    assert queue["review_reasons"].str.contains("high_delay_risk").all()

