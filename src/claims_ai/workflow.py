"""Historical model evaluation and open-claim review prioritization."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .model import DelayRiskModel, ModelMetrics, evaluate_predictions, temporal_split
from .monitoring import population_stability_index, segment_performance
from .quality import QualityCheck, run_quality_checks
from .synthetic import SyntheticClaimsData, build_feature_table


@dataclass(frozen=True)
class WorkflowResult:
    model: DelayRiskModel
    metrics: ModelMetrics
    review_queue: pd.DataFrame
    quality_checks: list[QualityCheck]
    monitoring: dict[str, object]
    model_card: dict[str, object]


def build_review_queue(
    claims: pd.DataFrame,
    probability: np.ndarray,
    *,
    risk_threshold: float = 0.70,
    severity_threshold: float = 250_000,
) -> pd.DataFrame:
    """Prioritize current open claims and explain each review reason."""
    probabilities = np.asarray(probability, dtype=float)
    if len(claims) != len(probabilities):
        raise ValueError("claims and probability must have equal lengths")
    if (
        probabilities.ndim != 1
        or not np.all(np.isfinite(probabilities))
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("probabilities must be a finite vector within [0, 1]")
    if not np.isfinite(risk_threshold) or not 0 < risk_threshold < 1:
        raise ValueError("risk_threshold must be between zero and one")
    required = {
        "claim_id",
        "closed_flag",
        "line_of_business",
        "region",
        "severity_estimate",
        "days_since_last_activity",
        "missing_documents",
        "fraud_indicator",
    }
    missing = sorted(required - set(claims.columns))
    if missing:
        raise ValueError(f"missing review columns: {', '.join(missing)}")

    frame = claims.copy()
    frame["delay_risk_score"] = probabilities
    frame = frame[frame["closed_flag"] == 0].copy()
    reasons = []
    for row in frame.itertuples():
        row_reasons = []
        if row.delay_risk_score >= risk_threshold:
            row_reasons.append("high_delay_risk")
        if row.severity_estimate >= severity_threshold:
            row_reasons.append("high_severity")
        if row.fraud_indicator == 1:
            row_reasons.append("existing_fraud_referral")
        if row.missing_documents > 0:
            row_reasons.append("missing_documents")
        reasons.append(",".join(row_reasons))
    frame["review_reasons"] = reasons
    queue = frame[frame["review_reasons"] != ""].copy()
    queue["priority_score"] = (
        queue["delay_risk_score"]
        + 0.15 * (queue["severity_estimate"] >= severity_threshold)
        + 0.20 * queue["fraud_indicator"]
    )
    columns = [
        "claim_id",
        "closed_flag",
        "line_of_business",
        "region",
        "severity_estimate",
        "days_since_last_activity",
        "delay_risk_score",
        "review_reasons",
        "priority_score",
    ]
    return queue.sort_values(["priority_score", "claim_id"], ascending=[False, True])[columns]


def _model_card(estimator: str, metrics: ModelMetrics, training_rows: int) -> dict[str, object]:
    return {
        "model_name": f"claims-delay-risk-{estimator}",
        "version": "0.1.0",
        "intended_use": "Prioritize open claims for adjuster review when service delay is likely.",
        "target": "For labeled closed claims, cycle_time_days exceeds target_days.",
        "training_rows": training_rows,
        "metrics": asdict(metrics),
        "explanation_note": (
            "Importance values describe model associations, not causal effects or claim decisions."
        ),
        "human_oversight": "An adjuster reviews every queued claim and owns all claim decisions.",
        "prohibited_uses": [
            "coverage or eligibility decisions",
            "automatic claim denial",
            "automatic settlement or reserve changes",
            "fraud determinations",
        ],
        "limitations": (
            "Trained entirely on synthetic data; results do not estimate real performance."
        ),
    }


def run_workflow(
    data: SyntheticClaimsData,
    *,
    estimator: str = "logistic",
    risk_threshold: float = 0.70,
) -> WorkflowResult:
    """Evaluate on historical labels, then score only current open claims."""
    if not isinstance(data, SyntheticClaimsData):
        raise TypeError("run_workflow requires SyntheticClaimsData")
    checks = run_quality_checks(data)
    if any(not check.passed for check in checks):
        raise ValueError("synthetic source tables failed data-quality checks")

    features = build_feature_table(data)
    historical = features[features["closed_flag"] == 1].copy()
    current_open = features[features["closed_flag"] == 0].copy()
    if current_open.empty:
        raise ValueError("operational scoring requires current open claims")

    train, test = temporal_split(historical)
    model = DelayRiskModel(estimator).fit(train)
    train_probability = model.predict_proba(train)
    test_probability = model.predict_proba(test)
    open_probability = model.predict_proba(current_open)
    metrics = evaluate_predictions(test["delay_flag"], test_probability, threshold=risk_threshold)
    queue = build_review_queue(current_open, open_probability, risk_threshold=risk_threshold)
    monitoring: dict[str, object] = {
        "train_to_open_score_psi": population_stability_index(train_probability, open_probability),
        "evaluation_region": segment_performance(
            test, test_probability, segment="region", threshold=risk_threshold
        ),
        "evaluation_line_of_business": segment_performance(
            test,
            test_probability,
            segment="line_of_business",
            threshold=risk_threshold,
        ),
        "historical_test_rows": len(test),
        "open_scored_rows": len(current_open),
    }
    return WorkflowResult(
        model=model,
        metrics=metrics,
        review_queue=queue,
        quality_checks=checks,
        monitoring=monitoring,
        model_card=_model_card(estimator, metrics, len(train)),
    )
