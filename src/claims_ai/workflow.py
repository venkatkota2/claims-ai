"""Operational review workflow and model documentation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .model import DelayRiskModel, ModelMetrics, evaluate_predictions, temporal_split
from .monitoring import population_stability_index, segment_performance
from .quality import QualityCheck, run_quality_checks


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
    if len(claims) != len(probability):
        raise ValueError("claims and probability must have equal lengths")
    frame = claims.copy()
    frame["delay_risk_score"] = np.asarray(probability)
    reasons = []
    for row in frame.itertuples():
        row_reasons = []
        if row.delay_risk_score >= risk_threshold:
            row_reasons.append("high_delay_risk")
        if row.severity_estimate >= severity_threshold:
            row_reasons.append("high_severity")
        if row.fraud_indicator == 1:
            row_reasons.append("fraud_indicator")
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
        "target": "Claim exceeds its line-of-business service target.",
        "training_rows": training_rows,
        "metrics": asdict(metrics),
        "human_oversight": "An adjuster reviews every queued claim and owns all claim decisions.",
        "prohibited_uses": [
            "coverage or eligibility decisions",
            "automatic claim denial",
            "automatic settlement or reserve changes",
            "fraud determinations",
        ],
        "limitations": "Trained entirely on synthetic data; results do not estimate real-world performance.",
    }


def run_workflow(
    claims: pd.DataFrame,
    *,
    estimator: str = "logistic",
    risk_threshold: float = 0.70,
) -> WorkflowResult:
    checks = run_quality_checks(claims)
    if any(not check.passed for check in checks):
        raise ValueError("claims failed data-quality checks")
    train, test = temporal_split(claims)
    model = DelayRiskModel(estimator).fit(train)
    train_probability = model.predict_proba(train)
    test_probability = model.predict_proba(test)
    metrics = evaluate_predictions(test["delay_flag"], test_probability, threshold=risk_threshold)
    queue = build_review_queue(test, test_probability, risk_threshold=risk_threshold)
    monitoring: dict[str, object] = {
        "score_psi": population_stability_index(train_probability, test_probability),
        "region": segment_performance(test, test_probability, segment="region", threshold=risk_threshold),
        "line_of_business": segment_performance(
            test,
            test_probability,
            segment="line_of_business",
            threshold=risk_threshold,
        ),
    }
    return WorkflowResult(
        model=model,
        metrics=metrics,
        review_queue=queue,
        quality_checks=checks,
        monitoring=monitoring,
        model_card=_model_card(estimator, metrics, len(train)),
    )

