# insurance-claims-ai

An end-to-end insurance claims operations project combining synthetic relational data, SQL analytics, delay-risk modelling, monitoring, and human-review controls.

The model predicts whether an open claim is at risk of exceeding its service target. It is a workflow-prioritization signal—not a coverage, settlement, fraud, or denial decision. High-risk claims enter a review queue with explicit reasons so an adjuster remains accountable for the next action.

## What is included

### Analytics

- Reproducible synthetic claim, policy, activity, and payment features.
- Data-quality rules for identifiers, dates, financial bounds, and required fields.
- SQL schema plus operational queries for backlog, SLA, severity, and bottlenecks.
- Cycle-time, closure, backlog, and segment performance measures.

### AI and controls

- Interpretable logistic-regression delay-risk model.
- Optional XGBoost estimator behind the same interface.
- Time-based train/test split to reduce look-ahead leakage.
- ROC AUC, average precision, Brier score, and threshold diagnostics.
- Population Stability Index and segment-level performance monitoring.
- Human-review queue for high delay risk, severe claims, fraud indicators, and data exceptions.
- Generated model card documenting intended use and prohibited uses.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
claims-ai-demo --claims 10000 --seed 42
pytest
```

Optional boosted-tree model:

```bash
pip install -e ".[xgboost]"
claims-ai-demo --model xgboost
```

## Example workflow

```python
from claims_ai import generate_claims, run_workflow

claims = generate_claims(10_000, seed=42)
result = run_workflow(claims, estimator="logistic")

print(result.metrics)
print(result.review_queue.head())
print(result.model_card["prohibited_uses"])
```

## Responsible-use design

The predictor is limited to operational delay. Protected characteristics are not generated or used. Region and channel are retained because they describe operating processes, but their segment metrics are monitored. Predictions never automatically change coverage, reserve, settlement, investigation, or customer eligibility. Missing or invalid source data creates a review reason instead of silently defaulting to a low-risk score.

## Repository layout

```text
src/claims_ai/synthetic.py    reproducible portfolio generation
src/claims_ai/quality.py      data-quality rules
src/claims_ai/model.py        delay-risk estimators and evaluation
src/claims_ai/monitoring.py   drift and segment monitoring
src/claims_ai/workflow.py     review queue and model card
sql/                          relational schema and operating queries
tests/                        data, model, and control tests
```

## Limitations

All data are synthetic. Reported performance demonstrates the pipeline, not expected performance on an insurer's claims. A real deployment would require privacy review, target and feature validation, temporal back-testing, calibration, fairness review, change control, access controls, and ongoing human oversight.

