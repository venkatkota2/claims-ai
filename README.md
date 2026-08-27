# insurance-claims-ai

An end-to-end insurance claims operations project combining synthetic relational data, SQL analytics, delay-risk modelling, monitoring, and human-review controls.

The model predicts whether an open claim is at risk of exceeding its service target. It is a workflow-prioritization signal—not a coverage, settlement, fraud, or denial decision. High-risk claims enter a review queue with explicit reasons so an adjuster remains accountable for the next action.

## What is included

### Analytics

- Reproducible policy, claim, activity, and payment source tables with enforced foreign keys.
- As-of feature engineering for claim age, recent activity, activity recency, and payments-to-date.
- Data-quality rules for identifiers, foreign keys, dates, closure/target consistency, financial bounds, and model-feature finiteness.
- SQL schema plus operational queries for backlog, SLA, severity, and bottlenecks.
- Cycle-time, closure, backlog, and segment performance measures.

### AI and controls

- Interpretable logistic-regression delay-risk model.
- Optional XGBoost estimator behind the same interface.
- Time-based train/test split to reduce look-ahead leakage.
- ROC AUC, average precision, Brier score, and threshold diagnostics.
- Population Stability Index and segment-level performance monitoring.
- Open-claim-only human-review queue for high delay risk, severe claims, existing referral flags, and missing documents.
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
from claims_ai import generate_synthetic_data, run_workflow

data = generate_synthetic_data(10_000, seed=42)
result = run_workflow(data, estimator="logistic")

print(result.metrics)
print(result.review_queue.head())
print(result.model_card["prohibited_uses"])
```

## Workflow design

```text
synthetic relational data → as-of feature engineering → quality checks
→ temporal historical train/test → delay-risk validation
→ current open-claim scoring → explained review queue → human decision
```

Historical closed claims have a label defined literally as
`cycle_time_days > target_days`. They are used for temporal training and
evaluation. Current open claims do not require an outcome label and are the
only records eligible for the operational review queue. Activity and payment
features are calculated at each record's snapshot date to avoid using future
events.

## Responsible-use design

The predictor is limited to operational delay. Protected characteristics are not generated or used. Region and channel are retained because they describe operating processes, but their segment metrics are monitored. Predictions never automatically change coverage, reserve, settlement, investigation, fraud status, or customer eligibility. Feature importance is model association, not a causal explanation. Every queued claim remains a human decision.

## Repository layout

```text
src/claims_ai/synthetic.py    relational generation and as-of features
src/claims_ai/quality.py      relational and feature-quality rules
src/claims_ai/model.py        delay-risk estimators and evaluation
src/claims_ai/monitoring.py   drift and segment monitoring
src/claims_ai/workflow.py     review queue and model card
sql/                          relational schema and operating queries
tests/                        data, model, and control tests
```

## Limitations

All data are synthetic. Reported performance demonstrates the pipeline, not expected insurer performance. This is not a coverage, fraud, reserve, settlement, denial, or eligibility model. A real deployment would require privacy review, target and feature validation, temporal back-testing, calibration, fairness review, change control, access controls, and ongoing human oversight.
