"""Delay-risk model training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "coverage_limit",
    "severity_estimate",
    "activity_count_7d",
    "days_since_last_activity",
    "missing_documents",
    "adjuster_open_load",
    "fraud_indicator",
    "complexity_score",
]
CATEGORICAL_FEATURES = ["line_of_business", "channel", "region"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class ModelMetrics:
    roc_auc: float
    average_precision: float
    brier_score: float
    positive_rate: float
    review_rate_at_threshold: float
    recall_at_threshold: float


def temporal_split(claims: pd.DataFrame, test_fraction: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.1 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.1 and 0.5")
    ordered = claims.sort_values(["reported_at", "claim_id"]).reset_index(drop=True)
    cut = int(len(ordered) * (1.0 - test_fraction))
    if cut == 0 or cut == len(ordered):
        raise ValueError("not enough rows for a temporal split")
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def _estimator(kind: str, random_state: int):
    if kind == "logistic":
        return LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=random_state)
    if kind == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError("install the 'xgboost' optional dependency") from error
        return XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=random_state,
        )
    raise ValueError("estimator must be 'logistic' or 'xgboost'")


class DelayRiskModel:
    def __init__(self, estimator: str = "logistic", *, random_state: int = 42) -> None:
        numeric = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transform = ColumnTransformer(
            [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
        )
        self.estimator_name = estimator
        self.pipeline = Pipeline(
            [("features", transform), ("model", _estimator(estimator, random_state))]
        )
        self.fitted = False

    def fit(self, claims: pd.DataFrame) -> "DelayRiskModel":
        _require_columns(claims, MODEL_FEATURES + ["delay_flag"])
        if claims["delay_flag"].nunique() < 2:
            raise ValueError("training data must contain both target classes")
        self.pipeline.fit(claims[MODEL_FEATURES], claims["delay_flag"])
        self.fitted = True
        return self

    def predict_proba(self, claims: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("fit the model before scoring")
        _require_columns(claims, MODEL_FEATURES)
        return self.pipeline.predict_proba(claims[MODEL_FEATURES])[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("fit the model before inspecting it")
        names = self.pipeline.named_steps["features"].get_feature_names_out()
        estimator = self.pipeline.named_steps["model"]
        values = (
            np.abs(estimator.coef_[0])
            if hasattr(estimator, "coef_")
            else estimator.feature_importances_
        )
        return (
            pd.DataFrame({"feature": names, "importance": values})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


def _require_columns(claims: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(claims.columns))
    if missing:
        raise ValueError(f"missing model columns: {', '.join(missing)}")


def evaluate_predictions(
    actual: pd.Series | np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float = 0.70,
) -> ModelMetrics:
    actual_array = np.asarray(actual, dtype=int)
    probability_array = np.asarray(probability, dtype=float)
    if len(actual_array) != len(probability_array) or len(actual_array) == 0:
        raise ValueError("actual and probability must have the same non-zero length")
    if not 0 < threshold < 1 or np.any((probability_array < 0) | (probability_array > 1)):
        raise ValueError("invalid probability or threshold")
    reviewed = probability_array >= threshold
    positives = actual_array == 1
    recall = float(np.sum(reviewed & positives) / max(1, np.sum(positives)))
    return ModelMetrics(
        roc_auc=float(roc_auc_score(actual_array, probability_array)),
        average_precision=float(average_precision_score(actual_array, probability_array)),
        brier_score=float(brier_score_loss(actual_array, probability_array)),
        positive_rate=float(np.mean(positives)),
        review_rate_at_threshold=float(np.mean(reviewed)),
        recall_at_threshold=recall,
    )

