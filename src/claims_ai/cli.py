from __future__ import annotations

import argparse
from dataclasses import asdict

from .synthetic import generate_claims
from .workflow import run_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the claims delay-risk workflow")
    parser.add_argument("--claims", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["logistic", "xgboost"], default="logistic")
    args = parser.parse_args()

    result = run_workflow(generate_claims(args.claims, seed=args.seed), estimator=args.model)
    print("test metrics")
    for name, value in asdict(result.metrics).items():
        print(f"  {name:28s} {value:.4f}")
    print(f"review queue                 {len(result.review_queue):,} claims")
    print(f"score PSI                    {result.monitoring['score_psi']:.4f}")
    print("top model features")
    print(result.model.feature_importance().head(8).to_string(index=False))


if __name__ == "__main__":
    main()

