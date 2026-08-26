from dataclasses import asdict

from claims_ai import generate_claims, run_workflow


claims = generate_claims(10_000, seed=42)
result = run_workflow(claims)

print(asdict(result.metrics))
print(result.review_queue.head(10).to_string(index=False))

