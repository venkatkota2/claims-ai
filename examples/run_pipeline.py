from dataclasses import asdict

from claims_ai import generate_synthetic_data, run_workflow

data = generate_synthetic_data(10_000, seed=42)
result = run_workflow(data)

print(asdict(result.metrics))
print(result.review_queue.head(10).to_string(index=False))
