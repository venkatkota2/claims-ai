from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_postgresql_schema_matches_generated_identifier_and_feature_contract():
    schema = (ROOT / "sql" / "schema.sql").read_text()

    for column in (
        "snapshot_at",
        "settlement_amount",
        "missing_documents",
        "adjuster_open_load",
        "fraud_indicator",
        "complexity_score",
        "cycle_time_days",
        "closed_flag",
        "delay_flag",
    ):
        assert column in schema
    assert "activity_id VARCHAR" in schema
    assert "payment_id VARCHAR" in schema


def test_operational_queries_use_an_explicit_as_of_date():
    queries = (ROOT / "sql" / "operations_queries.sql").read_text()

    assert ":as_of_date" in queries
    assert "CURRENT_DATE" not in queries
    assert "CURRENT_TIMESTAMP" not in queries
