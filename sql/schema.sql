CREATE TABLE policies (
    policy_id VARCHAR(20) PRIMARY KEY,
    line_of_business VARCHAR(30) NOT NULL,
    region VARCHAR(30) NOT NULL,
    coverage_limit DECIMAL(16, 2) NOT NULL CHECK (coverage_limit > 0)
);

CREATE TABLE claims (
    claim_id VARCHAR(20) PRIMARY KEY,
    policy_id VARCHAR(20) NOT NULL REFERENCES policies(policy_id),
    reported_at TIMESTAMP NOT NULL,
    closed_at TIMESTAMP NULL,
    snapshot_at TIMESTAMP NOT NULL,
    channel VARCHAR(30) NOT NULL,
    severity_estimate DECIMAL(16, 2) NOT NULL CHECK (severity_estimate >= 0),
    settlement_amount DECIMAL(16, 2) NOT NULL CHECK (settlement_amount >= 0),
    missing_documents INTEGER NOT NULL CHECK (missing_documents >= 0),
    adjuster_open_load INTEGER NOT NULL CHECK (adjuster_open_load >= 0),
    fraud_indicator SMALLINT NOT NULL CHECK (fraud_indicator IN (0, 1)),
    complexity_score DECIMAL(8, 6) NOT NULL CHECK (complexity_score BETWEEN 0 AND 1),
    target_days INTEGER NOT NULL CHECK (target_days > 0),
    cycle_time_days INTEGER NULL CHECK (cycle_time_days > 0),
    closed_flag SMALLINT NOT NULL CHECK (closed_flag IN (0, 1)),
    delay_flag SMALLINT NULL CHECK (delay_flag IN (0, 1)),
    CHECK (snapshot_at >= reported_at),
    CHECK (closed_at IS NULL OR closed_at >= reported_at),
    CHECK (closed_flag = 1 OR settlement_amount = 0),
    CHECK (
        delay_flag IS NULL
        OR delay_flag = CASE WHEN cycle_time_days > target_days THEN 1 ELSE 0 END
    ),
    CHECK (
        (closed_flag = 1 AND closed_at IS NOT NULL AND cycle_time_days IS NOT NULL AND delay_flag IS NOT NULL)
        OR
        (closed_flag = 0 AND closed_at IS NULL AND cycle_time_days IS NULL AND delay_flag IS NULL)
    )
);

CREATE TABLE claim_activities (
    activity_id VARCHAR(20) PRIMARY KEY,
    claim_id VARCHAR(20) NOT NULL REFERENCES claims(claim_id),
    activity_at TIMESTAMP NOT NULL,
    activity_type VARCHAR(50) NOT NULL,
    actor_type VARCHAR(30) NOT NULL
);

CREATE TABLE claim_payments (
    payment_id VARCHAR(20) PRIMARY KEY,
    claim_id VARCHAR(20) NOT NULL REFERENCES claims(claim_id),
    paid_at TIMESTAMP NOT NULL,
    amount DECIMAL(16, 2) NOT NULL CHECK (amount >= 0)
);

CREATE INDEX idx_claims_status ON claims (closed_at, reported_at);
CREATE INDEX idx_activities_claim_time ON claim_activities (claim_id, activity_at);
