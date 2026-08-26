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
    channel VARCHAR(30) NOT NULL,
    severity_estimate DECIMAL(16, 2) NOT NULL CHECK (severity_estimate >= 0),
    target_days INTEGER NOT NULL CHECK (target_days > 0),
    CHECK (closed_at IS NULL OR closed_at >= reported_at)
);

CREATE TABLE claim_activities (
    activity_id BIGINT PRIMARY KEY,
    claim_id VARCHAR(20) NOT NULL REFERENCES claims(claim_id),
    activity_at TIMESTAMP NOT NULL,
    activity_type VARCHAR(50) NOT NULL,
    actor_type VARCHAR(30) NOT NULL
);

CREATE TABLE claim_payments (
    payment_id BIGINT PRIMARY KEY,
    claim_id VARCHAR(20) NOT NULL REFERENCES claims(claim_id),
    paid_at TIMESTAMP NOT NULL,
    amount DECIMAL(16, 2) NOT NULL CHECK (amount >= 0)
);

CREATE INDEX idx_claims_status ON claims (closed_at, reported_at);
CREATE INDEX idx_activities_claim_time ON claim_activities (claim_id, activity_at);

