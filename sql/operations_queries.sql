-- Open backlog and age by line of business.
SELECT
    p.line_of_business,
    COUNT(*) AS open_claims,
    AVG(CAST(:as_of_date AS DATE) - CAST(c.reported_at AS DATE)) AS average_open_days,
    SUM(CASE
        WHEN CAST(:as_of_date AS DATE) - CAST(c.reported_at AS DATE) > c.target_days THEN 1 ELSE 0
    END) AS claims_past_target
FROM claims c
JOIN policies p ON p.policy_id = c.policy_id
WHERE c.closed_at IS NULL
GROUP BY p.line_of_business
ORDER BY claims_past_target DESC;

-- Claims with no recorded activity in the last seven days.
SELECT
    c.claim_id,
    p.line_of_business,
    c.severity_estimate,
    MAX(a.activity_at) AS last_activity_at
FROM claims c
JOIN policies p ON p.policy_id = c.policy_id
LEFT JOIN claim_activities a ON a.claim_id = c.claim_id
WHERE c.closed_at IS NULL
GROUP BY c.claim_id, p.line_of_business, c.severity_estimate
HAVING MAX(a.activity_at) IS NULL
    OR MAX(a.activity_at) < CAST(:as_of_date AS TIMESTAMP) - INTERVAL '7 days'
ORDER BY c.severity_estimate DESC;

-- Closed-claim cycle time and paid severity by operating segment.
SELECT
    p.region,
    c.channel,
    COUNT(*) AS closed_claims,
    AVG(CAST(c.closed_at AS DATE) - CAST(c.reported_at AS DATE)) AS average_cycle_days,
    AVG(payment.total_paid) AS average_paid
FROM claims c
JOIN policies p ON p.policy_id = c.policy_id
LEFT JOIN (
    SELECT claim_id, SUM(amount) AS total_paid
    FROM claim_payments
    GROUP BY claim_id
) payment ON payment.claim_id = c.claim_id
WHERE c.closed_at IS NOT NULL
GROUP BY p.region, c.channel
ORDER BY average_cycle_days DESC;
