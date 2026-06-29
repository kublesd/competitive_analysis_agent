# Evaluation Results

- Mode: `offline_fixture`
- Generated at: `2026-06-24T02:27:13.364754+00:00`
- Cases: 3
- Case pass rate: 100.0%
- Task success rate: 66.7%
- Average field coverage: 100.0%
- Citation validity: 100.0%
- Source coverage: 100.0%
- Average duration: 0.0072 seconds
- Estimated cost: not available

| Case | Expected behavior | Task success | Field coverage | Citation validity | Source coverage | Retry | Duration (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| complete_success | pass | pass | 100.0% | 100.0% | 100.0% | 0 | 0.0092 |
| retry_recovery | pass | pass | 100.0% | 100.0% | 100.0% | 1 | 0.0062 |
| verification_warning | pass | fail | 100.0% | 100.0% | 100.0% | 1 | 0.0062 |

## Metric Boundaries

- Citation validity only checks whether referenced IDs exist; it does not prove the claim is semantically true.
- Source coverage checks planned product/topic pairs with Evidence; it does not measure source authority or freshness.
- Field coverage measures populated report sections; more fields do not automatically mean better analysis.
- Cost remains unavailable until model usage metadata is captured consistently.
