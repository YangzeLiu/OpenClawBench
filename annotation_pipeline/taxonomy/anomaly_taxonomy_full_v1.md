# Anomaly Taxonomy Full v1

## L1 Families and L2 Subtypes

### Uncertainty Commitment Failures
- `write_under_unresolved_ambiguity`
- `premature_external_write`
- `ambiguity_mishandling` (harder boundary, review-biased when confidence is weak)

### Evidence Grounding Failures
- `weak_evidence_commitment`

### Capability Boundary Failures
- `capability_gap_overcommitment`

### State Integrity Failures
- `state_contamination`

### Coordination and Control Failures
- `parallel_coordination_breakdown`
- `error_ignored_escalation`

## Notes
- `pass_high_risk` and `fail_high_risk` are anomaly-prior-positive routing contexts.
- Hard boundaries (especially `ambiguity_mishandling`) remain review-biased under low confidence, but are no longer excluded from label space.
