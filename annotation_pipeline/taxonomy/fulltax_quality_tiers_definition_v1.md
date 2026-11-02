# FullTax Quality Tiers v1

## high_confidence_accepted
- Closed taxonomy valid
- No unresolved review cause
- No parse/truncation failure
- No major subtype/structure contradiction
- No unsupported onset left unresolved

## accepted_repaired
- Usable after deterministic normalization or revise_inline
- Internally coherent after repair

## review_or_unresolved
- parse/truncation failure
- unresolved subtype boundary
- missing critical structure
- unsafe-to-repair contradictions

Note: review tier is retained for audit and FN estimation, not dropped.
