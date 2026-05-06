# Final Scientific Decision Policy v1

This policy defines how AutoResearch v1 converts frozen analysis outputs into a
publication decision. The policy is evaluated before manuscript writing and is not a
mechanism for feature hunting.

## Decision Categories

- `main_paper_result`: central result suitable for the main claim.
- `secondary_result`: supports interpretation in the main text or a compact secondary
  result section.
- `appendix_result`: useful robustness, diagnostics, or supporting analysis.
- `defer`: scientifically plausible but not ready for the current paper.
- `drop`: not supported by the frozen evidence.

## Main-Result Gates

A result can be treated as `main_paper_result` only when all required gates pass:

- Leave-one-participant-out ROC-AUC lower bootstrap bound is greater than `0.70`.
- Permutation p-value is less than `0.01`, using the standard `+1` correction.
- The DFM sensitivity or residual-gaze model outperforms the DFM exposure-only model.
- The primary result does not depend on direct exposure-count variables.
- The final predictions contain exactly one prediction per participant.
- Leakage validation has no errors.
- Feature interpretation is stable enough to describe without overclaiming.

## Current Expected Classification

Expected main paper result:

`D3_dfm_residual_gaze_only` logistic regression with participant-level
leave-one-participant-out validation.

Expected secondary results:

- DFM surprisal interaction as explanatory support.
- Word-length interaction as secondary support.
- Previous-boundary opacity interaction as secondary interpretability support.

Expected appendix results:

- Calibration and reliability diagnostics.
- Participant influence and error analysis.
- Permutation and bootstrap robustness details.
- Text-exposure sensitivity audits.

Expected deferred results:

- Gemma comparison while access remains pending.
- Pronunciation-aware segmentation labels.
- Independent external validation.

Expected dropped or non-central results:

- Standalone segmentation-opacity main-effect framing.
- Random word-level split results.
- Parser-syntax claims while parser status remains `surface_heuristic_fallback`.
- Clinical diagnosis, screening, or medical utility claims.

## Selection Rule

When multiple candidates pass the gates, select the simplest model that preserves the
scientific interpretation and robustness. Do not select a more complex candidate for a
tiny metric gain. Any deviation from the locked Phase 4 story must be marked
`needs_review` unless the reason is methodological correction rather than score
optimization.
