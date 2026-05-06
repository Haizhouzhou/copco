# AutoResearch v1 Analysis Plan

AutoResearch v1 is the end-to-end publication decision loop for the frozen CopCo
dyslexia-labeled reader analysis package. It starts from the completed Feature Release
v1, Label Release v1.1, Phase 3 exploration, and Phase 4 confirmatory outputs. It does
not create new core labels, add new feature families, or run uncontrolled model search.

## Frozen Inputs

- Feature Release v1: `results/feature_release_v1_20260505_2155`
- Label Release v1.1: `results/label_release_v1_1_20260506_0041`
- Phase 3 controlled exploration: `results/research_exploration_v1_20260506_0149`
- Phase 4 confirmatory sensitivity analysis:
  `results/phase4_confirmatory_sensitivity_v1_20260506_0715`

The input validation gate checks row counts, label counts, participant-grouped split
labels, duplicate participant-word keys, DFM feature availability, residualized Phase 4
outputs, parser and segmentation status, and exposure-count variable exclusion.

## Locked Primary Analysis

The locked primary result is the Phase 4 model:

- Feature group: `D3_dfm_residual_gaze_only`
- Model: standardized logistic regression
- Split: leave-one-participant-out
- Unit of prediction: participant

The AutoResearch run verifies the frozen Phase 4 metrics and predictions rather than
using the package to choose a new score-maximizing model.

## Stress Tests

AutoResearch v1 records the predefined stress-test suite:

- DFM exposure-only versus DFM sensitivity and residual-gaze feature groups.
- Removal of exposure-count variables and raw-speed/global-duration variables.
- Text-exposure sensitivity audits against speech count, word rows, and exposure
  summaries.
- Participant influence, leave-one-dyslexia-labeled-participant sensitivity, and
  misclassification audits.
- Calibration, permutation, bootstrap, and feature-stability summaries.
- Focused synthesis of the Phase 4 interaction results.

## Constrained Refinement

The refinement loop is bounded to the Phase 4 short list:

- `D3_dfm_residual_gaze_only`
- `D2_dfm_sensitivity_only`
- `D3` with raw-speed/global-duration variables removed
- `D3` with stable features only when already supported by Phase 4 stability output
- `D3` with calibration only when calibration improves Brier without materially
  degrading ROC-AUC
- Optional inner-fold regularization only when already supported by existing code

The selection rule prefers the simplest passing candidate. A more complex model is not
selected for a negligible metric gain.

## Paper Package

The run creates manuscript-ready outputs under the timestamped results directory and
mirrors small drafting artifacts under `analysis/autoresearch_v1/`:

- Tables in CSV and Markdown.
- Figures or explicit skip-reason Markdown files.
- Manuscript skeleton.
- Reviewer-risk report.
- Reproducibility and deployment scripts.
- Final publication decision report.

## Scientific Framing

Main story:

Participant-level DFM predictability sensitivity and cross-fitted residualized gaze-cost
profiles distinguish dyslexia-labeled and typical/control readers in Danish natural
reading.

Secondary story:

Reader-group interactions involving DFM surprisal, word length, and previous-boundary
opacity help interpret the participant-level result.

Not selected as main claims:

- Standalone segmentation-opacity main effects.
- Word-level classification.
- Parser-syntax findings, because the parser status is `surface_heuristic_fallback`.
- Clinical screening or diagnosis.
