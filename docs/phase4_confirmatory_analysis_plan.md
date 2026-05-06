# Phase 4 Confirmatory Sensitivity Analysis Plan

## Scope

Phase 4 confirms, stress-tests, and explains the strongest Phase 3 participant-level
signal. It uses Label Release v1.1, the prepared dataset, and Phase 3 sensitivity
profiles. It does not add core labels, rebuild prior releases, or run broad exploratory
feature expansion.

## Primary Questions

1. Does participant-level prediction survive cross-fitted residualization inside LOPO?
2. Is DFM sensitivity stronger than DFM text exposure?
3. Does prediction survive removal of exposure-count and exposure-only variables?
4. Does prediction survive removal of raw speed/global-duration features?
5. Are reader-group interactions with word length, DFM surprisal, and previous-boundary
   opacity stable after controls?
6. Should segmentation be framed as a main result, secondary interaction result,
   appendix result, deferred, or dropped?

## Required Inputs

- `results/label_release_v1_1_20260506_0041/prepared_dataset/analysis_ready_word_level_v1_1.parquet`
- `results/label_release_v1_1_20260506_0041/prepared_dataset/analysis_ready_participant_level_v1_1.parquet`
- `results/label_release_v1_1_20260506_0041/labels/split_labels_v1.parquet`
- `results/research_exploration_v1_20260506_0149/analysis/research_exploration/participant_sensitivity_profiles.parquet`

## Confirmatory Feature Families

- `D1_dfm_exposure_only`: text-level DFM exposure variables only.
- `D2_dfm_sensitivity_only`: participant-level DFM sensitivity slopes and cross-fitted
  residual DFM sensitivity features only.
- `D3_dfm_residual_gaze_only`: cross-fitted DFM residual gaze-cost slopes only.
- `D4_dfm_exposure_plus_sensitivity`: combined Phase 3 DFM family for comparison.

Exposure-count variables such as `n_words_read`, `n_speeches`, `n_word_rows`,
`total_word_rows`, and `word_observation_count` are excluded from every prediction
feature group.

## Modeling

Participant-level prediction uses LOPO as primary validation and participant-grouped
k-fold as secondary validation. Models are limited to standardized logistic regression
and linear SVM. The main robustness tests are participant-label permutation, bootstrap
confidence intervals, leave-one-dyslexia-labeled-participant sensitivity, remove-one
participant influence, removal of raw speed/global-duration variables, removal of
exposure-only variables, and DFM sensitivity-only prediction.

## Interaction Analysis

Interaction analysis is restricted to the Phase 3 candidates:

- reader group by word length
- reader group by DFM surprisal
- reader group by previous-boundary opacity

Outcomes are skipping, first fixation duration, first-pass duration, go-past time,
total fixation duration, and fixation count. Controls include lexical, DFM,
sentence-position, speech fixed-effect, segmentation, and quality fields where
available. If mixed-effects models are infeasible, cluster-robust alternatives are
documented.

## Decision Categories

Final reporting uses:

- `main_paper_result`
- `secondary_result`
- `appendix_result`
- `defer`
- `drop`

The expected publication framing is participant-level DFM predictability sensitivity
and residualized gaze-cost profiles as the main result. Boundary opacity is retained
only as a secondary interaction and interpretability feature unless stronger evidence
emerges.
