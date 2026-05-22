# D3OnlineTargetedOptimization v1 Execution Document

This phase deploys and evaluates D3 as both an offline reader-profile method and
an online sequential detector. It is not a framework-building phase: subgoals are
complete only after code runs, artifacts are produced, validation passes, and
reports include real row counts and metrics.

## Inputs and Outputs

Primary input:

- `results/label_release_v1_1_20260506_0041/prepared_dataset/`

Primary output directory:

- `results/d3_online_targeted_optimization_v1_<timestamp>/`

Small committed analysis outputs:

- `analysis/d3_online_targeted_optimization_v1/`

Large generated artifacts such as Parquet prefix data and full prediction CSVs
remain under `results/` and are not committed.

## Required Prefix Data

Each prefix row must include participant identity for grouping, operational
reader label for evaluation, prefix type/value/order, observed evidence counts,
cumulative gaze features, cumulative residual gaze features, DFM exposure,
DFM residual-gaze summaries, DFM sensitivity estimates when stable,
uncertainty/stability features, segmentation/boundary summaries, missingness
summaries, and `stable_enough_for_prediction`.

Feature construction must not use `reader_group` or `reader_group_binary`.
Prediction models must not use `participant_id`, `speech_id`, `text_id`, or
future total evidence counts as predictors.

## Split and Adaptation Contract

For each feasible split regime and outer fold, the runner writes:

- `train_prefix_predictions.csv`
- `inner_oof_prefix_predictions.csv`
- `calibration_prefix_predictions.csv`
- `outer_test_prefix_predictions.csv`
- `fold_manifest.json`

Every prediction row must declare `split_role`. Clean threshold, calibration,
accumulation, stopping, and candidate-selection logic may use only
`train_fit`, `inner_oof`, or `calibration` rows. `outer_test` labels are used
only for final evaluation and for explicitly marked oracle diagnostics.

## Online Model Families

Required model families:

- `raw_gaze_prefix`
- `residual_gaze_prefix`
- `dfm_exposure_prefix`
- `dfm_sensitivity_prefix`
- `dfm_residual_gaze_prefix`
- `dfm_residual_plus_uncertainty_prefix`
- `all_allowed_online`

Primary model: standardized logistic regression with training-only imputation
and scaling. Class weighting is selected from training or inner-validation data
only.

## Final Required Questions

The final decision report must answer:

1. Is D3 best understood as offline-only, online-capable, or both?
2. How much reading evidence is needed before online D3 becomes reliable?
3. Can legal threshold learning improve online balanced accuracy?
4. Can calibration improve Brier/calibration quality?
5. Can online evidence accumulation improve over single-trial D3_Lite?
6. Can stopping policies reduce reading burden while preserving performance?
7. Which online configuration is best under the project-specific target metric?
8. Does this change the paper story, benchmark-relative claim, or official SOTA
   status?

Expected claim boundary: offline D3 remains the main result. Online D3 may
become a targeted secondary or supplementary result. Official SOTA status remains
unchanged unless the official protocol is satisfied.
