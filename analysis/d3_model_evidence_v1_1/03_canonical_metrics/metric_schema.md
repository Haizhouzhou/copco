# Metric Schema v1.1

Required canonical metric columns:

- `evidence_id`
- `source_phase`
- `source_file`
- `source_row_identifier`
- `model_family`
- `model_name`
- `candidate_id`
- `algorithm_regime`
- `task`
- `evaluation_level`
- `split_regime`
- `data_scope`
- `prefix_type`
- `prefix_value`
- `evidence_budget`
- `feature_family`
- `calibrator`
- `threshold_policy`
- `threshold_source`
- `accumulator`
- `stopping_policy`
- `clean_or_oracle`
- `result_origin`
- `result_scope`
- `preferred_for_future_tables`
- `preferred_for_future_figures`
- `official_claim_allowed`
- `benchmark_relative_claim_allowed`
- `n_predictions`
- `n_readers`
- `n_trials`
- `n_prefix_rows`
- `coverage`
- `AUROC`
- `PR_AUC`
- `balanced_accuracy`
- `macro_F1`
- `Brier`
- `RMSE`
- `MAE`
- `R2`
- `calibration_intercept`
- `calibration_slope`
- `CI_low`
- `CI_high`
- `p_value`
- `metric_scale`
- `value_source_text`
- `source_trace_status`
- `notes`

Allowed result origins: blocked_or_skipped, clean_evaluation, external_reference, oracle_diagnostic, source_summary, validation_summary.
Allowed result scopes: blocked_result, deprecated_or_fast_run, diagnostic_completed_result, external_reference_baseline, primary_completed_result, secondary_completed_result, unresolved_conflict.
Allowed source trace status values: copied_from_v1, exact_file_trace, missing_source, report_text_trace, unresolved_conflict.

Preferred-for-future fields are nullable source-material fields and are not advisory fields in this vault.
