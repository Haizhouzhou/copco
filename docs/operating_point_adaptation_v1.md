# OperatingPointAdaptation v1

OperatingPointAdaptation v1 analyzes D3-family prediction outputs as probability-first
reader-profile evidence. It does not run a new feature search, tune model features,
rerun EyeBench leaderboard methods, or use online W&B/API baseline reproduction.

The analysis separates five evidence types:

- fixed threshold 0.5 metrics;
- legal threshold policies that require train/inner-validation/calibration predictions;
- fitted calibration, again only from non-test calibration predictions;
- probability-first reader aggregation;
- test-oracle threshold diagnostics.

Test-oracle thresholds use outer test labels and are therefore diagnostic upper bounds
only. They must always be marked `official_claim_allowed=false` and must not support
an official EyeBench SOTA claim.

## Inputs

Configured inputs are existing prediction tables from:

- D3_EyeBench_Lite candidate_0000 from the own-method score-max phase;
- BenchmarkBridge D3 full-data outputs;
- OfficialEyeBenchAlignment D3 full-feature outputs;
- Phase 4 D3 reader-profile outputs;
- AutoResearch final D3 reader-profile outputs.

If a configured prediction file is unavailable, the runner writes a missing-prediction
blocker report and does not fabricate predictions. If train/inner-validation or
calibration prediction rows are unavailable, legal threshold learning and fitted
calibration are reported as not computed rather than inferred from test labels.

## Outputs

The runner writes the required reports under both the runtime output directory and
`analysis/operating_point_adaptation_v1/`. Runtime result directories remain ignored.
Small analysis reports may be committed.

Key reports:

- `probability_output_audit.md`
- `threshold_source_audit.md`
- `fixed_threshold_metrics.csv`
- `legal_threshold_metrics.csv`
- `test_oracle_threshold_metrics.csv`
- `test_oracle_information_budget.csv`
- `calibration_metrics.csv`
- `reader_probability_aggregation_metrics.csv`
- `threshold_curve_tables.csv`
- `before_after_operating_point_comparison.csv`
- `final_operating_point_decision_report.md`

## Validation

The validator checks that:

- BA and macro F1 rows record a threshold source;
- AUROC and PR-AUC are computed from `p_pred`;
- legal threshold rows do not use test-oracle sources;
- oracle rows are diagnostic and `official_claim_allowed=false`;
- reader aggregation uses probabilities or logits, with hard-label majority vote marked
  as a baseline only;
- oracle information budgets are present;
- the final decision report exists;
- the official SOTA claim is unchanged.
