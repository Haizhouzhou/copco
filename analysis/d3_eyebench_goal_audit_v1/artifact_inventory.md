# D3 EyeBench Goal Audit v1 Artifact Inventory

Audit branch: `codex/d3-eyebench-goal-audit-v1`

Base campaign commit audited: `270f86d`

This audit is read-only with respect to model results. It did not run a new
candidate search, tune models, change scores, or update manuscript claims.

## Required Protocol And Campaign Artifacts

| item | path | status |
| --- | --- | --- |
| Goal protocol | `docs/goals/d3_eyebench_protocol_aligned_optimization_v1.md` | present, tracked |
| Campaign config | `configs/d3_eyebench_protocol_aligned_optimization_v1.yaml` | present, tracked |
| Accelerated campaign config | `configs/d3_eyebench_protocol_aligned_optimization_v1_accelerated.yaml` | present, tracked |
| Candidate manifest, committed copy | `analysis/d3_eyebench_protocol_aligned_optimization_v1/candidate_specs.json` | present, tracked |
| Candidate manifest, run output | `results/d3_eyebench_protocol_aligned_optimization_v1_20260522_074957/optimization/candidate_specs.json` | present, ignored result |
| Candidate leaderboard table, committed copy | `analysis/d3_eyebench_protocol_aligned_optimization_v1/tables/candidate_summary.csv` | present, tracked |
| Candidate leaderboard table, run output | `results/d3_eyebench_protocol_aligned_optimization_v1_20260522_074957/optimization/candidate_summary.csv` | present, ignored result |
| Inner-validation fold metrics, committed copy | `analysis/d3_eyebench_protocol_aligned_optimization_v1/tables/inner_validation_fold_metrics.csv` | present, tracked |
| Final candidate diagnostics | `results/d3_eyebench_protocol_aligned_optimization_v1_20260522_074957/optimization/final_candidate_diagnostics.csv` | present, ignored result |
| Final trial metrics, committed copy | `analysis/d3_eyebench_protocol_aligned_optimization_v1/tables/d3_optimized_trial_metrics.csv` | present, tracked |
| Final trial predictions | `results/d3_eyebench_protocol_aligned_optimization_v1_20260522_074957/typ/d3_optimized_trial_predictions.csv` | present, ignored result |
| Decision JSON, committed copy | `analysis/d3_eyebench_protocol_aligned_optimization_v1/official_sota_decision.json` | present, tracked |
| Decision report, committed copy | `analysis/d3_eyebench_protocol_aligned_optimization_v1/decision_report.md` | present, tracked |
| Leakage report, committed copy | `analysis/d3_eyebench_protocol_aligned_optimization_v1/leakage_report.md` and `analysis/d3_eyebench_protocol_aligned_optimization_v1/leakage_report.json` | present, tracked |
| Manifest, committed copy | `analysis/d3_eyebench_protocol_aligned_optimization_v1/manifest.json` | present, tracked |

## Selected Candidate

Selected candidate: `d3opt_0024_2d9a9f9c46`

The selected candidate has no standalone config file. Its config is embedded in:

- `analysis/d3_eyebench_protocol_aligned_optimization_v1/candidate_specs.json`
- `analysis/d3_eyebench_protocol_aligned_optimization_v1/official_sota_decision.json`
- `results/d3_eyebench_protocol_aligned_optimization_v1_20260522_074957/optimization/final_candidate_diagnostics.csv`

Selected config:

```json
{
  "aggregation_set": "central_spread",
  "candidate_id": "d3opt_0024_2d9a9f9c46",
  "classifier": "logistic_regression",
  "classifier_params": {
    "C": 0.1,
    "class_weight": "balanced",
    "max_iter": 2000,
    "penalty": "l2",
    "solver": "liblinear"
  },
  "outcome_set": "duration_plus_count",
  "predictor_set": "surface_surprisal_syntax",
  "residual_alpha": 1.0,
  "seed": 20260522,
  "transform": "log1p_duration"
}
```

## Previous D3_EyeBench_Lite Artifacts

| item | path | status |
| --- | --- | --- |
| Runtime-fix D3 Lite report | `analysis/official_eyebench_runtime_fix_v1/d3_eyebench_lite_official_evaluation_report.md` | present, tracked |
| Runtime-fix D3 Lite comparison table | `analysis/official_eyebench_runtime_fix_v1/tables/copco_typ_official_sota_comparison.csv` | present, tracked |
| Runtime-fix D3 Lite feature report | `analysis/official_eyebench_runtime_fix_v1/d3_eyebench_lite_feature_report.md` | present, tracked |
| Runtime-fix D3 Lite leakage report | `analysis/official_eyebench_runtime_fix_v1/d3_eyebench_lite_leakage_report.md` | present, tracked |
| Runtime-fix D3 Lite trial metrics | `results/official_eyebench_runtime_fix_v1_20260522_0005/typ/d3_lite_trial_metrics.csv` | present, ignored result |
| Runtime-fix D3 Lite combined metrics | `results/official_eyebench_runtime_fix_v1_20260522_0005/typ/d3_eyebench_lite_metrics.csv` | present, ignored result |
| Runtime-fix D3 Lite trial predictions | `results/official_eyebench_runtime_fix_v1_20260522_0005/typ/d3_lite_trial_predictions.csv` | present, ignored result |
| Runtime-fix D3 Lite alternative prediction copy | `results/official_eyebench_runtime_fix_v1_20260522_0005/typ/d3_eyebench_lite_predictions.csv` | present, ignored result |
| Runtime-fix D3 Lite official-format predictions | `results/official_eyebench_runtime_fix_v1_20260522_0005/typ/trial_level_test_results.csv` | present, ignored result |
| Closure D3 Lite reuse report | `analysis/official_eyebench_baseline_evaluator_closure_v1/d3_reuse_validation_report.md` | present, tracked |
| Closure D3 Lite trial metrics | `results/official_eyebench_baseline_evaluator_closure_v1_20260522_015211/typ/d3_lite_trial_metrics.csv` | present, ignored result |
| Closure D3 Lite reader metrics | `results/official_eyebench_baseline_evaluator_closure_v1_20260522_015211/typ/d3_lite_reader_aggregated_metrics.csv` | present, ignored result |
| Closure D3 Lite predictions | `results/official_eyebench_baseline_evaluator_closure_v1_20260522_015211/typ/*predictions*` | not present in closure output; closure reused runtime-fix metrics |

Canonical previous D3 Lite trial-level BA/AUROC values are recorded in
`analysis/official_eyebench_baseline_evaluator_closure_v1/d3_reuse_validation_report.md`
and match the runtime-fix result:

| split | BA | AUROC |
| --- | ---: | ---: |
| unseen_reader | 0.727404 | 0.808507 |
| unseen_text | 0.734075 | 0.831851 |
| unseen_reader_and_text | 0.634159 | 0.715385 |
| simple mean | 0.698546 | 0.785248 |

## Local Official-Derived Logistic Anchor

| item | path | status |
| --- | --- | --- |
| Logistic anchor report | `analysis/official_eyebench_baseline_evaluator_closure_v1/local_official_logistic_baseline_report.md` | present, tracked |
| Logistic anchor report, run copy | `results/official_eyebench_baseline_evaluator_closure_v1_20260522_015211/analysis/official_eyebench_baseline_evaluator_closure_v1/local_official_logistic_baseline_report.md` | present, ignored result |
| Logistic anchor extended metrics | `results/official_eyebench_baseline_evaluator_closure_v1_20260522_015211/baseline/logistic/local_official_derived_extended_metrics.csv` | present, ignored result |
| Logistic anchor fold metrics | `results/official_eyebench_baseline_evaluator_closure_v1_20260522_015211/baseline/logistic/local_official_derived_fold_metrics.csv` | present, ignored result |
| Logistic anchor predictions | `results/official_eyebench_baseline_evaluator_closure_v1_20260522_015211/baseline/logistic/local_official_derived_predictions.csv` | present, ignored result |
| Logistic anchor official-format predictions | `results/official_eyebench_baseline_evaluator_closure_v1_20260522_015211/baseline/logistic/trial_level_test_results.csv` | present, ignored result |

The /goal config points to the extended metrics file as the local pipeline
anchor and sets `rerun_allowed: false`.

## Official Target Definition

| item | path | status |
| --- | --- | --- |
| Target in /goal config | `configs/d3_eyebench_protocol_aligned_optimization_v1.yaml` | `target_model: AhnCNN`, `target_balanced_accuracy: 0.750` |
| Official formatted Test table | `eyebench/results/formatted_eyebench_benchmark_results/CopCo_TYP_test.csv` | present, ignored submodule result |
| Target snapshot in decision JSON | `analysis/d3_eyebench_protocol_aligned_optimization_v1/official_sota_decision.json` | present, tracked |

The visible EyeBench table row for `AhnCNN` reports:

- Unseen Reader BA: `77.7 +/- 1.8`
- Unseen Text BA: `77.5 +/- 2.7`
- Unseen Text and Reader BA: `65.6 +/- 2.4`
- Average BA: `75.0 +/- 0.8`

Important audit note: the visible average `75.0` is not the simple mean of
`77.7`, `77.5`, and `65.6`, which is `73.6`. The /goal config used `0.750` as
the target, but helper rows in `official_sota_decision.json` recomputed simple
means for `published_reference_rows`.

## Validator Outputs

| item | path or evidence | status |
| --- | --- | --- |
| Base config validation | `results/d3_eyebench_protocol_aligned_optimization_v1_20260522_074957/config_validation.json` | passed |
| Accelerated config validation | `results/d3_eyebench_protocol_aligned_optimization_v1_accelerated_20260522_080058/config_validation.json` | passed |
| Base protocol validator stdout | `results/d3_eyebench_protocol_aligned_optimization_v1_sbatch/slurm/d3_eyebench_opt_v1_3368826.out` | reports `status: passed` |
| Accelerated protocol validator stdout | `results/d3_eyebench_protocol_aligned_optimization_v1_sbatch/slurm/d3_eyebench_opt_v1_accel_3368833.out` | reports `status: passed` |
| Base leakage report | `analysis/d3_eyebench_protocol_aligned_optimization_v1/leakage_report.json` | passed |
| Accelerated leakage report | `analysis/d3_eyebench_protocol_aligned_optimization_v1_accelerated/leakage_report.json` | passed |

## Slurm Logs And Manifests

| job | path | status |
| --- | --- | --- |
| 3368826 stdout | `results/d3_eyebench_protocol_aligned_optimization_v1_sbatch/slurm/d3_eyebench_opt_v1_3368826.out` | present |
| 3368826 stderr | `results/d3_eyebench_protocol_aligned_optimization_v1_sbatch/slurm/d3_eyebench_opt_v1_3368826.err` | present |
| 3368826 manifest | `results/d3_eyebench_protocol_aligned_optimization_v1_20260522_074957/slurm/d3_eyebench_opt_v1_3368826_manifest.json` | present |
| 3368833 stdout | `results/d3_eyebench_protocol_aligned_optimization_v1_sbatch/slurm/d3_eyebench_opt_v1_accel_3368833.out` | present |
| 3368833 stderr | `results/d3_eyebench_protocol_aligned_optimization_v1_sbatch/slurm/d3_eyebench_opt_v1_accel_3368833.err` | present |
| 3368833 manifest | `results/d3_eyebench_protocol_aligned_optimization_v1_accelerated_20260522_080058/slurm/d3_eyebench_opt_v1_accel_3368833_manifest.json` | present |

External `sacct` after job completion reported both successful jobs as
`COMPLETED` with exit code `0:0`.
