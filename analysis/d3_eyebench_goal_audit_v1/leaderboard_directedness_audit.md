# Leaderboard-Directedness Audit

## Classification

```yaml
leaderboard_directed_optimization: partially_true
```

The /goal campaign was leaderboard-aware and used the visible AhnCNN Test
Average BA `0.750` as the final SOTA target. However, candidate selection was
based on train/inner-validation BA, and the final average BA used by the /goal
was a simple mean of three per-regime Test BAs, not a reproduced official
EyeBench leaderboard average. The campaign was therefore partially
leaderboard-directed, but not a direct optimization of the official visible
leaderboard Average BA.

## Direct Answers

| question | answer |
| --- | --- |
| What was the primary objective? | Trial-level balanced accuracy under official `CopCo_TYP` Test protocol, with final comparison against a fixed published leaderboard target. |
| Was it official Test Average Balanced Accuracy? | The final gate used `target_balanced_accuracy: 0.750` from the visible AhnCNN Test Average BA, but the campaign's own final average was computed as a simple mean of three per-regime BAs. |
| Was it simple mean BA? | Yes for the /goal final D3 average: `_average_primary_ba` computes the mean of the three per-regime BA rows. |
| Was it validation BA? | Candidate selection used train/inner-validation fold mean BA. Final reporting used Test metrics for the selected candidate. |
| Was candidate selection based on test folds? | No. The decision records `selection_source: inner_validation_only`, and the code selects candidates before final Test evaluation. |
| Was candidate selection based on inner validation? | Yes. |
| Was the official AhnCNN target 75.0 used correctly? | Partially. The config and final decision gate used `0.750`, matching the visible table Average BA. But helper `published_reference_rows` recomputed AhnCNN average as `0.736`, which does not match the visible official average. |
| Was Logistic 73.8 official average distinguished from local simple mean 71.95? | Not cleanly. The local Logistic anchor average `0.719546` is distinct in config and decision checks, but the parsed official reference rows recomputed Logistic average as `0.718667` instead of the visible `0.738`. |
| Were official Test and Validation tables separated? | The campaign references `CopCo_TYP_test.csv`; no validation leaderboard table is used as the official target. |
| Did the goal optimize for balanced accuracy or AUROC? | Candidate selection and final decision optimized balanced accuracy. AUROC was secondary. |
| Were per-regime leaderboard targets tracked? | Yes, config records split targets: unseen_reader AhnCNN `0.777`, unseen_text Random Forest `0.815`, both-unseen AhnCNN `0.656`. Final decision primarily used the single average target `0.750`. |

## Evidence

Config target:

- `configs/d3_eyebench_protocol_aligned_optimization_v1.yaml`
  `published_leaderboard_snapshot.target_model: AhnCNN`
- `target_balanced_accuracy: 0.750`
- `target_metric: average_trial_level_balanced_accuracy`
- `local_formatted_table: eyebench/results/formatted_eyebench_benchmark_results/CopCo_TYP_test.csv`

Decision output:

- `analysis/d3_eyebench_protocol_aligned_optimization_v1/official_sota_decision.json`
  stores `selection_source: inner_validation_only`.
- The decision checks compare final average BA `0.679915506990474` to target BA
  `0.750`.

Code:

- Candidate ranking uses `selection_metric:
  inner_validation_fold_mean_balanced_accuracy`.
- `_average_primary_ba` computes a simple mean over the three per-regime Test BA
  values.
- `_official_reference_table` recomputes official reference averages as simple
  means, even though the visible EyeBench table averages differ.

## Interpretation

The campaign was legal with respect to no Test-label tuning, but it was not a
strict reproduction of the official leaderboard Average BA objective. It was an
inner-validation-directed search with a final post-selection leaderboard gate.
