# Metric Consistency Audit

This audit compares metric computation for previous `D3_EyeBench_Lite`, the
/goal selected candidate, the local official-derived Logistic anchor, and the
official EyeBench leaderboard reference.

## Summary Flags

```yaml
metric_matches_previous_d3_lite: false
metric_matches_logistic_anchor: false
metric_matches_official_leaderboard_average: unknown
```

The /goal selected candidate shares official data, official folds, per-regime
fold-mean reporting, and probability-based AUROC with previous D3 Lite and the
local Logistic anchor. It does not fully match their balanced-accuracy metric
procedure because the /goal uses train/inner-validation-selected thresholds,
whereas D3 Lite and the Logistic anchor use the hard-label threshold implied by
`y_score >= 0.5`.

The /goal final average BA is a simple mean of the three per-regime BA values.
That average is not proven to be the official EyeBench leaderboard average,
because the visible EyeBench `Average_\makecell{Balanced\\Accuracy}` values do
not equal the simple mean of the three visible regime values.

## Side-By-Side Metric Definitions

| source | data source | folds | fold aggregation | trial weighting | class weighting | threshold selection | per-regime metrics | average BA | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Previous D3_EyeBench_Lite | official EyeBench processed CopCo data | official `CopCo_TYP` folds | mean of fold metrics per regime | each fold metric computed over trial rows; regime value is unweighted fold mean | logistic classifier has `class_weight=balanced` | fixed `0.5` via `y_score >= 0.5` | yes | simple mean in comparison table, not official leaderboard average | probabilities |
| /goal d3opt_0024 | official EyeBench processed CopCo data | official `CopCo_TYP` folds | mean of fold metrics per regime | each fold metric computed over trial rows; regime value is unweighted fold mean | candidate classifiers may use class weighting; selected logistic uses `class_weight=balanced` | selected on train/inner-validation only, frozen per split/fold | yes | simple mean of three per-regime Test BA values | probabilities |
| Local Logistic anchor | official EyeBench processed CopCo data | official `CopCo_TYP` folds | mean of fold metrics per regime | each fold metric computed over trial rows; regime value is unweighted fold mean | official-derived logistic uses balanced class weighting | fixed `0.5` hard prediction threshold | yes | local simple mean `0.719546` | probabilities |
| Official EyeBench leaderboard | EyeBench published formatted Test table | official `CopCo_TYP` Test folds | published table values | not fully reproduced by this audit | model-specific | model-specific official evaluation | yes | visible table average, not reproduced by /goal code | table probabilities/AUROC values |

## Code Paths

Previous D3 Lite:

- `src/copco_eye_bench/official_eyebench_sota_check.py`
  `evaluate_d3_eyebench_lite`
- `src/copco_eye_bench/official_eyebench_sota_check.py`
  `_evaluate_feature_matrix`
- `src/copco_eye_bench/research_exploration.py`
  `_classification_metrics`, which sets `y_pred = (y_score >= 0.5)`

/goal candidate:

- `src/copco_eye_bench/d3_eyebench_protocol_optimization.py`
  `_evaluate_candidate_inner`
- `src/copco_eye_bench/d3_eyebench_protocol_optimization.py`
  `_select_threshold`
- `src/copco_eye_bench/d3_eyebench_protocol_optimization.py`
  `_classification_metrics_at_threshold`
- `src/copco_eye_bench/d3_eyebench_protocol_optimization.py`
  `_metric_frame`
- `src/copco_eye_bench/d3_eyebench_protocol_optimization.py`
  `_average_primary_ba`

Local Logistic anchor:

- `src/copco_eye_bench/official_eyebench_baseline_evaluator_closure.py`
  `_baseline_extended_metrics`, which sets `y_pred = (y_score >= 0.5)`

Official reference parsing:

- `src/copco_eye_bench/official_eyebench_sota_check.py`
  `_official_reference_table` parses the formatted table but recomputes
  `average_balanced_accuracy` as a simple mean of the three split columns.

## Average Mismatch Evidence

The visible official formatted table contains:

| model | visible split BA values | simple mean | visible Average BA |
| --- | --- | ---: | ---: |
| AhnCNN | 0.777, 0.775, 0.656 | 0.736 | 0.750 |
| Logistic Regression | 0.755, 0.766, 0.635 | 0.718667 | 0.738 |
| Random Forest | 0.698, 0.815, 0.597 | 0.703333 | 0.727 |

The /goal config correctly stored the visible AhnCNN target as `0.750`, but the
internal `published_reference_rows` written to
`analysis/d3_eyebench_protocol_aligned_optimization_v1/official_sota_decision.json`
contain recomputed simple means such as `0.736` for AhnCNN and `0.718667` for
Logistic Regression.

Therefore:

- The final /goal average BA `0.679915506990474` is a simple mean of the three
  selected-candidate per-regime BAs.
- The audit cannot certify that this is the same aggregation as the official
  EyeBench leaderboard average.
- The mismatch affects leaderboard-directed interpretation, although the final
  candidate is below both `0.736` and the visible `0.750` AhnCNN target.
