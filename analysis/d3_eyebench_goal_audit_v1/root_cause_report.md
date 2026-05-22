# Root Cause Report

Question: why did the /goal best candidate end at average trial-level BA
`0.679915506990474`, below previous D3_EyeBench_Lite simple mean BA
`0.6985462748783746`?

## Supported Causes

### 1. Previous D3_Lite omitted from candidates

Status: supported.

Evidence:

- `analysis/d3_eyebench_protocol_aligned_optimization_v1/candidate_specs.json`
  contains 96 generated `d3opt_*` candidates and no `D3_EyeBench_Lite`,
  `d3_lite`, or `candidate_0000`.
- `analysis/d3_eyebench_protocol_aligned_optimization_v1/tables/candidate_summary.csv`
  contains 56 evaluated candidates and no previous D3 Lite row.
- `src/copco_eye_bench/d3_eyebench_protocol_optimization.py`
  `build_candidate_specs` only enumerates configured grid values.
- The optimizer initializes `best_score = -1.0`, not from prior D3 Lite.

Effect:

The final selected candidate could be lower than previous D3 Lite by design.
There was no monotonic anchor that guaranteed the previous score remained the
floor.

### 2. Metric mismatch

Status: supported.

Evidence:

- Previous D3 Lite uses `_classification_metrics` with `y_pred = y_score >= 0.5`.
- Local Logistic anchor uses `_baseline_extended_metrics` with `y_pred =
  y_score >= 0.5`.
- /goal candidates use `_select_threshold` on inner validation and then
  `_classification_metrics_at_threshold` on Test with frozen per-fold
  thresholds.
- /goal final average BA is a simple mean of three per-regime BA values.
- The visible official EyeBench Average BA values do not equal simple means of
  visible split values. AhnCNN visible average is `0.750`, while the simple mean
  of `0.777`, `0.775`, and `0.656` is `0.736`.

Effect:

The /goal balanced-accuracy values are not directly metric-identical to the
previous D3 Lite values or to the visible official leaderboard average
aggregation, even though they share official data/folds and fold-mean
per-regime reporting.

### 3. Feature regression / reduced wrapper

Status: supported.

Evidence:

- Previous D3 Lite reported `n_features = 12`.
- Selected d3opt_0024 reported `n_features = 9`.
- Previous D3 Lite used the fixed residual D3 Lite path in
  `evaluate_d3_eyebench_lite`.
- The /goal selected candidate used `duration_plus_count`, omitted skipping,
  used `log1p_duration`, and added syntax predictors for residualization.
- Full D3 feature families and full prepared CopCo joins were not attempted.

Effect:

The selected candidate is not the previous D3 Lite feature set and is not full
D3. Lower average BA cannot be read as a negative result for the previous D3
Lite anchor or full D3.

### 4. Candidate space too narrow for full D3

Status: supported.

Evidence:

- Config candidate grid is limited to residual aggregate features from official
  EyeBench IA/trial fields and sklearn classifiers.
- `no_full_prepared_copco_joins: true` blocks full prepared CopCo joins unless
  exact mapping and leakage policy are proven.
- No sequence model, full D3 feature family, or full DFM residual profile is
  represented in the candidate grid.

Effect:

The search optimized a narrow official-compatible residual-feature family, not
the whole D3 algorithmic space.

## Causes Not Supported By Evidence

| candidate cause | audit result |
| --- | --- |
| Protocol mismatch in official data/folds | not supported; artifacts show official EyeBench data/folds and split labels were used |
| Test-label tuning | not supported; decision and leakage reports mark selection as inner-validation only |
| Synthetic outputs | not supported; leakage report says false |
| Random predictions | not supported; leakage report says false |
| Leaderboard target entirely absent | not supported; config used visible AhnCNN `0.750` target |
| Evaluation bug causing row mismatch | not supported by current artifacts; row counts match previous D3 Lite and local Logistic anchor for all regimes |

## Root Cause Conclusion

The lower best score is explained by a combination of:

- previous D3 Lite omitted from the search;
- no anchor-monotonic initialization;
- threshold and average-aggregation differences;
- reduced-wrapper feature space relative to previous D3 Lite and full D3.

The `0.679915506990474` result should not be treated as evidence against D3
potential. It is evidence that this particular reduced wrapper, selected by
inner-validation BA, did not beat the official target, the local Logistic anchor,
or the previous D3 Lite simple-mean BA on the final Test average used by /goal.

## Recommended Next Step

Define a v2 audit-safe optimization protocol before any new run:

- include previous `D3_EyeBench_Lite` as `candidate_0000`;
- initialize the monotonic best floor from that anchor;
- reproduce or explicitly avoid claiming the official visible Average BA
  aggregation;
- report both official-style average if reproducible and simple mean;
- separate full D3, D3 Lite, and reduced-wrapper candidate families in the
  config and decision gates.
