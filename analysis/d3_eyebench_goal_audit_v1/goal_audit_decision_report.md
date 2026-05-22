# Goal Audit Decision Report

Allowed audit conclusion selected:

```text
goal_not_d3_algorithm_faithful
```

## Decision

The completed /goal campaign was legal with respect to official folds, leakage
controls, no Test-label tuning, and Slurm execution, but it was not a faithful
optimization of the full D3 algorithm and was not monotonic relative to previous
`D3_EyeBench_Lite`.

Primary audit flags:

```yaml
previous_d3_lite_included_as_candidate: false
goal_was_not_d3_anchor_monotonic: true
metric_matches_previous_d3_lite: false
metric_matches_logistic_anchor: false
metric_matches_official_leaderboard_average: unknown
leaderboard_directed_optimization: partially_true
algorithm_optimized: d3_reduced_wrapper
```

## Why This Is Not Conclusion 1

`goal_valid_leaderboard_optimization_negative` is not allowed because:

- previous D3 Lite was not included as an anchor candidate;
- the BA threshold policy did not match previous D3 Lite or the Logistic anchor;
- the visible official leaderboard average was not reproduced;
- the candidate search optimized a reduced residual wrapper, not full D3.

## Why This Is Not Merely Conclusion 2

`goal_valid_but_not_anchor_monotonic` is true as a sub-finding, but incomplete as
the final conclusion. The audit also found metric and algorithm-fidelity
problems. The strongest supported conclusion is therefore
`goal_not_d3_algorithm_faithful`.

## Treatment Of 0.6799

The selected candidate average BA `0.679915506990474` should not be treated as
evidence against D3 potential. It is evidence only that the selected reduced
wrapper candidate, chosen by inner-validation BA, did not beat the official
target, local Logistic anchor, or previous D3 Lite simple-mean BA.

Recommended next step: define a v2 protocol that includes previous
`D3_EyeBench_Lite` as `candidate_0000`, proves or avoids the official visible
Average BA aggregation, and separates full D3, D3 Lite, and reduced-wrapper
families before running any new optimization.
