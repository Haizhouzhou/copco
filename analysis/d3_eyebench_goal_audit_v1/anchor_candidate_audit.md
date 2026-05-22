# Anchor Candidate Audit

Question: was previous `D3_EyeBench_Lite` included as `candidate_0000` or as any
other candidate in the /goal campaign?

## Findings

| question | answer |
| --- | --- |
| Was previous `D3_EyeBench_Lite` included in the candidate set? | No |
| If yes, what candidate ID? | Not applicable |
| What were its recomputed per-regime BA/AUROC inside the /goal candidate table? | Not applicable; no D3 Lite candidate row exists |
| Did those match previous values `0.7274 / 0.7341 / 0.6342`? | Not testable inside the /goal search because D3 Lite was omitted |
| If no, why was it omitted? | The /goal `candidate_grid` only enumerated new residual-feature variants from `residual_alphas`, `predictor_sets`, `outcome_sets`, `aggregation_sets`, `transforms`, `classifiers`, and seeds. There is no `D3_EyeBench_Lite`, `candidate_0000`, `anchor_candidate`, or previous-output candidate in the config or candidate builder. |
| Did `best_so_far` initialize from previous D3 Lite or from the first new candidate? | It initialized from `best_score = -1.0` and the first completed generated candidate. |
| Could the final best be lower than previous D3 Lite by design? | Yes. Since previous D3 Lite was not in the candidate set and `best_score` was not initialized from it, the selected best among generated candidates can be lower than previous D3 Lite. |

## Evidence

Campaign config:

- `configs/d3_eyebench_protocol_aligned_optimization_v1.yaml` defines only a
  generated grid under `d3_eyebench_protocol_optimization.candidate_grid`.
- The only anchor in the config is
  `d3_eyebench_protocol_optimization.local_logistic_anchor`, which points to the
  local official-derived LogisticRegressionMLArgs metrics.
- The config contains no previous D3 Lite prediction path, no previous D3 Lite
  metrics path, and no anchor candidate option.

Candidate generation code:

- `src/copco_eye_bench/d3_eyebench_protocol_optimization.py`
  `build_candidate_specs` creates candidates only from the configured Cartesian
  grid and names them `d3opt_0001_...`, `d3opt_0002_...`, etc.
- The same file initializes candidate selection with `best_score = -1.0`,
  `best_candidate = None`, and then compares each candidate's
  `selection_score`.

Candidate artifacts:

- `analysis/d3_eyebench_protocol_aligned_optimization_v1/candidate_specs.json`
  contains 96 generated specs and no string matching `D3_EyeBench_Lite`,
  `d3_lite`, `lite`, or `candidate_0000`.
- `analysis/d3_eyebench_protocol_aligned_optimization_v1/tables/candidate_summary.csv`
  contains 56 evaluated generated candidates. It starts at
  `d3opt_0001_a08ea0296c` and does not include a prior D3 Lite row.

Previous D3 Lite anchor values from closure/runtime-fix:

| split | previous D3 Lite BA | previous D3 Lite AUROC |
| --- | ---: | ---: |
| unseen_reader | 0.727404 | 0.808507 |
| unseen_text | 0.734075 | 0.831851 |
| unseen_reader_and_text | 0.634159 | 0.715385 |
| simple mean | 0.698546 | 0.785248 |

Selected /goal candidate:

| split | d3opt_0024 BA | d3opt_0024 AUROC |
| --- | ---: | ---: |
| unseen_reader | 0.663349 | 0.821326 |
| unseen_text | 0.759744 | 0.842543 |
| unseen_reader_and_text | 0.616654 | 0.692886 |
| simple mean used by /goal | 0.679916 | 0.785585 |

## Audit Flag

`goal_was_not_d3_anchor_monotonic = true`

The /goal was not monotonic with respect to the previous D3 Lite result. The
lower final average BA is therefore not surprising under the implemented
search procedure and is not evidence that all prior D3-compatible variants were
outperformed.
