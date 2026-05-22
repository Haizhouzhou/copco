# Per-Regime Candidate Audit

Primary comparison level: official trial-level fold mean on official
`CopCo_TYP` Test folds, except where explicitly marked as inner-validation only.

## Test-Set Per-Regime Summary

| model | unseen_reader BA | unseen_reader AUROC | unseen_text BA | unseen_text AUROC | unseen_reader_and_text BA | unseen_reader_and_text AUROC | average BA used by /goal | simple mean BA | official-style average BA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Previous D3_EyeBench_Lite | 0.727404 | 0.808507 | 0.734075 | 0.831851 | 0.634159 | 0.715385 | not used by /goal | 0.698546 | not available |
| d3opt_0024_2d9a9f9c46 | 0.663349 | 0.821326 | 0.759744 | 0.842543 | 0.616654 | 0.692886 | 0.679916 | 0.679916 | not available |
| Local Logistic anchor | 0.754150 | 0.830438 | 0.766522 | 0.831545 | 0.637966 | 0.690967 | 0.719546 | 0.719546 | not available locally; official visible Logistic average is 0.738 |

## Fold-Level Test Metrics

Previous D3_EyeBench_Lite, from
`results/official_eyebench_runtime_fix_v1_20260522_0005/typ/d3_lite_trial_predictions.csv`:

| split | fold | rows | threshold | BA | AUROC |
| --- | ---: | ---: | --- | ---: | ---: |
| unseen_reader | 0 | 884 | 0.5 | 0.732087 | 0.835667 |
| unseen_reader | 1 | 906 | 0.5 | 0.623010 | 0.702366 |
| unseen_reader | 2 | 799 | 0.5 | 0.813043 | 0.889689 |
| unseen_reader | 3 | 965 | 0.5 | 0.741477 | 0.806306 |
| unseen_reader_and_text | 0 | 321 | 0.5 | 0.700309 | 0.758693 |
| unseen_reader_and_text | 1 | 269 | 0.5 | 0.485739 | 0.592165 |
| unseen_reader_and_text | 2 | 404 | 0.5 | 0.711992 | 0.779010 |
| unseen_reader_and_text | 3 | 234 | 0.5 | 0.638598 | 0.731672 |
| unseen_text | 0 | 913 | 0.5 | 0.768083 | 0.864411 |
| unseen_text | 1 | 861 | 0.5 | 0.740099 | 0.826922 |
| unseen_text | 2 | 797 | 0.5 | 0.743052 | 0.838648 |
| unseen_text | 3 | 983 | 0.5 | 0.685067 | 0.797422 |

d3opt_0024_2d9a9f9c46, from
`results/d3_eyebench_protocol_aligned_optimization_v1_20260522_074957/typ/d3_optimized_trial_predictions.csv`:

| split | fold | rows | threshold | BA | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| unseen_reader | 0 | 884 | 0.65 | 0.701384 | 0.840075 |
| unseen_reader | 1 | 906 | 0.58 | 0.616495 | 0.733098 |
| unseen_reader | 2 | 799 | 0.41 | 0.789815 | 0.890404 |
| unseen_reader | 3 | 965 | 0.78 | 0.545700 | 0.821726 |
| unseen_reader_and_text | 0 | 321 | 0.37 | 0.644907 | 0.730401 |
| unseen_reader_and_text | 1 | 269 | 0.60 | 0.489553 | 0.603849 |
| unseen_reader_and_text | 2 | 404 | 0.59 | 0.721512 | 0.783832 |
| unseen_reader_and_text | 3 | 234 | 0.63 | 0.610642 | 0.653463 |
| unseen_text | 0 | 913 | 0.40 | 0.792814 | 0.862047 |
| unseen_text | 1 | 861 | 0.39 | 0.753953 | 0.851609 |
| unseen_text | 2 | 797 | 0.55 | 0.758016 | 0.849453 |
| unseen_text | 3 | 983 | 0.45 | 0.734193 | 0.807065 |

Local Logistic anchor, from
`results/official_eyebench_baseline_evaluator_closure_v1_20260522_015211/baseline/logistic/local_official_derived_predictions.csv`:

| split | fold | rows | threshold | BA | AUROC |
| --- | ---: | ---: | --- | ---: | ---: |
| unseen_reader | 0 | 884 | 0.5 | 0.783847 | 0.881388 |
| unseen_reader | 1 | 906 | 0.5 | 0.651374 | 0.731677 |
| unseen_reader | 2 | 799 | 0.5 | 0.811566 | 0.890936 |
| unseen_reader | 3 | 965 | 0.5 | 0.769811 | 0.817751 |
| unseen_reader_and_text | 0 | 321 | 0.5 | 0.745833 | 0.849074 |
| unseen_reader_and_text | 1 | 269 | 0.5 | 0.514364 | 0.598213 |
| unseen_reader_and_text | 2 | 404 | 0.5 | 0.726211 | 0.786343 |
| unseen_reader_and_text | 3 | 234 | 0.5 | 0.565456 | 0.530236 |
| unseen_text | 0 | 913 | 0.5 | 0.812363 | 0.873925 |
| unseen_text | 1 | 861 | 0.5 | 0.758656 | 0.834325 |
| unseen_text | 2 | 797 | 0.5 | 0.771905 | 0.832661 |
| unseen_text | 3 | 983 | 0.5 | 0.723163 | 0.785270 |

## Top 10 /goal Candidates

Top 10 ranking is by inner-validation `selection_score`, not by official Test
metrics. Only the selected candidate was evaluated on official Test labels by
the /goal campaign. The remaining top-10 candidates have inner-validation fold
metrics only.

| rank | candidate | inner selection BA | inner unseen_reader BA/AUROC | inner unseen_text BA/AUROC | inner both-unseen BA/AUROC | mean threshold |
| ---: | --- | ---: | --- | --- | --- | ---: |
| 1 | d3opt_0024_2d9a9f9c46 | 0.770324 | 0.825532 / 0.854148 | 0.748530 / 0.814239 | 0.736911 / 0.796334 | 0.533333 |
| 2 | d3opt_0036_1d46253469 | 0.770160 | 0.803491 / 0.852336 | 0.773123 / 0.843759 | 0.733866 / 0.794644 | 0.525000 |
| 3 | d3opt_0009_29325c8105 | 0.769238 | 0.801668 / 0.852332 | 0.772558 / 0.843680 | 0.733489 / 0.794479 | 0.522500 |
| 4 | d3opt_0017_e55ad47811 | 0.767871 | 0.821641 / 0.836902 | 0.749904 / 0.810333 | 0.732068 / 0.785373 | 0.545000 |
| 5 | d3opt_0054_be5ccbbec5 | 0.767763 | 0.763450 / 0.814121 | 0.798069 / 0.871146 | 0.741769 / 0.805961 | 0.264167 |
| 6 | d3opt_0008_df66566550 | 0.767034 | 0.798386 / 0.830129 | 0.765247 / 0.828295 | 0.737471 / 0.791925 | 0.495000 |
| 7 | d3opt_0014_c5981f7cbf | 0.766929 | 0.797732 / 0.844796 | 0.764500 / 0.839848 | 0.738556 / 0.800575 | 0.504167 |
| 8 | d3opt_0006_20828c33ee | 0.766498 | 0.779749 / 0.815833 | 0.772830 / 0.835424 | 0.746914 / 0.795692 | 0.500833 |
| 9 | d3opt_0029_7543248d5b | 0.766268 | 0.808208 / 0.846320 | 0.748128 / 0.811450 | 0.742468 / 0.789019 | 0.528333 |
| 10 | d3opt_0056_76e63ddcc9 | 0.766163 | 0.814296 / 0.850001 | 0.750100 / 0.812601 | 0.734093 / 0.779954 | 0.551667 |

Full fold-level inner-validation rows for these candidates are in
`analysis/d3_eyebench_protocol_aligned_optimization_v1/tables/inner_validation_fold_metrics.csv`.

## Direct Answers

Did d3opt_0024 improve any regime relative to previous D3 Lite?

- BA: yes, unseen_text improved from `0.734075` to `0.759744`.
- AUROC: yes, unseen_reader improved from `0.808507` to `0.821326`, and
  unseen_text improved from `0.831851` to `0.842543`.
- It regressed on unseen_reader BA and both-unseen BA/AUROC.

Did any candidate beat previous D3 Lite on any regime?

- Official Test: only d3opt_0024 was evaluated, and it beat D3 Lite on unseen_text
  BA, unseen_reader AUROC, and unseen_text AUROC.
- Runner-up /goal candidates: unknown on official Test because they were not
  evaluated on official Test.
- Inner validation: several top candidates have inner-validation per-regime
  values above previous D3 Lite Test values, but those are not comparable Test
  evidence.

Did any candidate beat previous D3 Lite on both-unseen AUROC?

- Official Test: no. d3opt_0024 both-unseen AUROC was `0.692886`, below previous
  D3 Lite `0.715385`.
- Runner-up candidates: unknown on official Test.

Did any candidate beat Logistic anchor on any metric?

- Official Test BA: d3opt_0024 did not beat the Logistic anchor in any regime.
- Official Test AUROC: d3opt_0024 beat the Logistic anchor on unseen_text AUROC
  (`0.842543` vs `0.831545`) and both-unseen AUROC (`0.692886` vs `0.690967`).
- Runner-up candidates: unknown on official Test.
