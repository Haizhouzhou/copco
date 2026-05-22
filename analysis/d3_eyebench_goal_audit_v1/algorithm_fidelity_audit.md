# Algorithm Fidelity Audit

## Classification

```yaml
algorithm_optimized: d3_reduced_wrapper
```

The /goal campaign optimized generated residual gaze-summary feature variants
with shallow sklearn classifiers. It did not preserve previous
`D3_EyeBench_Lite` exactly, did not include the previous Lite adapter as an
anchor candidate, and did not attempt full D3 feature families. The implemented
search is best classified as a reduced D3 wrapper, not `full_d3` and not a
monotonic `d3_lite` optimization.

## /goal Candidate Feature Space

Defined in `src/copco_eye_bench/d3_eyebench_protocol_optimization.py` and
`configs/d3_eyebench_protocol_aligned_optimization_v1.yaml`.

Predictor sets used for residualization:

- `surface`: word length, word frequency, trial IA count, normalized ID,
  start/end of line, content-word flag.
- `surface_surprisal`: `surface` plus `gpt2_surprisal`.
- `surface_surprisal_syntax`: `surface_surprisal` plus syntactic dependency
  counts and head-distance fields.

Outcome sets:

- `duration_core`: first fixation duration, first pass duration, go-past time,
  total fixation duration.
- `duration_plus_count`: `duration_core` plus fixation count.
- `all_gaze`: `duration_plus_count` plus skipping.

Aggregation sets:

- `central_spread`: mean, median, standard deviation.
- `robust_full`: mean, median, standard deviation, q25, q75, IQR, absolute mean.

Transforms:

- raw
- `log1p_duration`

Classifiers:

- logistic regression
- random forest
- extra trees
- gradient boosting

Selected candidate `d3opt_0024_2d9a9f9c46` used:

- `predictor_set: surface_surprisal_syntax`
- `outcome_set: duration_plus_count`
- `aggregation_set: central_spread`
- `transform: log1p_duration`
- `classifier: logistic_regression`
- 9 final features after availability and numeric-safety filtering.

## Previous D3_EyeBench_Lite Feature Space

Previous D3 Lite is implemented in
`src/copco_eye_bench/official_eyebench_sota_check.py`:

- Function `evaluate_d3_eyebench_lite`
- Function `_trial_residual_features`

Previous Lite residualization predictors:

- `word_length`
- `wordfreq_frequency`
- `gpt2_surprisal`
- `TRIAL_IA_COUNT`
- `normalized_ID`
- `start_of_line`
- `end_of_line`
- `is_content_word`

Previous Lite outcomes:

- first fixation duration
- first pass duration
- go-past time
- total fixation duration
- skipping
- fixation count

Previous Lite aggregation:

- mean
- median
- standard deviation

Previous Lite classifier:

- logistic regression through `_model_pipeline("logistic_regression", task="typ")`
- fixed `0.5` prediction threshold for BA/F1

Previous D3 Lite reported `n_features = 12` in the official-compatible metrics.
The selected /goal candidate reported `n_features = 9`.

## Component Questions

| question | answer |
| --- | --- |
| Were previous D3_EyeBench_Lite features preserved? | No. Some ingredient families overlap, but the exact previous D3 Lite feature generator was not included as a candidate and the selected candidate used a different predictor set, outcome set, transform, threshold policy, classifier hyperparameters, and final feature count. |
| Were full D3 feature families attempted? | No. The runtime-fix feature report explicitly says full D3 is not claimed unless all DFM residual profile inputs are available from official data. The /goal only used official EyeBench-loaded fields and generated residual aggregate variants. |
| Were word-level CopCo gaze features used? | Only official EyeBench IA/trial fields loaded from official processed CopCo data were used. Full prepared CopCo joins were prohibited and not used. |
| Were surprisal/gaze interactions used? | No explicit interaction features were generated. Surprisal was used as a residualization predictor in some predictor sets. |
| Were residual gaze features used? | Yes. The /goal candidates are residual gaze summary features. |
| Were sequence/distributional summaries used? | Limited distributional summaries were used for candidates with `robust_full`; selected d3opt_0024 used central/spread summaries only. No sequence model or temporal D3 architecture was attempted. |
| Was this a true D3 optimization or mostly a logistic wrapper over reduced features? | Mostly a classifier wrapper over reduced residual feature summaries. |
| Which D3 components were unavailable? | Full D3/DFM residual profile inputs outside official EyeBench fields were unavailable under the proven leakage policy. Full prepared CopCo joins were prohibited unless exact mapping and leakage policy were proven. |
| Which D3 components were intentionally not explored? | Full prepared CopCo joins, target-derived fields, identity/exposure/fold predictors, synthetic outputs, random predictions, sequence models, full D3 feature families, and candidate selection on Test labels. |
| Did old constraints such as "no new feature families" limit the search? | The prior runtime-fix and closure configs had `no_feature_engineering_search: true` for the frozen D3 Lite evidence. The /goal config did allow a bounded residual-feature grid, but it still intentionally constrained the search to official EyeBench fields and did not include full D3 feature families or the exact prior D3 Lite as a candidate. |

## Fidelity Conclusion

The /goal result should not be interpreted as an optimization of the full D3
algorithm. It is an official-fold, leakage-controlled optimization of a reduced
residual-feature wrapper. The selected lower average BA therefore does not
constitute evidence against the potential of full D3 or even against the prior
`D3_EyeBench_Lite` anchor.
