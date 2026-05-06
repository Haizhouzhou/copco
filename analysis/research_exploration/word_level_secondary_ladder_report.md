# Word-Level Secondary Ladder Report

This is a secondary analysis because labels are participant-level and word rows are not independent. Only participant-grouped folds are used; no random word-level split is used.

- Deterministically sampled rows: 120000 of 335203

## Metrics
| stage | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score | n_predictions | usable_folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gaze_only | 0.5958 | 0.3565 | 0.5918 | 0.5812 | 0.2374 | 120000 | 5 |
| gaze_plus_lexical_classical | 0.6130 | 0.3665 | 0.5930 | 0.5810 | 0.2362 | 120000 | 5 |
| gaze_plus_dfm_lm | 0.6069 | 0.3623 | 0.5938 | 0.5825 | 0.2369 | 120000 | 5 |
| gaze_plus_segmentation | 0.5971 | 0.3571 | 0.5915 | 0.5809 | 0.2374 | 120000 | 5 |
| gaze_plus_dfm_plus_segmentation | 0.6073 | 0.3623 | 0.5935 | 0.5823 | 0.2369 | 120000 | 5 |
| full_validated_feature_set | 0.6143 | 0.3662 | 0.5927 | 0.5801 | 0.2362 | 120000 | 5 |
