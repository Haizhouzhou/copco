# Metric Alignment Report

- Primary metric: per-regime trial-level balanced accuracy on official `CopCo_TYP` Test folds, with internal simple mean reported separately.
- AUROC is computed from probabilities/scores, not hard labels.
- The internal simple mean is not called the official leaderboard average.
- Thresholds for new candidates are fixed at 0.5 or selected only on train/inner-validation.
- Candidate selection for the locked candidate is based on inner validation; top-k Test evaluations are labeled exploratory evidence.

- Visible official average used as primary: False
- Internal simple mean used: True
- Test-label tuning: False
