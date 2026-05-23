# D3OnlineTargetedOptimizationAuditAndRerun v2

This v2 analysis audits the D3OnlineTargetedOptimization v1 artifacts and reruns
a stricter online selection layer. It separates full-evidence/offline-like rows
from late, mid, early, and stopping-detector online rows.

The v2 clean-selection rules are:

- `prefix_value=all` is allowed only for `offline_all_full_evidence`.
- `stopping_policy=no_stop` is allowed only for full-evidence/offline-like baselines.
- Online selected detectors must be selected from inner-validation evidence only.
- Oracle/test-label rows are not mixed into clean selection.
- Official EyeBench SOTA is not claimed by this project-specific rerun.

The output categories are:

- `offline_all_full_evidence`: reports full-reader/full-sequence upper-bound behavior.
- `online_late_accumulation`: uses at most 1000 words or 5 texts.
- `online_mid_detection`: uses at most 500 words or 3 texts.
- `online_early_detection`: uses at most 250 words or 1 text.
- `online_stopping_detector`: must stop before full evidence using learned inner thresholds.
- `unseen_text_specialist`: diagnostic legal candidate for the weak unseen-text regime.

The primary v2 selection score is the same weighted project-specific score used in v1:

`0.35 * AUROC + 0.25 * PR-AUC + 0.20 * BA + 0.10 * (1 - Brier) + 0.10 * earliness_score`

Primary regimes are `unseen_reader` and `unseen_reader_and_text`. The report also
keeps an internal simple mean comparison against published CopCo_TYP baselines, but
that internal mean is not an official EyeBench average.
