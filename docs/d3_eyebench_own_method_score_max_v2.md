# D3 EyeBench Own-Method Score Maximization v2

This implementation executes
`docs/goals/d3_eyebench_own_method_score_maximization_v2.md`.

The phase is restricted to official EyeBench `CopCo_TYP` data and folds. It
does not rerun official leaderboard methods, does not use W&B online APIs, and
does not reproduce AhnCNN, Random Forest, SVM, BEyeLSTM, RoBERTEye, PLM-AS,
Reading Speed, Text-only Roberta, or other published baselines.

## Runner

- Config: `configs/d3_eyebench_own_method_score_max_v2.yaml`
- Code: `src/copco_eye_bench/d3_eyebench_own_method_score_max.py`
- Slurm templates: `scripts/slurm/d3_eyebench_own_method_score_max_v2/`
- Reports: `analysis/d3_eyebench_own_method_score_max_v2/`

The first candidate is always `candidate_0000`, the exact previous
`D3_EyeBench_Lite` adapter. It is evaluated first and must reproduce the prior
per-regime trial-level BA/AUROC values within tolerance before any search runs.

## Primary Metric

The primary evidence is per-regime official-trial-fold balanced accuracy and
AUROC for:

- `unseen_reader`
- `unseen_text`
- `unseen_reader_and_text`

The runner also reports `internal_simple_mean` BA and AUROC. This value is not
called the official EyeBench leaderboard average unless the official average
logic is separately reproduced.

## Candidate Families

All new candidates preserve the previous D3 Lite feature adapter first, then add
or alter only D3-family components:

- D3 Lite calibration variants
- D3 Lite plus robust residual summaries
- D3 Lite plus official word-level raw gaze summaries
- D3 Lite plus text/gaze interaction summaries
- D3 Lite plus fuller official-data residual extensions

Random forest and extra-trees candidates, when enabled, are D3-feature models
only and are not official baseline reproductions.

## Legal Gates

The runner rejects or reports failures for:

- missing `candidate_0000` anchor reproduction;
- missing official data/folds;
- test-label threshold or hyperparameter tuning;
- synthetic or random predictions;
- full prepared CopCo joins;
- prohibited predictors such as `participant_id`, `speech_id`, `text_id`,
  `fold_id`, exposure counts, or target-derived variables; and
- held-out reader/text leakage in feature fitting.

Final reports never claim official SOTA by default. Published leaderboard
numbers and the reproduced local Logistic anchor are fixed references only.
