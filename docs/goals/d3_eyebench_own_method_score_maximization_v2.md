# D3 EyeBench Own-Method Score Maximization v2

This document defines a new modeling and optimization phase for our own
D3-family method on official EyeBench `CopCo_TYP` data and folds.

This is not a blocker-report phase, baseline-reproduction phase, environment-fix
phase, or manuscript-claim update phase. Its purpose is to aggressively improve
our own D3-family method under legal, leakage-controlled official EyeBench
constraints and to collect every valid piece of evidence that supports the
strength, usefulness, robustness, and scientific value of the method.

## Scope

This phase must not reproduce official leaderboard methods. It must not rerun
AhnCNN, Random Forest, SVM, BEyeLSTM, RoBERTEye, PLM-AS, Reading Speed,
Text-only Roberta, or other published EyeBench baselines. It must not spend time
on W&B online API access, official leaderboard command-source reproduction, or
exact official environment reproduction.

Published EyeBench `CopCo_TYP` leaderboard numbers may be used only as fixed
comparison targets. They are not methods to rerun.

The phase starts from the current pushed audit branch:

```text
codex/d3-eyebench-goal-audit-v1
```

Work must happen on:

```text
codex/d3-eyebench-own-method-score-max-v2
```

Do not work directly on `main`. Do not discard prior reports. Do not modify
tracked EyeBench submodule source unless absolutely necessary. Keep EyeBench
data, environments, caches, W&B files, and generated results ignored.

## Prior Evidence

Official EyeBench `CopCo_TYP` data and folds exist. A local
official-derived `LogisticRegressionMLArgs` anchor was reproduced and is
available as a sanity reference. Published leaderboard values are available and
must be treated as fixed references.

Previous official-compatible `D3_EyeBench_Lite` trial-level metrics:

| split | BA | AUROC |
| --- | ---: | ---: |
| unseen_reader | 0.7274 | 0.8085 |
| unseen_text | 0.7341 | 0.8319 |
| unseen_reader_and_text | 0.6342 | 0.7154 |

Previous `D3_EyeBench_Lite` reader-aggregated secondary metrics:

| split | BA | AUROC |
| --- | ---: | ---: |
| unseen_reader | 0.7530 | 0.8468 |
| unseen_text | 0.7629 | 0.8606 |
| unseen_reader_and_text | 0.6201 | 0.7792 |

The previous protocol-aligned optimization campaign was audited and found
invalid as evidence against D3 potential because:

- previous `D3_EyeBench_Lite` was not included as `candidate_0000`;
- metrics differed from the previous D3 Lite anchor;
- the official leaderboard average was not reproduced;
- the campaign optimized a 9-feature reduced residual wrapper rather than the
  previous 12-feature D3 Lite adapter or full D3; and
- the audit conclusion was `goal_not_d3_algorithm_faithful`.

This v2 phase must fix those issues.

## Mandatory Invariants

### Invariant 1: Anchor Preservation

`candidate_0000` must be the exact previous `D3_EyeBench_Lite` adapter. It must
be evaluated first.

It must reproduce, within declared tolerance, the previous per-regime
trial-level metrics:

| split | BA | AUROC |
| --- | ---: | ---: |
| unseen_reader | 0.7274 | 0.8085 |
| unseen_text | 0.7341 | 0.8319 |
| unseen_reader_and_text | 0.6342 | 0.7154 |

If `candidate_0000` cannot be reproduced, stop optimization and audit the metric
or pipeline mismatch.

### Invariant 2: Monotonic Best-So-Far

`best_so_far` must initialize from `candidate_0000`.

The final selected candidate may not be worse than `candidate_0000` under the
declared primary metric. If no new candidate improves over `candidate_0000`, the
final best remains `candidate_0000`.

### Invariant 3: Method Fidelity

Do not optimize a reduced wrapper unless it is explicitly labeled exploratory.
The main campaign must preserve the previous D3 Lite feature adapter first.

New main-campaign candidates must be one of:

- D3 Lite plus legal enhancements;
- D3 Fuller official-data reconstruction;
- D3 calibration variants;
- D3 ensembles; or
- D3 sequence or word-level extensions.

Do not silently replace D3 with a generic sklearn baseline. Do not drop previous
D3 Lite features unless running a documented ablation.

### Invariant 4: No Leaderboard Reproduction

Do not rerun official published baselines. Do not rerun AhnCNN, Random Forest,
SVM, BEyeLSTM, RoBERTEye, PLM-AS, Reading Speed, Text-only Roberta, or other
leaderboard models. Use leaderboard values only as fixed comparison references.

### Invariant 5: Legal Optimization Only

The campaign must obey all of these constraints:

- no test-label tuning;
- no synthetic predictions;
- no random prediction files;
- no manually entered metrics;
- no target-derived features;
- no `participant_id` as predictor;
- no `speech_id` or `text_id` as predictor;
- no `fold_id` as predictor;
- no exposure-count variables as predictors;
- no held-out reader rows in residualization for `unseen_reader`;
- no held-out text rows in residualization for `unseen_text`;
- no held-out readers or texts in residualization for
  `unseen_reader_and_text`;
- no `reader_group` or target variable in residualization;
- no full prepared CopCo feature joins unless exact official trial mapping is
  proven and the leakage policy is explicitly validated;
- no changing official folds;
- no changing labels; and
- no official SOTA claim unless the strict claim gates pass.

## Primary Optimization Objective

Primary objective: improve our own D3-family method's official-compatible
`CopCo_TYP` trial-level balanced accuracy while preserving or improving AUROC.

Primary metrics:

- per-regime trial-level balanced accuracy for `unseen_reader`, `unseen_text`,
  and `unseen_reader_and_text`;
- per-regime trial-level AUROC for the same regimes;
- simple mean trial-level BA, clearly labeled `internal_simple_mean`;
- simple mean trial-level AUROC, clearly labeled `internal_simple_mean`.

Do not call the simple mean the official leaderboard average. The visible
EyeBench Average BA may not equal the simple mean of the three regime columns.
If the official average computation is unknown, report per-regime values and
internal simple means separately.

Secondary evidence metrics:

- PR-AUC;
- macro F1;
- sensitivity/recall for the dyslexic class;
- specificity;
- Brier score;
- calibration curves or expected calibration error when cheap;
- reader-aggregated metrics as secondary only;
- both-unseen AUROC as a robustness/generalization signal;
- fold-level stability;
- confidence intervals or bootstrap intervals if cheap;
- ablation evidence showing which D3 components matter;
- feature importance, coefficients, SHAP, or permutation importance if feasible;
- legal error analysis by reader, text, or trial length; and
- robustness to threshold choices.

The campaign should collect evidence that D3 captures reader-level dyslexia
signal after aggregation.

## Targets

Hard target: beat the published `CopCo_TYP` Test leaderboard target under a
clearly declared primary metric, if legally possible.

Practical target: beat the reproduced local Logistic anchor on at least one of:

- average trial-level BA;
- two of three per-regime BAs; or
- hardest both-unseen AUROC plus competitive BA.

Minimum valuable outcome: produce a stronger D3-family candidate than previous
`D3_EyeBench_Lite`, or produce rigorous evidence that current D3 Lite is the
strongest among tested legal D3 variants while identifying the next feature gap.

## Allowed Legal Score-Improvement Methods

### Candidate Anchor And Ablations

- `candidate_0000` exact previous D3 Lite;
- D3 Lite ablations;
- D3 Lite plus feature additions;
- D3 Lite calibration variants; and
- D3 Lite ensembles.

### Feature Expansion From Official EyeBench Data

Allowed feature families include:

- word-level gaze aggregation features;
- first fixation duration;
- mean fixation duration;
- total fixation duration;
- first pass duration;
- go-past time;
- number of fixations;
- landing position;
- mean saccade duration;
- peak saccade velocity;
- skipping rates;
- regression indicators;
- distribution summaries including mean, median, standard deviation,
  quantiles, max, and trimmed means;
- per-trial robust summaries;
- per-text-normalized features;
- train-only residualized features;
- text/gaze interactions;
- surprisal by gaze interactions if computed from official text;
- word length by gaze interactions;
- frequency by gaze interactions if the frequency source is deterministic and
  documented; and
- sentence or paragraph position features when not text-ID leakage.

### Model Families For Our D3 Method

Allowed model families, only on D3 features:

- logistic regression with class weights;
- elastic-net logistic regression;
- calibrated logistic regression;
- linear SVM with calibrated probabilities if legal;
- gradient boosting or HistGradientBoosting;
- random forest as a D3-feature model, not as official baseline reproduction;
- small controlled MLP on D3 features;
- stacking or ensembling of D3 variants using train/validation only; and
- regime-specific hyperparameters selected only from training data.

### Calibration And Thresholding

Allowed calibration and thresholding:

- fixed `0.5` threshold;
- train-only threshold calibration;
- inner-validation threshold maximizing balanced accuracy;
- fold-specific threshold selected only from training/inner-validation;
- sigmoid, isotonic, or Platt probability calibration;
- class-weight search; and
- positive-class weight search.

### Evidence Generation

Generate per-regime result tables, fold-level result tables,
best-vs-anchor comparisons, ablation tables, feature group contribution tables,
calibration evidence, robustness evidence, reader aggregation evidence,
both-unseen generalization evidence, and failure analysis for the remaining gap
to the leaderboard reference.

## Required Campaign Stages

### Stage 0: Audit And Anchors

Locate previous D3 Lite reports and predictions. Re-evaluate
`candidate_0000` exactly. Re-evaluate the local Logistic anchor only if needed
for sanity, not as a reproduction task. Verify data, folds, and metrics.

### Stage 1: Metric Alignment

Implement one canonical metric engine for all D3 candidates. Report per-regime
metrics and internal simple means. Clearly separate fixed-threshold metrics from
calibrated-threshold metrics.

### Stage 2: Candidate Generation

Generate D3 candidates that preserve previous D3 Lite features. Add feature
families incrementally.

Each candidate must record:

- feature set;
- model type;
- calibration method;
- threshold method;
- hyperparameters;
- seed;
- regime;
- folds; and
- output paths.

### Stage 3: Legal Model Selection

Use only training/inner-validation data for hyperparameter and threshold
selection. Official test folds may be used for final reporting of a locked
candidate.

If exploratory search evaluates many candidates on official test folds, label
results exploratory and do not overclaim. Prefer nested or inner-validation
selection when feasible.

### Stage 4: Evidence Extraction

For the best candidate and top candidates, generate:

- trial-level metrics;
- reader-aggregated secondary metrics;
- fold-level stability;
- ablations;
- feature importance;
- calibration/threshold report;
- leakage validation; and
- prohibited predictor report.

### Stage 5: Decision

Use one of these final decision categories:

- `d3_method_improved`;
- `d3_method_competitive_but_not_improved`;
- `d3_method_exploratory_gain_only`;
- `d3_method_not_improved`;
- `blocked_by_environment`;
- `blocked_by_data`; or
- `blocked_by_evaluator`.

Do not use `official_sota_claim_allowed` unless strict official-claim gates pass.
Prefer careful wording such as:

- official-compatible D3-family improvement;
- D3 method evidence strengthened;
- D3 not yet official SOTA; and
- exploratory score-improvement result.

## Slurm And Resource Rules

Use `sbatch` for real jobs. Do not run heavy work on a login node. Do not use
tiny default resources.

For CPU-only D3 search jobs, use UZH teaching resources:

```bash
#SBATCH --partition=teaching
#SBATCH --account=mlnlp2.pilot.s3it.uzh
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=04:00:00
```

For GPU-heavy experiments, only when neural, MLP, or transformer features are
genuinely used:

```bash
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
```

Each `sbatch` job must write stdout, stderr, manifest JSON, command, Git commit,
config, runtime environment, start/end time, exit code, and metrics path.

## Required Outputs

Create these project artifacts for the implementation phase:

- `configs/d3_eyebench_own_method_score_max_v2.yaml`;
- `src/copco_eye_bench/d3_eyebench_own_method_score_max.py`;
- `tests/test_d3_eyebench_own_method_score_max.py`;
- `docs/d3_eyebench_own_method_score_max_v2.md`;
- `analysis/d3_eyebench_own_method_score_max_v2/`;
- `scripts/slurm/d3_eyebench_own_method_score_max_v2/`; and
- `logs/ai_runs/<timestamp>_d3_eyebench_own_method_score_max_v2.md`.

Required reports:

- `analysis/d3_eyebench_own_method_score_max_v2/anchor_reproduction_report.md`;
- `analysis/d3_eyebench_own_method_score_max_v2/metric_alignment_report.md`;
- `analysis/d3_eyebench_own_method_score_max_v2/candidate_search_manifest.md`;
- `analysis/d3_eyebench_own_method_score_max_v2/candidate_leaderboard.csv`;
- `analysis/d3_eyebench_own_method_score_max_v2/best_candidate_report.md`;
- `analysis/d3_eyebench_own_method_score_max_v2/feature_family_ablation_report.md`;
- `analysis/d3_eyebench_own_method_score_max_v2/calibration_threshold_report.md`;
- `analysis/d3_eyebench_own_method_score_max_v2/reader_aggregation_secondary_report.md`;
- `analysis/d3_eyebench_own_method_score_max_v2/both_unseen_generalization_report.md`;
- `analysis/d3_eyebench_own_method_score_max_v2/leakage_validation_report.md`;
- `analysis/d3_eyebench_own_method_score_max_v2/prohibited_predictor_report.md`;
- `analysis/d3_eyebench_own_method_score_max_v2/evidence_summary_for_manuscript.md`;
- `analysis/d3_eyebench_own_method_score_max_v2/final_decision.json`; and
- `analysis/d3_eyebench_own_method_score_max_v2/final_decision_report.md`.

Do not commit large result directories, prediction CSVs, Feather/Parquet files,
model artifacts, caches, environments, W&B files, or Slurm logs unless they are
tiny and explicitly useful.

## Validation

Run:

```bash
conda run -n copco python -m pip install -e .
conda run -n copco python scripts/validate_env.py
conda run -n copco python -m ruff check .
conda run -n copco python -m pytest tests/test_d3_eyebench_own_method_score_max.py -q
conda run -n copco python -m pytest tests/ -q
```

If full pytest fails only due resource issues, rerun via `sbatch` and document.

Also validate:

- no synthetic predictions;
- `candidate_0000` included;
- `best_so_far` initialized from `candidate_0000`;
- final best not worse than `candidate_0000` under the declared primary metric;
- no prohibited predictors;
- no leakage;
- no leaderboard reproduction;
- no official baseline reruns; and
- no large generated files staged.

## Commit And Push Policy

Commit only validated code, configs, small reports, docs, tests, `sbatch`
templates, and decision artifacts.

Suggested implementation commit message:

```text
feat: optimize D3 EyeBench own-method score evidence
```

Push implementation work to:

```bash
git push origin codex/d3-eyebench-own-method-score-max-v2
```

## Final Response Requirements For Implementation Phase

The final implementation response must report:

- branch;
- commit;
- push status;
- whether `candidate_0000` reproduced previous D3 Lite;
- primary metric definition;
- whether official leaderboard average was used or avoided;
- number of candidates;
- candidate families explored;
- best candidate ID;
- best candidate per-regime BA/AUROC;
- best candidate internal simple mean BA/AUROC;
- comparison to previous D3 Lite;
- comparison to local Logistic anchor as reference only;
- comparison to published leaderboard as reference only;
- reader-aggregated secondary evidence;
- both-unseen evidence;
- calibration/threshold evidence;
- ablation findings;
- feature importance findings if available;
- leakage validation;
- prohibited predictor validation;
- final decision category;
- whether our method evidence improved;
- whether official SOTA is claimed;
- manuscript/supplement updates if any;
- validation commands/status;
- large files not committed; and
- remaining limitations.

## Most Important Instruction

Do not reproduce leaderboard methods. Do not optimize a reduced wrapper. Preserve
previous D3 Lite as `candidate_0000`, improve our own D3-family method legally,
and collect all valid evidence that makes our method look strong without
overclaiming.
