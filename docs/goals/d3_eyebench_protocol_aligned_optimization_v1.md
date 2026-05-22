# D3 EyeBench Protocol-Aligned Optimization v1

This document defines the bounded optimization campaign for D3 on official
EyeBench `CopCo_TYP`. It is a legal optimization protocol: it permits only
changes that preserve official data, official folds, official test separation, and
leakage controls. It is not a new benchmark definition, a baseline-rerun campaign,
or a mechanism for post-hoc tuning on test labels.

## Goal

Optimize D3 under the official EyeBench `CopCo_TYP` Test protocol while keeping
the scientific claim boundary explicit:

- Use only official EyeBench `CopCo_TYP` data and official EyeBench folds.
- Evaluate the primary result with trial-level balanced accuracy under the
  official `CopCo_TYP` Test protocol.
- Use the published EyeBench leaderboard as a fixed external reference.
- Use the reproduced local Logistic baseline as a pipeline anchor, not as a
  moving target.
- Decide the final claim category using the gates in this protocol.

## Fixed Inputs

The campaign may use only these fixed inputs:

- Official EyeBench `CopCo_TYP` processed data as loaded by the official EyeBench
  `CopCo_TYP` task configuration.
- Official EyeBench `CopCo_TYP` train/validation/test folds and split semantics.
- The campaign-start published leaderboard snapshot, recorded in the experiment
  manifest with URL or citation, retrieval date, task name, split/protocol name,
  metric name, and reported score values.
- The reproduced local Logistic baseline from the official-compatible pipeline,
  used only to verify data loading, fold alignment, label orientation, evaluator
  conventions, and result-format compatibility.

The published leaderboard snapshot is fixed for this v1 campaign. If the public
leaderboard changes after the snapshot is recorded, those changes require a new
protocol version or an explicit addendum before they can affect the claim.

## Baseline Policy

Official baselines must not be rerun as part of the optimization loop. A rerun is
allowed only when needed for sanity, such as checking that an environment repair,
fold loader, label direction, metric implementation, or evaluator interface is
working. Any such rerun must be logged as a sanity check and must not replace the
fixed published leaderboard reference.

The reproduced local Logistic baseline is the required pipeline anchor. Before
D3 results are interpreted, the campaign must show that the local Logistic anchor:

- uses official EyeBench `CopCo_TYP` data and folds;
- writes the expected prediction and metric artifacts;
- matches the expected label orientation and task definition;
- produces metrics through the same local evaluator path used for D3; and
- remains separate from the published leaderboard reference.

## Optimization Boundary

Optimization is allowed only on training data and inner-validation data derived
from the official training portion of each official fold. The campaign may tune:

- D3 model hyperparameters;
- feature normalization, imputation, calibration, and regularization choices;
- class weighting or sampling inside training only;
- threshold selection for thresholded metrics; and
- random seeds used by deterministic, logged training runs.

The final Test configuration must be selected before seeing official Test labels
or Test metrics. Thresholds must be selected on train/inner-validation data only
and then frozen before Test evaluation. AUROC and PR-AUC must be computed from
probabilities or scores without test-label-dependent threshold adjustment.

The campaign is bounded as follows:

- One primary D3 candidate family is selected before any official Test result is
  used for decision-making.
- Hyperparameter search space, seeds, folds, feature set, and threshold-selection
  rule are declared in a config before each run.
- Optional ablations or runner-up candidates may be evaluated only if they were
  predeclared in configs before the primary Test result is inspected.
- No candidate may be promoted, replaced, or retuned because of official Test
  labels, official Test metrics, or leaderboard-position feedback.

## Predictor Denylist

D3 candidates must not use any of the following as predictors, direct features,
join keys promoted to predictors, encodings, target encodings, residualizer inputs,
calibration predictors, threshold-selection groups, or model-selection signals:

- `participant_id`;
- `speech_id`;
- `text_id`;
- fold identifiers or split identifiers;
- exposure-count variables;
- target-derived predictors;
- labels, future labels, or variables computed from labels;
- leaderboard scores or Test metrics;
- any proxy feature created primarily to recover participant identity, text
  identity, speech identity, fold membership, or target labels.

Participant, speech, text, and fold identifiers may appear only in manifests,
leakage reports, grouping checks, fold construction audits, and reader-aggregated
secondary reporting. They must not enter training, hyperparameter selection,
threshold selection, calibration, or prediction.

## Data-Join Policy

The campaign must not use full prepared CopCo data joins by default. Such joins
are disallowed unless the exact official EyeBench-to-CopCo mapping and the
leakage policy for every joined field are proven in a written report before use.

Any proposed join outside official EyeBench data must document:

- row-level mapping keys and cardinality;
- whether each joined field is available at prediction time under the official
  protocol;
- whether the field leaks participant identity, text identity, fold membership,
  exposure count, label information, or target-derived information;
- deterministic checks that the join does not create duplicate or dropped trials;
- an explicit allow/deny decision for each joined field; and
- a config/version bump before the field can be used.

Until those conditions are met, D3 uses only official EyeBench-loaded fields and
features derived from them without external full-prepared-data joins.

## Prohibited Shortcuts

The following are prohibited for all D3 optimization, validation, reporting, and
claim decisions:

- synthetic outputs;
- random predictions;
- predictions not produced by a logged D3 model run;
- test-label tuning;
- threshold selection using official Test labels or official Test metrics;
- post-hoc candidate selection using Test results;
- leaderboard-chasing after the campaign-start reference snapshot;
- CPU or login-node execution for heavy jobs that should run under Slurm; and
- committing raw data, copied data, large result artifacts, caches, model
  checkpoints, or environment directories.

Randomness is allowed only as a logged training or optimization seed. It must not
be used as a prediction generator, baseline substitute, or claim-supporting output.

## Metrics

Primary metric:

- Trial-level balanced accuracy under the official EyeBench `CopCo_TYP` Test
  protocol.

Secondary metrics:

- AUROC;
- PR-AUC;
- macro F1;
- Brier score;
- reader-aggregated metrics.

Secondary metrics must be reported as diagnostics and interpretation support.
They cannot override the primary trial-level balanced-accuracy decision. Reader
aggregation must use a predeclared aggregation rule, such as mean predicted
probability per reader, and any reader-level threshold must be selected on
train/inner-validation data only.

## Execution Policy

All heavy D3 experiments must run through `sbatch` with the UZH teaching account
profile:

```bash
#SBATCH --partition=teaching
#SBATCH --account=mlnlp2.pilot.s3it.uzh
#SBATCH --qos=normal
```

Each heavy job must request CPU, memory, GPU, and wall time appropriate to the
configured workload. GPU resources may be requested only when the code actually
uses them. CPU-only heavy jobs must still run through `sbatch` with the same
teaching/account/QoS policy and sufficient CPU and memory.

Every heavy job must record:

- exact `sbatch` script and command;
- Git commit or working-tree state;
- active environment;
- Slurm job ID;
- hostname;
- allocated CPU, memory, GPU, and wall time;
- CUDA/GPU visibility when applicable;
- stdout and stderr paths;
- output directory;
- config path and config hash when feasible; and
- post-run Slurm accounting summary when available.

Heavy work must not run on a login node. Failed or downgraded resource attempts
must be recorded with the attempted request, failure reason, and selected
replacement request.

## Experiment Logging

Every experiment must create or update a manifest containing:

- campaign protocol version;
- task name `CopCo_TYP`;
- official data/fold references;
- published leaderboard snapshot reference;
- local Logistic anchor reference;
- D3 candidate name;
- config path;
- seed list;
- feature list and predictor-denylist audit result;
- hyperparameter search space;
- threshold-selection rule;
- train/inner-validation metrics;
- final Test metrics;
- stdout and stderr paths;
- leakage report path;
- code version;
- environment summary;
- Slurm resource summary; and
- decision category.

Leakage reports must explicitly check for denied predictors, target-derived
predictors, fold leakage, identity leakage, exposure-count leakage, duplicate or
missing trial rows, and mismatch between prediction rows and official evaluator
rows.

## Validation Gates

A D3 result is official-compatible only if all gates pass:

- official EyeBench `CopCo_TYP` data are used;
- official EyeBench folds and Test protocol are used;
- local Logistic anchor confirms pipeline compatibility;
- candidate config was declared before Test evaluation;
- no official Test labels or Test metrics were used for tuning;
- denied predictors are absent from training, thresholding, calibration, and
  model-selection inputs;
- no synthetic outputs or random predictions are used;
- prediction rows match the official evaluator expectation;
- primary and secondary metrics are generated by the logged evaluator path;
- stdout, stderr, config, seeds, metrics, and leakage reports are present; and
- heavy jobs were run through the required `sbatch` teaching/account/QoS profile.

## Final Decision Categories

`official_sota_claim_allowed`

D3 is official-compatible, all validation gates pass, and the final locked D3
candidate beats the fixed published EyeBench `CopCo_TYP` leaderboard reference on
the primary trial-level balanced-accuracy metric under the official Test protocol.
Secondary metrics are reported but do not define the SOTA claim.

`official_compatible_d3_improved_but_not_sota`

D3 is official-compatible and improves over the reproduced local Logistic anchor
on the primary metric, but it does not beat the fixed published leaderboard
reference or the leaderboard comparison is insufficient for an official SOTA
claim.

`official_compatible_but_not_sota`

D3 is official-compatible, all validation gates pass, and the result can be
reported as an official-compatible D3 result, but it does not improve over the
local Logistic anchor and does not support an official SOTA claim.

`optimization_inconclusive`

The campaign completes enough work to produce partial evidence, but the evidence
is insufficient for one of the official-compatible decision categories. Examples
include incomplete predeclared candidate coverage, unstable validation, failed
sanity checks that do not identify a data/environment/evaluator blocker, or
insufficient evidence to interpret the local Logistic anchor.

`blocked_by_environment`

The campaign cannot complete because the required runtime, dependency, Slurm, or
execution environment cannot be made usable without changing the protocol.

`blocked_by_data`

The campaign cannot complete because official EyeBench `CopCo_TYP` data, folds,
mapping evidence, or permitted fields are unavailable, inconsistent, or
insufficient for official-compatible evaluation.

`blocked_by_evaluator`

The campaign cannot complete because the official-compatible evaluator path,
result format, metric computation, or row alignment cannot be validated.

## Commit Policy

Commit and push only validated code, configs, scripts, protocol documents, and
reports. Do not commit raw data, copied datasets, full result directories, large
prediction files, caches, model checkpoints, local environments, or other large
data/results artifacts.
