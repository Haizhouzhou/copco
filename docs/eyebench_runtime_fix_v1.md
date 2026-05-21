# OfficialEyeBenchRuntimeFix v1

OfficialEyeBenchRuntimeFix v1 is an infrastructure-only gate for the frozen CopCo
D3 residualized DFM gaze-profile method. It does not change the frozen Phase 4,
BenchmarkBridge, or D3 full-data result.

The phase separates three facts that must not be collapsed:

- BenchmarkBridge full-data results remain internal EyeBench-style,
  benchmark-relative results.
- Official EyeBench claims require official EyeBench processed CopCo data, official
  folds, official evaluator compatibility, and at least one reproduced official
  CopCo_TYP baseline.
- D3_EyeBench_Lite is only an official-compatible adapter if it uses official
  EyeBench processed data/folds and passes leakage and predictor gates.

Runtime artifacts are isolated under `eyebench/`:

- `eyebench/.envs/`
- `eyebench/.conda_pkgs/`
- `eyebench/.pip_cache/`
- `eyebench/.cache/`
- `eyebench/.runtime_logs/`
- `eyebench/wandb/`

The revised runtime rule is:

- Use `conda run -n copco ...` first for CopCo-side and EyeBench-side commands.
- Run EyeBench source through `PYTHONPATH=$PWD:$PWD/src`; do not install
  `eyebench` editable into the Python 3.11 CopCo environment first.
- Repair imports incrementally, installing only concrete missing packages.
- Create `eyebench/.envs/eyebench_official_py312_minimal` only if the CopCo
  environment fails because the official EyeBench source requires Python 3.12.
- Do not claim exact `environment.yml` reproduction unless that environment was
  actually solved and used.
- Classify successful non-`environment.yml` execution as
  `official-code/data/fold-compatible`, subject to the data/fold/evaluator/baseline
  gates.

Generated phase outputs go under ignored
`results/official_eyebench_runtime_fix_v1_<timestamp>/` directories. No EyeBench
processed data, caches, environments, model artifacts, or generated prediction
files should be committed.

The decision categories are:

- `official_sota_claim_allowed`
- `official_compatible_but_not_sota`
- `benchmark_relative_sota_only`
- `blocked_by_environment`
- `blocked_by_data`
- `blocked_by_evaluator`
- `blocked_by_baseline_reproduction`

Official SOTA is allowed only when all official runtime gates pass. Otherwise the
manuscript must retain benchmark-relative wording.
