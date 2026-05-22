# D3 EyeBench Own-Method Score Maximization v2 Execution

## Request Summary

Execute `docs/goals/d3_eyebench_own_method_score_maximization_v2.md` as the
active modeling/search contract. Implement the D3-family own-method
score-maximization pipeline, run real candidate experiments through Slurm,
preserve `candidate_0000` as the exact previous D3_EyeBench_Lite anchor, and
commit/push validated code and small reports.

## Plan

1. Inspect the branch, environment, prior D3 Lite artifacts, prior audit, and
   existing EyeBench/D3 code.
2. Implement config, runner, tests, docs, and Slurm templates for the v2
   own-method campaign.
3. Run static validation and unit tests.
4. Submit the mandatory `candidate_0000` anchor checkpoint through `sbatch`.
5. If the anchor reproduces, submit the full legal D3-family search through
   `sbatch`.
6. Validate outputs, copy only small reports into tracked analysis paths, then
   commit and push.

## Files Modified

- `configs/d3_eyebench_own_method_score_max_v2.yaml`
- `src/copco_eye_bench/d3_eyebench_own_method_score_max.py`
- `src/copco_eye_bench/cli.py`
- `pyproject.toml`
- `tests/test_d3_eyebench_own_method_score_max.py`
- `docs/d3_eyebench_own_method_score_max_v2.md`
- `scripts/slurm/d3_eyebench_own_method_score_max_v2/run_anchor.sbatch`
- `scripts/slurm/d3_eyebench_own_method_score_max_v2/run_full_search.sbatch`
- `scripts/slurm/d3_eyebench_own_method_score_max_v2/run_full_pytest.sbatch`
- `logs/ai_runs/2026-05-22_1136_d3_eyebench_own_method_score_max_v2_execution.md`
- `logs/ai_runs/INDEX.md`

## Commands Run

- `pwd`
- `git status --short --branch --untracked-files=all`
- `python --version`
- `conda run -n copco python --version`
- `conda run -n copco python -c "import sys; print(sys.executable)"`
- repository and artifact inventory commands with `rg`, `find`, and `sed`
- official EyeBench processed-data schema inspection
- `conda run -n copco python -m pip install -e .`
- `conda run -n copco python scripts/validate_env.py`
- `conda run -n copco python -m ruff check src/copco_eye_bench/d3_eyebench_own_method_score_max.py tests/test_d3_eyebench_own_method_score_max.py src/copco_eye_bench/cli.py`
- `conda run -n copco python -m pytest tests/test_d3_eyebench_own_method_score_max.py -q`
- `sbatch scripts/slurm/d3_eyebench_own_method_score_max_v2/run_anchor.sbatch`
- `sbatch scripts/slurm/d3_eyebench_own_method_score_max_v2/run_full_search.sbatch`
- `sbatch scripts/slurm/d3_eyebench_own_method_score_max_v2/run_full_pytest.sbatch`
- Slurm monitoring with `squeue`, `sstat`, `sacct`, `scontrol show job`, and log tails.
- `conda run -n copco copco-validate-d3-eyebench-own-method-score-max --config configs/d3_eyebench_own_method_score_max_v2.yaml --output-dir results/d3_eyebench_own_method_score_max_v2_20260522_115731`
- `conda run -n copco python -m ruff check .`
- `conda run -n copco python -m pytest tests/test_d3_eyebench_own_method_score_max.py -q`
- `conda run -n copco python -m pytest tests/ -q`

## Validation Results

- Editable install in `copco`: passed.
- `scripts/validate_env.py`: passed.
- Targeted Ruff for the new runner/CLI/tests: passed.
- `tests/test_d3_eyebench_own_method_score_max.py -q`: passed (`5 passed`).
- Campaign output validator for
  `results/d3_eyebench_own_method_score_max_v2_20260522_115731`: passed with
  decision `d3_method_not_improved`, best candidate `candidate_0000`, no
  errors, and no official SOTA claim.
- Full Ruff: passed.
- Targeted pytest: passed (`5 passed`).
- Full pytest on the login node failed with `MemoryError` after reaching about
  90% of the test suite. It was rerun through Slurm job `3374101` with 64 CPUs
  and 256G; Slurm pytest passed (`80 passed, 4 warnings`) in 72.29 seconds.
- `git diff --check`: passed.

## Slurm Jobs

- Anchor attempt `3370813`: submitted with teaching account/partition/qos,
  64 CPUs, 256G, 4h. It remained pending past the immediate-use checkpoint
  and was cancelled; Slurm later reported `CANCELLED+`. No result used.
- Anchor retry `3370868`: submitted with teaching account/partition/qos,
  32 CPUs, 128G, 4h. Started immediately and reproduced the anchor metrics,
  but failed in report writing because the anchor-only ablation report lacked a
  `family` column. This exposed a code bug; no result treated as final.
- Anchor clean rerun `3370887`: submitted with teaching account/partition/qos,
  32 CPUs, 128G, 4h. Completed successfully in 00:01:43. `candidate_0000`
  reproduced the previous D3_EyeBench_Lite metrics within tolerance:
  `unseen_reader` BA 0.727404/AUROC 0.808507, `unseen_text` BA
  0.734075/AUROC 0.831851, `unseen_reader_and_text` BA 0.634159/AUROC
  0.715385.
- Full search `3370905`: submitted with teaching account/partition/qos,
  64 CPUs, 256G, 4h. Started immediately on `u24-chiivm0-606`. Preflight
  recorded 64 CPUs, 256G requested memory, no GPU device expected for the CPU
  job, editable install, and environment validation. As of 2026-05-22 12:51,
  the job was still running and had completed 11 legal D3-family candidates;
  the best current inner-validation internal simple-mean BA was 0.798814 from
  `candidate_0011_c51e744a96`.
- Full search `3370905` final: completed successfully in 03:34:23, exit code
  `0:0`, MaxRSS `6051332K`, AveCPU `19:31:46`. It evaluated `candidate_0000`
  plus 23 new D3-family candidates, Test-evaluated the top 6 new candidates,
  and selected `candidate_0000` as final best because no locked candidate
  improved over the anchor under internal simple-mean Test BA.
- Safe acceleration attempt `3373296`: failed immediately before modeling
  because a generated staging config path was missing. No result used.
- Safe acceleration attempt `3373541`: ran in a separate staging output path
  with recipe-level inner feature caching. It was cancelled after the original
  validated, to avoid spending cluster time on duplicate results. No result
  used.
- Safe acceleration attempt `3374057`: ran in a separate staging output path
  with recipe-level inner and Test feature caching. It was cancelled after the
  original validated, to avoid spending cluster time on duplicate results. No
  result used.
- Full pytest Slurm job `3374101`: completed successfully in 00:01:18, exit
  code `0:0`, MaxRSS `2775876K`; full suite passed (`80 passed, 4 warnings`).

## Result Summary

- `candidate_0000` reproduced the previous D3_EyeBench_Lite anchor within
  tolerance:
  - `unseen_reader` BA `0.727404`, AUROC `0.808507`
  - `unseen_text` BA `0.734075`, AUROC `0.831851`
  - `unseen_reader_and_text` BA `0.634159`, AUROC `0.715385`
- Primary metric: per-regime official-compatible trial-level BA/AUROC plus
  internal simple means; visible official leaderboard average was not used as
  the primary metric.
- Final best: `candidate_0000`
- Final internal simple-mean BA/AUROC: `0.698546` / `0.785248`
- Best inner-validation candidate: `candidate_0014_3a7538097b`, a
  D3_Lite-plus fuller official-feature logistic candidate, inner BA `0.800737`;
  Test internal simple-mean BA/AUROC `0.673367` / `0.742044`, below
  `candidate_0000`.
- Best Test-evaluated new candidate by internal simple-mean BA:
  `candidate_0013_936f0c9788`, BA `0.677885`, AUROC `0.744333`, below
  `candidate_0000`.
- Decision category: `d3_method_not_improved`.
- Official SOTA claimed: false.

## Commit / Push

- Commit `d173dfd`: `feat: optimize D3 EyeBench own-method score evidence`
- Push status: pushed to `origin/codex/d3-eyebench-own-method-score-max-v2`
- Follow-up log-only commit is expected to record this final push status without
  rewriting the already-pushed modeling commit.
