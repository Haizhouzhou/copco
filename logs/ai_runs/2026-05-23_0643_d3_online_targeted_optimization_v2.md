# D3OnlineTargetedOptimizationAuditAndRerun v2 AI Run

## Request Summary

Audit the v1 D3 online targeted optimization outputs and rerun a stricter online
selection layer that separates full-evidence/offline-like rows from late, mid,
early, and stopping-detector online rows.

## Plan

1. Inspect v1 outputs and preserve dirty pre-existing v1 status drift unstaged.
2. Implement v2 config, docs, runner, validator, CLI, and tests.
3. Run v2 against the validated v1 output.
4. Validate v2 outputs, run tests/checks, commit, and push.

## Commands Run

- `pwd`
- `git status --short --branch`
- `python --version`
- `conda run -n copco python --version`
- v1 artifact inspection commands
- `conda run -n copco python -m py_compile src/copco_eye_bench/d3_online_targeted_optimization_v2.py src/copco_eye_bench/cli.py`
- `conda run -n copco python -m pytest tests/test_d3_online_targeted_optimization_v2.py -q`
- `conda run -n copco python -m pip install -e .`
- `conda run -n copco python scripts/validate_env.py`
- `conda run -n copco copco-run-d3-online-targeted-optimization-v2 --config configs/d3_online_targeted_optimization_v2.yaml --output-dir results/d3_online_targeted_optimization_v2_20260523_070506`
- `conda run -n copco copco-validate-d3-online-targeted-optimization-v2 --config configs/d3_online_targeted_optimization_v2.yaml --output-dir results/d3_online_targeted_optimization_v2_20260523_070506`
- `~/bin/claim_best_immediate_resource.sh --mode cpu ... "conda run -n copco python -m pytest tests/ -q"`
- `sacct -j 3375226 --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW`
- `seff 3375226`
- `conda run -n copco python -m ruff check .`
- `git diff --check`
- `git diff --stat`

## Validation Results

- Editable install passed.
- `scripts/validate_env.py` passed in the CopCo conda environment.
- Focused v2 pytest passed: 3 tests.
- V2 runner completed:
  - output dir: `results/d3_online_targeted_optimization_v2_20260523_070506`
  - analysis dir: `analysis/d3_online_targeted_optimization_v2`
  - candidate rows: 52
  - per-prefix curve rows: 1,232
  - error trajectory rows: 7,076
  - locked rows: 23
  - final model rows: 24
- V2 validator passed with no errors.
- Full pytest passed on Slurm job `3375226`: 100 passed, 8 warnings, 71.30s.
- Slurm resource notes:
  - Default `standard` CPU candidates failed immediately because the account/partition
    combination was invalid.
  - Retried immediately on `teaching` with 32 CPUs, 128G, 4h; allocation started on
    `u24-chiivm0-603`.
  - Preflight confirmed `SLURM_CPUS_PER_TASK=32`, `SLURM_MEM_PER_NODE=131072`,
    `CUDA_VISIBLE_DEVICES=` for CPU-only work, and large available system memory.
  - `sacct` reported job `3375226` `COMPLETED` with exit code `0:0`; `seff` was not
    available on PATH.
- Ruff passed.
- `git diff --check` passed.

## Output Summary

- V1 audit: v1 was fast-mode/truncated, evaluated 36 candidates, selected
  `online_d3_0021` with `no_stop`, and is best treated as offline-like late/full
  sequence accumulation rather than a true early detector.
- Earliest reliability gates:
  - unseen_reader: AUROC >= 0.80 at 250 words; BA >= 0.75 at 500 words.
  - unseen_reader_and_text: AUROC >= 0.80 at 500 words; BA >= 0.75 at 500 words.
- Best clean online rows:
  - late: 1000 words, dfm residual plus uncertainty, isotonic, mean probability.
  - mid: 500 words, dfm residual plus uncertainty, inner regime threshold, learned
    meta aggregator.
  - early: 1 text, dfm residual plus uncertainty, inner regime threshold, learned
    meta aggregator.
  - stopping: coverage-constrained stop, mean words 214.3 on unseen_reader but weak BA.
- Unseen-text rescue improved BA in a specialist row to 0.8261 with sigmoid/regime-
  specific calibration, but this is reported as an unseen-text specialist, not the
  general final detector.

## Commit / Push Status

- Primary v2 implementation commit: `8cd47dc` (`feat: audit and rerun strict online D3 targeted optimization`).
- Pushed branch: `codex/d3-online-targeted-optimization-v1`.
- Large/participant-level artifacts intentionally left uncommitted:
  - `results/d3_online_targeted_optimization_v2_20260523_070506/`
  - `analysis/d3_online_targeted_optimization_v2/error_source_by_prefix.csv`
  - pre-existing v1 status drift and `analysis/d3_online_targeted_optimization_v1_fast2/`.
