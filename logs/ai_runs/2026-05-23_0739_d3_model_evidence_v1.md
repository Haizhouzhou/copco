# D3ModelEvidenceVault v1 AI Run

## Request Summary

Create a committed D3 evidence vault under `analysis/d3_model_evidence_v1/` that
collects algorithm descriptions, source artifact inventory, canonical metrics, claim
ledger, paper-source manifests, and validation reports. This is documentation and
evidence extraction only; no new models, feature searches, figures, or final paper
tables are generated.

## Plan

1. Inspect D3 source artifacts and keep pre-existing v1/v2 dirty outputs unstaged.
2. Add a reproducible builder and validator CLI for the evidence vault.
3. Generate the committed evidence vault from existing source artifacts.
4. Run validation, tests, formatting checks, commit, and push.

## Commands Run

- `pwd`
- `git status --short --branch`
- `python --version`
- `conda run -n copco python -c "import sys; print(sys.version); print(sys.executable)"`
- Source artifact inventory and metric-header inspection commands.
- `conda run -n copco python -m py_compile src/copco_eye_bench/d3_model_evidence_v1.py src/copco_eye_bench/cli.py`
- `conda run -n copco python -m pytest tests/test_d3_model_evidence_v1.py -q`
- `conda run -n copco python -m pip install -e .`
- `conda run -n copco python scripts/validate_env.py`
- `conda run -n copco copco-build-d3-model-evidence-v1 --output-dir results/d3_model_evidence_v1_20260523_074726`
- `conda run -n copco copco-validate-d3-model-evidence-v1 --output-dir results/d3_model_evidence_v1_20260523_074726`
- `conda run -n copco python -m ruff check .`
- `git diff --check`
- `~/bin/claim_best_immediate_resource.sh --mode cpu --candidate "--partition=teaching ... --cpus-per-task=32 --mem=128G --time=04:00:00" "... python -m pytest tests/ -q"`
- `sacct -j 3375242 --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW`
- `seff 3375242`

## Validation Results

- Evidence vault build completed:
  - analysis folder: `analysis/d3_model_evidence_v1`
  - generated output: `results/d3_model_evidence_v1_20260523_074726`
  - source artifact inventory rows: 42
  - canonical metric rows: 160
  - online prefix rows: 2,477
  - online stopping rows: 2,133
  - claim ledger rows: 20
  - number registry rows: 13
  - missing sources: none
  - figures generated: false
  - final paper tables generated: false
- Evidence vault validator passed with no errors or warnings.
- Focused evidence-vault tests passed: 5 passed.
- Full pytest passed on Slurm job `3375242`: 105 passed, 8 warnings, 69.17s.
- Slurm notes:
  - CPU-only `teaching` candidate started immediately on `u24-chiivm0-603`.
  - Preflight confirmed `SLURM_CPUS_PER_TASK=32`, `SLURM_MEM_PER_NODE=131072`,
    no CUDA devices for CPU-only work, and sufficient available memory.
  - `sacct` reported job `3375242` `COMPLETED`, exit code `0:0`.
  - `seff` was not available on PATH.
- Ruff passed.
- `git diff --check` passed.

## Commit / Push Status

- Primary evidence vault commit: `0d701c9`
  (`docs: add D3 model evidence vault v1`).
- Primary commit pushed to `origin/codex/d3-online-targeted-optimization-v1`.
- Generated output not committed:
  `results/d3_model_evidence_v1_20260523_074726/`.
- Pre-existing unrelated working-tree artifacts intentionally left unstaged:
  `analysis/d3_online_targeted_optimization_v1/subgoal_status.json`,
  `analysis/d3_online_targeted_optimization_v1/subgoal_status.md`,
  `analysis/d3_online_targeted_optimization_v1_fast2/`, and
  `analysis/d3_online_targeted_optimization_v2/error_source_by_prefix.csv`.
