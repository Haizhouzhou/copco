# Strengthen CPU/GPU Resource Policy

## Prompt / Request Summary

Rewrite `AGENTS.md` into a strong operational policy for `copco`, with explicit CopCo
environment usage, missing-package behavior, strongest-immediate V100 GPU policy,
strongest-immediate CPU/memory policy for CPU-only heavy work, Slurm preflight/post-run
checks, long-job triage, safe acceleration, AI-run logging, and git safety. Update the
decision record, update the AI run log index, validate, commit, and push.

## Plan

1. Inspect repository state and current agent policy.
2. Check for an existing CopCo environment and use it for validation when available.
3. Install missing dev validation packages inside the CopCo environment if needed.
4. Rewrite `AGENTS.md` and update decision/log files.
5. Run lightweight validation through the CopCo environment.
6. Stage only explicit safe paths, run git safety checks, commit, and push.

## Files Inspected

- `AGENTS.md`
- `docs/decisions.md`
- `logs/ai_runs/INDEX.md`
- local activation paths for `start_copco.sh`, `copco_env`, and `.venv`
- conda environment list
- repo run/Slurm script discovery results

## Files Modified

- `AGENTS.md`
- `docs/decisions.md`
- `logs/ai_runs/INDEX.md`
- `logs/ai_runs/2026-05-02_0346_strengthen_cpu_gpu_resource_policy.md`

## Environment / Dependency Changes

- Active environment: `/home/haizhe/conda/envs/copco/bin/python`
- `pytest` and `ruff` were initially missing from the CopCo environment.
- Install command: `python -m pip install -e ".[dev]"` inside the `copco` conda
  environment.
- Dependency file changed: none. Existing `pyproject.toml` already declared the dev
  dependencies.

## Commands Run

- `pwd`
- `git status --short`
- `python --version`
- `git branch --show-current`
- `git remote -v`
- `sed -n '1,340p' AGENTS.md`
- `date +%Y-%m-%d_%H%M`
- activation-path checks for `start_copco.sh`, `copco_env`, `.venv`, `venv`, and `env`
- `command -v conda`
- `command -v micromamba`
- `command -v mamba`
- `conda env list`
- repo run/Slurm script discovery with `find`
- `sed -n '1,220p' docs/decisions.md`
- `sed -n '1,160p' logs/ai_runs/INDEX.md`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate copco && which python && python --version && python -c "import sys; print(sys.executable)"`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate copco && python -c "import pytest; print(pytest.__version__)"`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate copco && python -c "import ruff; print('ruff import ok')"`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate copco && python -m ruff --version`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate copco && python -m pip install -e ".[dev]"`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate copco && which python && python --version && python -c "import sys; print(sys.executable)" && python scripts/validate_env.py`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate copco && python -m pytest tests/ -q`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate copco && python -m ruff check .`
- `git diff --check`
- `git status --short`
- `git diff --stat`
- targeted `rg` content check for required and forbidden policy terms
- `sed -n '1,520p' AGENTS.md`
- `git add AGENTS.md docs/decisions.md logs/ai_runs/INDEX.md logs/ai_runs/2026-05-02_0346_strengthen_cpu_gpu_resource_policy.md`
- `git diff --stat --cached`
- `git diff --cached --check`
- staged-file size check using `git cat-file -s`

## Validation Results

- CopCo environment activation: passed, using `/home/haizhe/conda/envs/copco/bin/python`
  with Python 3.11.15.
- `python scripts/validate_env.py`: passed inside the CopCo environment.
- `python -m pytest tests/ -q`: passed, 1 test.
- `python -m ruff check .`: passed.
- `git diff --check`: passed.
- Targeted policy content check: `AGENTS.md` contains the required CopCo environment,
  missing-package, V100, strongest-immediate GPU/CPU, `--immediate=120`, job-array,
  heavy-run, long-job triage, AI logging, and git safety language. No old STN-specific
  terms or phase language were found in `AGENTS.md`.
- Pre-commit staged diff check: passed.
- Staged large-file check: passed; staged files are small text/markdown files.

## Commit / Push Status

Pending. This run log will be committed with the task changes. The final assistant
response records the exact commit hash and push result because embedding the final hash
in this file would change the hash.

## Final Response Summary

Will report policy changes, modified files, validation outcomes, commit hash, push
status, and unresolved issues.
