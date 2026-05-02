# Add Tested GPU Profile

## Prompt / Request Summary

Correct `AGENTS.md` so it directly includes the user-tested high-power UZH Slurm
8-GPU `teaching` profile, requires it as the first strong GPU profile for large
GPU-capable jobs, and clarifies downgrade logging plus CPU-only strongest-immediate
behavior.

## Plan

1. Inspect repo state and current Slurm policy sections.
2. Confirm CopCo environment activation.
3. Patch `AGENTS.md` with the tested 8-GPU profile and discovery bullets.
4. Update the AI run log index.
5. Validate through the CopCo environment.
6. Stage explicit safe paths, run git safety checks, commit, and push.

## Files Inspected

- `AGENTS.md`
- `logs/ai_runs/INDEX.md`

## Files Modified

- `AGENTS.md`
- `logs/ai_runs/INDEX.md`
- `logs/ai_runs/2026-05-02_0400_add_tested_gpu_profile.md`

## Commands Run

- `pwd`
- `git status --short`
- `python --version`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate copco && which python && python --version && python -c "import sys; print(sys.executable)"`
- `sed -n '90,190p' AGENTS.md`
- `date +%Y-%m-%d_%H%M`
- `sed -n '90,210p' AGENTS.md`
- `rg -n "Known Tested High-power GPU Profile|teaching|mlnlp2|--gres=gpu:8|--cpus-per-task=32|--mem=128G|--time=04:00:00|--immediate=120|not merely an example|Do not leave an interactive|CPU-only heavy jobs|claim_best_immediate_resource.sh --mode cpu|balanced|A100|H100|V100" AGENTS.md`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate copco && python scripts/validate_env.py`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate copco && python -m pytest tests/ -q`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate copco && python -m ruff check .`
- `git diff --check`
- `git status --short`
- `git diff --stat`
- `git diff -- AGENTS.md logs/ai_runs/INDEX.md logs/ai_runs/2026-05-02_0400_add_tested_gpu_profile.md`
- `git add AGENTS.md logs/ai_runs/INDEX.md logs/ai_runs/2026-05-02_0400_add_tested_gpu_profile.md`
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
- Targeted content check: `AGENTS.md` contains the tested `teaching` 8-GPU profile,
  `--immediate=120`, downgrade logging requirements, and CPU-only strongest-immediate
  resource rules.
- Pre-commit staged diff check: passed.
- Staged large-file check: passed; staged files are small text/markdown files.

## Commit / Push Status

Pending. This run log is committed with the task changes. The final assistant response
records the exact commit hash and push result because embedding the final hash in this
file would change the hash.

## Final Response Summary

Will report the tested profile policy correction, files changed, validation outcomes,
commit hash, push status, and unresolved issues.
