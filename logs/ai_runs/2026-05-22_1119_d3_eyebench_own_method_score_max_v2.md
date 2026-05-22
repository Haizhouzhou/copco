# D3 EyeBench Own-Method Score Maximization v2 Goal Spec

## Request Summary

Create `docs/goals/d3_eyebench_own_method_score_maximization_v2.md` as the goal
specification for a new D3-family own-method score-maximization phase. Start
from `codex/d3-eyebench-goal-audit-v1`, create
`codex/d3-eyebench-own-method-score-max-v2`, and do not run the modeling phase
yet.

## Plan

1. Verify repository state and CopCo environment.
2. Create the requested branch.
3. Add the goal specification with anchor preservation, monotonic best-so-far,
   method-fidelity, no-leaderboard-rerun, legal optimization, metrics, stages,
   Slurm, outputs, validation, and final response rules.
4. Run lightweight document validation and Git safety checks.
5. Commit and push the new branch.

## Files Modified

- `docs/goals/d3_eyebench_own_method_score_maximization_v2.md`
- `logs/ai_runs/2026-05-22_1119_d3_eyebench_own_method_score_max_v2.md`
- `logs/ai_runs/INDEX.md`

## Commands Run

- `pwd`
- `git status --short --branch --untracked-files=all`
- `python --version`
- `conda run -n copco which python`
- `conda run -n copco python --version`
- `conda run -n copco python -c "import sys; print(sys.executable)"`
- `git switch -c codex/d3-eyebench-own-method-score-max-v2`
- `git diff --check`
- `conda run -n copco python scripts/validate_env.py`
- `rg -n "candidate_0000|best_so_far|No Leaderboard Reproduction|D3 Lite|official leaderboard average|no synthetic predictions|no random prediction|no test-label tuning|Do not reproduce leaderboard methods|reduced wrapper" docs/goals/d3_eyebench_own_method_score_maximization_v2.md`
- `git diff --stat`
- `git status --short --branch --untracked-files=all`
- `git add docs/goals/d3_eyebench_own_method_score_maximization_v2.md logs/ai_runs/2026-05-22_1119_d3_eyebench_own_method_score_max_v2.md logs/ai_runs/INDEX.md`
- `git diff --cached --stat`
- `git diff --cached --check`
- `git diff --cached --name-only -z | xargs -0 -r du -b | awk '$1 >= 100000000 {print}'`
- `git diff --cached --name-only`

## Validation Results

- `git diff --check`: passed.
- `conda run -n copco python scripts/validate_env.py`: passed with CopCo env
  Python `/home/haizhe/conda/envs/copco/bin/python`, Python `3.11.15`,
  package version `0.0.0`, and `required_structure: ok`.
- Goal-spec content check: confirmed key invariants and prohibitions are
  present, including `candidate_0000`, `best_so_far`, no leaderboard
  reproduction, D3 Lite preservation, no synthetic/random predictions, no
  test-label tuning, and avoiding the official leaderboard-average claim.
- Staged diff check: passed.
- Staged large-file check: no staged file greater than or equal to 100 MB.
- Staged files are limited to the goal spec and AI run logs.

## Commit / Push

- Commit message: `docs: add D3 EyeBench own-method score maximization goal`.
- Push target: `origin codex/d3-eyebench-own-method-score-max-v2`.
- Push status: to be attempted after final amended commit.
