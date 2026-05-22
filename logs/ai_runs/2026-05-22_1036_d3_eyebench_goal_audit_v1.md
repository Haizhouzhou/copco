# D3 EyeBench Goal Audit v1

## Request Summary

Audit the completed D3 EyeBench protocol-aligned optimization campaign from
commit `270f86d` on a new branch
`codex/d3-eyebench-goal-audit-v1`. Do not run a new optimization, candidate
search, model tuning, score change, or manuscript claim update.

## Plan

1. Start from `270f86d` and create the audit branch.
2. Inventory protocol, campaign, D3 Lite, Logistic anchor, official target,
   validator, and Slurm artifacts.
3. Audit anchor inclusion, metrics, per-regime results, algorithm fidelity,
   leaderboard-directedness, and root causes.
4. Write small audit reports and decision JSON under
   `analysis/d3_eyebench_goal_audit_v1/`.
5. Run requested validation and safe Git checks.
6. Commit and push the audit branch.

## Files Inspected

- `docs/goals/d3_eyebench_protocol_aligned_optimization_v1.md`
- `configs/d3_eyebench_protocol_aligned_optimization_v1.yaml`
- `configs/d3_eyebench_protocol_aligned_optimization_v1_accelerated.yaml`
- `src/copco_eye_bench/d3_eyebench_protocol_optimization.py`
- `src/copco_eye_bench/official_eyebench_sota_check.py`
- `src/copco_eye_bench/official_eyebench_runtime_fix.py`
- `src/copco_eye_bench/official_eyebench_baseline_evaluator_closure.py`
- `src/copco_eye_bench/research_exploration.py`
- `src/copco_eye_bench/benchmark_bridge.py`
- `analysis/d3_eyebench_protocol_aligned_optimization_v1/*`
- `analysis/official_eyebench_runtime_fix_v1/*`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/*`
- `results/d3_eyebench_protocol_aligned_optimization_v1_20260522_074957/*`
- `results/d3_eyebench_protocol_aligned_optimization_v1_accelerated_20260522_080058/*`
- `results/official_eyebench_runtime_fix_v1_20260522_0005/*`
- `results/official_eyebench_baseline_evaluator_closure_v1_20260522_015211/*`
- `eyebench/results/formatted_eyebench_benchmark_results/CopCo_TYP_test.csv`

## Findings

- Previous `D3_EyeBench_Lite` was not included in the /goal candidate set.
- No `candidate_0000` or D3 Lite anchor candidate exists in candidate specs.
- The optimizer initialized `best_score = -1.0`, not from previous D3 Lite.
- /goal selected candidate average BA `0.679915506990474` can be lower than
  previous D3 Lite by design.
- /goal BA threshold policy differs from previous D3 Lite and local Logistic
  anchor.
- /goal final average is a simple mean of regime BAs; the visible official
  leaderboard average was not reproduced.
- The selected candidate is a reduced residual-feature wrapper, not full D3 and
  not exact D3 Lite.

## Files Modified

- `analysis/d3_eyebench_goal_audit_v1/artifact_inventory.md`
- `analysis/d3_eyebench_goal_audit_v1/anchor_candidate_audit.md`
- `analysis/d3_eyebench_goal_audit_v1/metric_consistency_audit.md`
- `analysis/d3_eyebench_goal_audit_v1/candidate_per_regime_audit.md`
- `analysis/d3_eyebench_goal_audit_v1/algorithm_fidelity_audit.md`
- `analysis/d3_eyebench_goal_audit_v1/leaderboard_directedness_audit.md`
- `analysis/d3_eyebench_goal_audit_v1/root_cause_report.md`
- `analysis/d3_eyebench_goal_audit_v1/goal_audit_decision.json`
- `analysis/d3_eyebench_goal_audit_v1/goal_audit_decision_report.md`
- `logs/ai_runs/2026-05-22_1036_d3_eyebench_goal_audit_v1.md`
- `logs/ai_runs/INDEX.md`

## Commands Run

- `pwd`
- `git status --short --branch --untracked-files=all`
- `python --version`
- `git switch -c codex/d3-eyebench-goal-audit-v1 270f86d`
- `conda run -n copco which python`
- `conda run -n copco python --version`
- `conda run -n copco python -c "import sys; print(sys.executable)"`
- `rg --files ...`
- `find results ...`
- `find logs ...`
- `rg -n ...`
- `sed -n ...`
- `python - <<'PY' ...` read-only CSV/JSON summaries
- `sacct -j 3368826,3368833 --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW`
- `conda run -n copco python -m pip install -e .`
- `conda run -n copco python scripts/validate_env.py`
- `conda run -n copco python -m ruff check .`
- `conda run -n copco python -m pytest tests/test_d3_eyebench_protocol_aligned_optimization.py -q || true`
- `conda run -n copco python -m pytest tests/test_d3_eyebench_protocol_optimization.py -q`
- `git diff --check`
- `git status --short`
- `gh --version`
- `gh auth status`

## Validation Results

- Editable install in `copco`: passed.
- `python scripts/validate_env.py`: passed.
- `python -m ruff check .`: passed.
- Requested pytest path
  `tests/test_d3_eyebench_protocol_aligned_optimization.py`: file not found;
  command was run with `|| true` exactly as requested and did not block commit.
- Existing adjacent D3 optimization tests:
  `tests/test_d3_eyebench_protocol_optimization.py`: 4 passed.
- `git diff --check`: passed.
- `git status --short`: only audit reports and AI log/index changes present.
- `gh --version` and `gh auth status`: `gh` is not installed. This does not
  block the requested Git push because no PR creation was requested.

## Commit / Push

Pending at report creation time.
