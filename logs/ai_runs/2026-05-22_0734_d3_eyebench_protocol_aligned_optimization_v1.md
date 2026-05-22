# D3 EyeBench Protocol-Aligned Optimization v1

## Request Summary

Create `docs/goals/d3_eyebench_protocol_aligned_optimization_v1.md` defining a
bounded legal D3 optimization campaign on official EyeBench `CopCo_TYP`.

## Plan

- Check repository state.
- Inspect nearby EyeBench protocol and run-log docs for style and claim language.
- Add the new goal/protocol document under `docs/goals/`.
- Update the AI run log index.
- Run lightweight validation.
- Commit and push only safe text/report files.

## Files Inspected

- `docs/eyebench_baseline_evaluator_closure_v1.md`
- `docs/eyebench_runtime_fix_v1.md`
- `docs/benchmark_bridge_v1_analysis_plan.md`
- `docs/final_scientific_decision_policy_v1.md`
- `logs/ai_runs/INDEX.md`
- `logs/ai_runs/2026-05-22_0117_official_eyebench_baseline_evaluator_closure_v1.md`

## Files Modified

- `docs/goals/d3_eyebench_protocol_aligned_optimization_v1.md`
- `logs/ai_runs/2026-05-22_0734_d3_eyebench_protocol_aligned_optimization_v1.md`
- `logs/ai_runs/INDEX.md`

## Commands Run

- `pwd`
- `git status --short`
- `python --version`
- `git rev-parse --show-toplevel`
- `ls docs`
- `find docs/goals -maxdepth 1 -type f -print | sort | head -50`
- `find logs/ai_runs -maxdepth 1 -type f -print | sort | tail -20`
- `sed -n '1,220p' docs/eyebench_baseline_evaluator_closure_v1.md`
- `sed -n '1,220p' docs/final_scientific_decision_policy_v1.md`
- `sed -n '1,180p' logs/ai_runs/INDEX.md`
- `sed -n '1,180p' logs/ai_runs/2026-05-22_0117_official_eyebench_baseline_evaluator_closure_v1.md`
- `sed -n '1,260p' docs/benchmark_bridge_v1_analysis_plan.md`
- `sed -n '1,260p' docs/eyebench_runtime_fix_v1.md`
- `find docs/protocol -maxdepth 2 -type f -print | sort | head -50`
- `find docs/adr -maxdepth 1 -type f -print | sort | head -50`
- `date +%Y-%m-%d_%H%M`
- `mkdir -p docs/goals`
- `sed -n '1,320p' docs/goals/d3_eyebench_protocol_aligned_optimization_v1.md`
- `rg -n "official EyeBench|official EyeBench folds|no synthetic|random predictions|test-label|participant_id|speech_id|text_id|exposure-count|published leaderboard|local Logistic|trial-level balanced accuracy|AUROC|PR-AUC|macro F1|Brier|reader-aggregated|sbatch|teaching|official_sota_claim_allowed|official_compatible_d3_improved_but_not_sota|blocked_by_evaluator" docs/goals/d3_eyebench_protocol_aligned_optimization_v1.md`
- `git diff --stat`
- `git status --short`
- `git diff --check`
- `wc -c docs/goals/d3_eyebench_protocol_aligned_optimization_v1.md logs/ai_runs/2026-05-22_0734_d3_eyebench_protocol_aligned_optimization_v1.md logs/ai_runs/INDEX.md`
- `git status --short --untracked-files=all`
- `git add docs/goals/d3_eyebench_protocol_aligned_optimization_v1.md logs/ai_runs/2026-05-22_0734_d3_eyebench_protocol_aligned_optimization_v1.md logs/ai_runs/INDEX.md`
- `git status --short`
- `git diff --cached --stat`
- `git diff --cached --check`

## Validation Results

- `git diff --check`: passed.
- `git diff --cached --check`: passed.
- Required protocol terms and gates were verified with `rg`.
- Size check: staged text files total 18,536 bytes; no large data or results
  artifacts were staged.
- No project tests were run because this was a documentation-only change.

## Commit / Push

Pending at log update time.
