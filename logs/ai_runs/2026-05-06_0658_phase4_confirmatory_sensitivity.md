# Phase 4 Confirmatory Sensitivity Run

## Request Summary

Implement and run Phase 4 confirmatory analysis for DFM predictability sensitivity and
residualized gaze-cost profiles, using Label Release v1.1 and Phase 3 outputs without
creating new core labels or rebuilding prior releases.

## Plan

1. Inspect Phase 3 conventions and prepared data schema.
2. Add Phase 4 config, CLI entrypoints, reports, validation, and docs.
3. Add tests for cross-fitting, leakage guards, feature grouping, prediction schema, and reports.
4. Run environment, tests, lint, label-release validation, Phase 4 run, Phase 4 validation, and git checks.
5. Commit and push safe code/config/docs/small reports only.

## Files Inspected

- `pyproject.toml`
- `src/copco_eye_bench/cli.py`
- `src/copco_eye_bench/research_exploration.py`
- `configs/research_exploration_v1.yaml`
- `tests/test_research_exploration.py`
- Label Release v1.1 prepared tables
- Phase 3 participant sensitivity profiles

## Files Modified

- `configs/phase4_confirmatory_sensitivity_v1.yaml`
- `src/copco_eye_bench/phase4_confirmatory.py`
- `src/copco_eye_bench/cli.py`
- `pyproject.toml`
- `tests/test_phase4_confirmatory.py`
- `docs/phase4_confirmatory_analysis_plan.md`
- `analysis/phase4_confirmatory/*.md`
- `analysis/phase4_confirmatory/*.csv`
- `logs/ai_runs/INDEX.md`
- `logs/ai_runs/2026-05-06_0658_phase4_confirmatory_sensitivity.md`

## Commands Run

- `pwd && git status --short && python --version`
- `git rev-parse --show-toplevel`
- `rg --files ...`
- `conda run -n copco python --version`
- Prepared schema inspection with `conda run -n copco python -c ...`
- `conda run -n copco python -m pip install -e .`
- `conda run -n copco python scripts/validate_env.py`
- `conda run -n copco python -m pytest tests/ -q`
- `conda run -n copco python -m ruff check .`
- `conda run -n copco copco-validate-label-release --config configs/label_release_v1_1.yaml --output-dir results/label_release_v1_1_20260506_0041`
- `~/bin/claim_best_immediate_resource.sh --mode cpu ... copco-run-phase4-confirmatory ... --output-dir results/phase4_confirmatory_sensitivity_v1_20260506_0715`
- `conda run -n copco copco-validate-phase4-confirmatory --config configs/phase4_confirmatory_sensitivity_v1.yaml --output-dir results/phase4_confirmatory_sensitivity_v1_20260506_0715`
- `git diff --check`
- `sacct -j 2726918 --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW`
- `sacct -j 2727147 --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW`

## Validation Results

- Environment validation passed.
- Full test suite passed: 31 passed, 4 warnings.
- Ruff passed.
- Label Release v1.1 validation passed.
- Phase 4 validation passed.
- `git diff --check` passed.
- Slurm job `2726918` started immediately on `u24-chiivm0-603`, used 32 CPUs/128G,
  failed after 31:17 only at the interaction-report bug after producing heavy partial outputs.
- Slurm job `2727147` started immediately on `u24-chiivm0-603`, reused completed
  partial outputs, completed in 00:00:29 with exit code 0.
- `seff` produced no output in this environment.

## Result Summary

- Output directory: `results/phase4_confirmatory_sensitivity_v1_20260506_0715`
- Best confirmatory LOPO model: `D3_dfm_residual_gaze_only` logistic regression,
  ROC-AUC 0.8947, PR-AUC 0.8641, balanced accuracy 0.8421, macro F1 0.8421,
  Brier 0.1159.
- DFM sensitivity-only: `D2_dfm_sensitivity_only` logistic regression ROC-AUC 0.8892,
  PR-AUC 0.8611.
- DFM exposure-only: `D1_dfm_exposure_only` logistic regression ROC-AUC 0.4238,
  PR-AUC 0.3685.
- No raw speed/global-duration group: `J_all_except_raw_speed` ROC-AUC 0.8380,
  PR-AUC 0.8338.
- No exposure-only variables group: `K_all_except_exposure_variables` ROC-AUC 0.8241,
  PR-AUC 0.8152.
- Permutation p-value: 0.000999 from 1,000 valid permutations.
- Bootstrap CIs: ROC-AUC [0.7765, 0.9841], PR-AUC [0.7083, 0.9728].
- Leave-one-dyslexia-labeled minimum ROC-AUC: 0.8801.
- Mixed interactions: 8 of 15 controlled focus interaction terms survived; skip model
  failed with singular matrix, continuous/fixture-count cluster-robust fallbacks completed.
- Segmentation decision: `secondary_result`; drop standalone segmentation main-effect framing.

## Final Response Summary

_Pending final response._

## Commit / Push Status

_Pending commit/push._

## Dependency / Environment Update

- Active environment: `/home/haizhe/conda/envs/copco`.
- Install command: `conda run -n copco python -m pip install -e .`.
- Reason: refresh editable console-script entrypoints after adding
  `copco-run-phase4-confirmatory` and `copco-validate-phase4-confirmatory`.
- Dependency files changed: `pyproject.toml` script entrypoints only; no new packages added.
