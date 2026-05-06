# Final Manuscript Audit v1

## Request Summary

Audit and harden the frozen SubmissionSprint v1 manuscript package without changing
the scientific result, selected model, feature families, labels, or claims.

## Plan

1. Inspect the existing submission package, CLI, config, paper source, and analysis
   mirrors.
2. Add a Final Manuscript Audit v1 config, CLI, validator, and tests.
3. Revise manuscript sections for one clear DFM residual gaze-profile story.
4. Generate audit reports, final abstract, contribution list, reviewer plan, and
   readiness report.
5. Attempt LaTeX compilation or write a compile-skipped/source-validation report.
6. Validate the audit and submission package, run tests and Ruff, then commit and push.

## Files Inspected

- `pyproject.toml`
- `src/copco_eye_bench/cli.py`
- `src/copco_eye_bench/submission.py`
- `tests/test_submission.py`
- `configs/submission_v1.yaml`
- `paper/submission_v1/`
- `analysis/submission_v1/`
- `results/submission_v1_20260506_0936/manifest.json`

## Files Modified

- `configs/final_manuscript_audit_v1.yaml`
- `src/copco_eye_bench/manuscript_audit.py`
- `src/copco_eye_bench/cli.py`
- `pyproject.toml`
- `tests/test_manuscript_audit.py`
- `paper/submission_v1/sections/*.tex`
- `paper/submission_v1/tables/final_model_metrics.tex`
- `paper/submission_v1/tables/dfm_exposure_vs_sensitivity.tex`
- `analysis/submission_v1/manuscript/*.md`
- `analysis/submission_v1/tables/final_model_metrics.tex`
- `analysis/submission_v1/tables/dfm_exposure_vs_sensitivity.tex`
- `analysis/final_manuscript_audit_v1/*.md`
- `logs/ai_runs/INDEX.md`
- `logs/ai_runs/2026-05-06_1438_final_manuscript_audit_v1.md`

## Commands Run

- `pwd && git status --short && python --version`
- `conda run -n copco python -m pytest tests/test_manuscript_audit.py -q`
- `conda run -n copco python -m ruff check src/copco_eye_bench/manuscript_audit.py tests/test_manuscript_audit.py`
- `conda run -n copco python -m pip install -e .`
- `conda run -n copco copco-run-manuscript-audit --config configs/final_manuscript_audit_v1.yaml --output-dir results/final_manuscript_audit_v1_20260506_1438`
- `conda run -n copco copco-run-manuscript-audit --config configs/final_manuscript_audit_v1.yaml --output-dir results/final_manuscript_audit_v1_20260506_1438 --allow-existing-output`
- `conda run -n copco copco-validate-manuscript-audit --config configs/final_manuscript_audit_v1.yaml --output-dir results/final_manuscript_audit_v1_20260506_1438`
- `conda run -n copco copco-validate-submission-package --config configs/submission_v1.yaml --output-dir results/submission_v1_20260506_0936`
- `conda run -n copco python scripts/validate_env.py`
- `conda run -n copco python -m pytest tests/ -q`
- `conda run -n copco python -m ruff check .`
- `git diff --check`

## Validation Results

- Environment validation: passed.
- New audit tests: 5 passed.
- Full test suite: 46 passed, 3 warnings.
- Ruff: passed.
- Submission package validation: passed.
- Final manuscript audit validation: passed.
- `git diff --check`: passed.

## LaTeX Status

LaTeX compilation was skipped because `latexmk`, `pdflatex`, and `xelatex` were not
available. Source structure validation was written to
`analysis/final_manuscript_audit_v1/compile_skipped_report.md`.

## Output Path

- `results/final_manuscript_audit_v1_20260506_1438`

## Final Response Summary

Final response should report the revised manuscript package, audit output directory,
LaTeX skip status, green validation checks, remaining manual edits, and final commit
hash.

## Commit / Push Status

Committed and pushed in this run. The exact final commit hash is reported in the final
response because amending this log changes the commit SHA.
