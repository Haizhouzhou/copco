# Continuation Audit

- Current branch: `codex/official-eyebench-baseline-evaluator-closure-v1`
- Current commit: `22bf63e6db6e93092345127f1365c6c21c2c939b`
- EyeBench submodule status: `ce87f38a3083aeed029c255716a1a51e6ae51167 eyebench (heads/main)`
- EyeBench commit: `ce87f38a3083aeed029c255716a1a51e6ae51167`
- EyeBench clean: True
- Active Slurm job: `3366776`
- Closure config: `configs/official_eyebench_baseline_evaluator_closure_v1.yaml`
- Official processed data: `/home/haizhe/copco/eyebench/data/CopCo/processed`
- Official folds: `/home/haizhe/copco/eyebench/data/CopCo/folds_metadata`
- Config mapping bug/fix: global_processed_dir was missing from the closure config and has been added.

## Existing Closure Output Directories
- `results/official_eyebench_baseline_evaluator_closure_v1_20260522_012025`
- `results/official_eyebench_baseline_evaluator_closure_v1_20260522_012045`
- `results/official_eyebench_baseline_evaluator_closure_v1_20260522_014350`
- `results/official_eyebench_baseline_evaluator_closure_v1_20260522_014848`
- `results/official_eyebench_baseline_evaluator_closure_v1_20260522_015211`
- `results/official_eyebench_baseline_evaluator_closure_v1_sbatch`

## Existing Closure Reports
- `analysis/official_eyebench_baseline_evaluator_closure_v1/continuation_audit.md`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/d3_reuse_or_rerun_report.md`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/d3_reuse_validation_report.md`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/data_fold_revalidation_report.md`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/evaluator_without_wandb_report.md`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/local_official_baseline_inventory.md`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/local_official_baseline_vs_d3_report.md`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/local_official_logistic_baseline_report.md`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/official_baseline_command_source_report.md`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/official_baseline_reproduction_report.md`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/official_baseline_vs_d3_comparison_report.md`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/official_command_source_inventory.md`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/official_evaluator_closure_report.md`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/official_sota_decision.json`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/official_sota_decision_report.md`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/preflight_report.md`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/runtime_import_repair_report.md`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/tables/official_baseline_vs_d3_detail.csv`
- `analysis/official_eyebench_baseline_evaluator_closure_v1/wandb_bypass_policy.md`

## Minimal py312 Package Probe
```text
{'wandb': '0.23.1', 'lightning': '2.5.1', 'lightning_fabric': '2.6.4', 'pytorch_metric_learning': '2.9.0', 'transformers': '4.47.1', 'seaborn': '0.13.2', 'packaging': '24.2'}
```

## Git Status
```text
M .gitignore
 M logs/ai_runs/INDEX.md
 M paper/submission_v1/supplement_sections/18_benchmark_bridge.tex
 M pyproject.toml
 M src/copco_eye_bench/cli.py
?? analysis/official_eyebench_baseline_evaluator_closure_v1/
?? configs/official_eyebench_baseline_evaluator_closure_v1.yaml
?? docs/eyebench_baseline_evaluator_closure_v1.md
?? logs/ai_runs/2026-05-22_0117_official_eyebench_baseline_evaluator_closure_v1.md
?? scripts/slurm/
?? src/copco_eye_bench/official_eyebench_baseline_evaluator_closure.py
?? tests/test_official_eyebench_baseline_evaluator_closure.py
```
