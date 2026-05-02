# copco Agent Instructions

## Repository Identity

This repository is `copco`, a CopCo / Eye Bench research-code repository for Danish
natural-reading eye-tracking analysis and benchmarking.

CopCo is the Copenhagen Corpus of Eye-Tracking Recordings from Natural Reading of
Danish Texts. The first-release paper describes 1,832 sentences, 34,897 tokens, and
eye-tracking data from 22 adult native Danish-speaking participants with normal or
corrected-to-normal vision and no known reading impairments.

New project phases start at Phase 0. Phase 0 must freeze target definitions, scoring,
dataset splits, leakage controls, and claim language before generating benchmark
results.

## Scientific Boundaries

- Do not claim that CopCo itself is a dyslexia dataset.
- Do not claim clinical diagnosis, clinical validation, screening, or medical utility
  unless the project data and documentation explicitly support that claim.
- When discussing later dyslexia-related or reading-difficulty work, use careful
  language such as `dyslexia-labeled`, `reading-difficulty-labeled`, or `operational
  labels`.
- Treat reference labels as operational research labels, not biological or clinical
  ground truth.
- Keep benchmark, audit, and modeling claims limited to what the current data,
  protocol, and validation results support.

## Data Policy

- Do not commit raw data, copied datasets, derived participant-level tables, large
  artifacts, or local environments.
- Keep local source data and external mirrors outside version control. Use documented
  paths and retrieval instructions instead of committing data copies.
- Do not modify research/raw data while inspecting, validating, or probing runtime
  behavior.
- Store reproducible lightweight code, metadata schemas, and documentation in Git.
- Store generated results under `results/` and runtime logs under `logs/`.
- Store decisions under `docs/decisions.md` or `docs/adr/`.

## Environment Boot Rule

Start each non-trivial task by identifying the repository root and checking the local
state with lightweight commands such as:

```bash
pwd
git status --short
python --version
```

Do not assume local data paths, installed dependencies, active virtual environments,
or Slurm availability. Prefer lightweight validation before running any analysis.

## Slurm / Compute Policy

The global `~/.codex/AGENTS.md` Slurm policy should be inherited if present.

For UZH/HPC usage:

- Do not run heavy extraction, training, hyperparameter sweeps, bootstrap evaluation,
  large NLP model inference, or batch feature generation on login nodes.
- Use Slurm and job arrays for large independent workloads.
- Request CPU, memory, wall time, and GPU resources appropriate to the workload.
- Do not request GPU resources for CPU-only work.
- For GPU tasks, verify CUDA visibility, GPU count, GPU names, CPU count, memory, and
  PyTorch CUDA availability before the workload starts.
- Abort GPU tasks if CUDA/GPU visibility is wrong, PyTorch sees zero GPUs, or fewer
  GPUs are visible than requested.
- Do not submit jobs that wait indefinitely unless the user explicitly asks.

## Heavy Compute / Long Runtime Policy

- Make long-running jobs checkpointed, resumable, and restart-safe.
- Do not overwrite active outputs or launch duplicate jobs writing to the same path.
- Use isolated staging paths for accelerated or alternative attempts.
- For stalled or unexpectedly slow jobs, inspect safe status only: scheduler state,
  logs, timestamps, output sizes, and resource utilization when available.
- Estimate progress and remaining runtime from concrete evidence before recommending
  acceleration.
- Do not kill productive jobs without explicit user authorization.

## Output / Reproducibility Policy

- Put generated results under `results/`.
- Put runtime logs under `logs/`.
- Keep source-controlled documentation and decisions under `docs/`.
- Record commands, input references, code version, configuration, random seeds, split
  definitions, scoring definitions, and leakage controls for any benchmark result.
- Benchmark outputs must be reproducible from committed code plus documented external
  data inputs.
- Do not publish or summarize benchmark results before Phase 0 definitions are frozen.

## AI Run Logging Policy

For non-trivial Codex tasks:

- Create `logs/ai_runs/YYYY-MM-DD_HHMM_<slug>.md`.
- Update `logs/ai_runs/INDEX.md`.
- Include task summary, plan, files touched, commands run, validation outcomes, and
  commit/push status.
- Do not include secrets, tokens, private credentials, or massive stdout.

## Git Commit / Push Policy

Codex should commit and push after completed non-trivial tasks unless the user
explicitly says not to.

Never use broad staging commands such as:

```bash
git add .
git add -A
git add *
```

Use explicit safe-path staging only. Before committing, run:

```bash
git status --short
git diff --stat
git diff --check
```

Also check staged/tracked files for large artifacts before committing. If the working
tree contains unrelated local files, ignore or leave them unstaged unless the user
explicitly includes them.

If push fails, report the exact failure and do not retry blindly.

## Lightweight Test Commands

Use these lightweight checks for scaffold and small code changes:

```bash
python scripts/validate_env.py
python -m pytest tests/ -q
git diff --check
git status --short
git diff --stat
```

If Ruff is configured and installed, also run:

```bash
python -m ruff check .
```
