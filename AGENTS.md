# AGENTS.md - copco Working Rules

## Repository Identity

- Project name: `copco`.
- This is a CopCo / Eye Bench research-code repository for Danish natural-reading
  eye-tracking analysis and benchmarking.
- This GitHub repo is a research checkpoint system.
- Do not use it as a dumping ground for raw data, copied datasets, large derived
  artifacts, model checkpoints, or ad hoc local outputs.

## Scientific Claim Boundaries

- Do not claim CopCo itself is a dyslexia dataset.
- Do not claim clinical diagnosis, screening, clinical validation, or medical utility
  unless explicitly supported.
- Treat labels as operational research labels unless stronger documentation exists.
- Do not invent benchmark results or scientific conclusions.

## Data Policy

- Do not modify original/source data in place.
- Do not commit raw data, copied datasets, downloaded archives, participant-level
  derived tables, large artifacts, model checkpoints, caches, or local environments.
- Keep ignored by default:
  - `data/`
  - `raw/`
  - `external/`
  - `extracted/`
  - `derived/`
  - `results/`
  - local environment directories such as `.venv/`, `venv/`, and `env/`
- Write transforms to new derived/output locations, never over source files.
- Derived outputs need provenance when feasible: input reference, script, command,
  config, timestamp, and code version.
- Do not log subject-identifiable data, secrets, raw data dumps, or massive stdout.

## Environment Boot Rule

Start every non-trivial task with lightweight state checks:

```bash
pwd
git status --short
python --version
```

Then identify the active environment before running project code:

- Prefer the project environment if one exists and is documented.
- If no project environment exists, use the current Python only for lightweight
  validation and report missing dependencies instead of silently changing environments.
- Do not create, delete, or rewrite local environments unless the task requires it.
- Never commit local environment directories.
- For scaffold-level validation, use:

```bash
python scripts/validate_env.py
python -m pytest tests/ -q
```

## Tested Resource Profiles

Before launching any heavy work, learn the tested resource profiles available in the
local environment. Inspect, when available:

- `~/.codex/AGENTS.md`
- `~/bin/claim_best_immediate_resource.sh`
- repo scripts under `scripts/`
- Slurm submission scripts under the repo
- recent successful Slurm logs under `logs/`
- previous AI run logs under `logs/ai_runs/`
- user-provided tested commands, resource numbers, or launch notes

Rules:

- Use tested resource profiles over generic `balanced` guesses.
- If tested CPU/GPU/memory/time/partition/account numbers exist, preserve them exactly
  in scripts or docs unless the user asks for a change.
- If no tested numbers are found, do not invent exact cluster numbers. Inspect the
  policy/scripts and report that tested defaults are unavailable.
- For heavy work, prefer the best immediate appropriate resource profile, not the
  weakest acceptable one.
- The cluster is intended to be used for heavy analysis; do not waste user time with
  powerless local fallbacks.
- Still avoid obviously wasteful duplicate jobs, race conditions, and resources
  unrelated to the workload.
- Never silently downgrade from a requested or tested resource profile to a weaker one.
- If a strong tested profile cannot start immediately, log the failed attempt and then
  try the next tested immediate candidate. Do not submit waiting jobs by default.

## Slurm / Compute Policy

Inherit the global UZH Slurm policy from `~/.codex/AGENTS.md` if present. For heavy
CPU/GPU work, use the local immediate-resource launcher unless the user explicitly
requests another strategy:

```bash
~/bin/claim_best_immediate_resource.sh --mode cpu "cd /path/to/repo && <command>"
~/bin/claim_best_immediate_resource.sh --mode gpu "cd /path/to/repo && <command>"
```

Core rules:

- No heavy work on a login node.
- Do not run heavy extraction, batch feature generation, large joins, large text
  processing, embedding generation, large NLP/LLM inference, model training,
  hyperparameter sweeps, bootstrap evaluation, or cross-validation on a login node.
- Use Slurm job arrays for large independent workloads.
- Do not request GPU for CPU-only work.
- Do not fall back to CPU for GPU tasks.
- Do not silently weaken GPU or heavy-resource requests.
- Do not submit waiting jobs by default; use immediate/no-wait allocation behavior.
- If `claim_best_immediate_resource.sh` fails because of invalid account, partition,
  QoS, GRES, or permissions, inspect and report the resource-policy issue. Do not
  brute-force indefinitely.
- For allocated CPU jobs, verify hostname, job id, CPU count, memory, wall time, and
  environment before the workload starts.
- For allocated GPU jobs, verify hostname, job id, CUDA visibility, GPU count, GPU
  names, GPU memory, CPU count, memory, and PyTorch CUDA visibility before the workload
  starts.
- Abort GPU tasks if CUDA/GPU visibility is wrong, PyTorch sees zero GPUs, or fewer
  GPUs are visible than requested.
- If a CUDA-broken node is found, record it in the run log and retry with an explicit
  exclusion when possible.
- Multi-GPU allocation is not enough: the code must actually use the visible GPUs, for
  example through worker-per-GPU inference, tensor parallelism, distributed execution,
  or another tested parallel strategy.

## Heavy Compute / Long Runtime Policy

Before launching a computationally expensive task, estimate the workload size and avoid
large serial runs by default. Estimate, as relevant:

- number of files, participants, texts, tokens, windows, rows, or records
- number of grid combinations, folds, seeds, candidate batches, or bootstrap samples
- expected runtime and memory from a small representative sample
- expected output count, size, and merge behavior
- whether the workload is CPU-bound, memory-bound, I/O-bound, GPU-bound, or blocked by
  uncached recomputation

Scaling rules:

- If a full run is expected to take more than 30 minutes, first determine whether it
  can be split across files, participants, texts, folds, seeds, grid chunks, candidate
  batches, or time windows.
- Prefer parallel shards over a single serial loop when units are independent.
- On Slurm/HPC systems, prefer job arrays when work can be split cleanly.
- Do not assume that a Slurm allocation makes code parallel. Requesting many CPUs only
  helps when the code uses multiprocessing, threading, vectorized libraries, job arrays,
  or another tested parallel method.
- Each shard must write to a separate output file or directory.
- Sharded programs must support resume/restart: skip completed shards, preserve partial
  results, and merge outputs only after all required shards finish.
- Cache and reuse expensive extraction, tokenization, feature, embedding, and window
  computations when possible.
- Before scaling up, run a small test shard to verify correctness, runtime estimate,
  memory use, output format, logging, and merge behavior.
- Do not overwrite completed results. Use a new timestamped or configured output path.
- Do not launch duplicate jobs that write to the same output path as an active job.
- Do not kill running productive jobs without explicit user authorization.
- When a job is slow or appears stalled, distinguish Slurm queue wait time, reconnect
  or session issues, actual compute time, single-core bottlenecks, GPU underuse,
  memory pressure, I/O bottlenecks, repeated uncached computation, and excessive grid
  size.

Serial pattern to avoid for large workloads:

```python
for file in files:
    for grid_config in grid:
        for candidate_batch in batches:
            compute_features_or_windows()
            evaluate()
            save_result()
```

Preferred parallel pattern:

```python
shard_id = slurm_array_task_id
assigned_work = split(files, grid, batches)[shard_id]

for item in assigned_work:
    load_or_compute_cached_features_or_windows()
    evaluate()
    write_result_to_shard_specific_output()

# After all shards finish:
merge_shard_outputs()
```

## Output / Reproducibility Policy

- Results go under `results/`.
- Logs go under `logs/`.
- New nontrivial outputs must have configs or manifests.
- Do not overwrite completed results.
- Decision records go into `docs/decisions.md` or `docs/adr/`.
- Record command, input reference, script, config, environment, resource profile,
  timestamp, code version, and validation for nontrivial generated outputs.
- Benchmark artifacts must document scoring definitions, split definitions, leakage
  controls, and claim language before results are treated as publishable.

## AI Run Logging Policy

For every non-trivial Codex task:

- Create `logs/ai_runs/YYYY-MM-DD_HHMM_<slug>.md`.
- Update `logs/ai_runs/INDEX.md`.
- Log prompt summary, plan, files inspected, files modified, commands run, validation,
  final response summary, and commit/push status.
- For heavy jobs, log selected resource profile, failed stronger attempts, job id,
  output path, progress evidence, bottleneck hypothesis, validation plan, and whether
  any original job was left running.
- Do not log massive stdout, raw data, subject-identifiable data, secrets, ignored file
  contents, or credentials.

## Git Commit / Push Policy

- At the end of every completed non-trivial task, Codex should commit and push unless
  the user explicitly says not to.
- Never use broad staging commands:

```bash
git add .
git add -A
git add *
```

- Stage only explicit safe paths.
- Never stage `data/`, `raw/`, `external/`, `extracted/`, `derived/`, `results/`,
  local environments, model checkpoints, large scientific artifacts, local configs,
  caches, secrets, or files greater than or equal to 100 MB.
- Before commit/push run:

```bash
git status --short
git diff --stat
git diff --check
```

- Also run a large-file check over staged/tracked files.
- If the working tree contains unrelated local files, leave them unstaged.
- If push fails, record the exact failure and do not retry blindly.

## Lightweight Test Commands

Use these for documentation and scaffold-level changes:

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
