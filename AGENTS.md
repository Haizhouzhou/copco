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
  - local environment directories such as `.venv/`, `venv/`, `env/`, and `copco_env/`
- Write transforms to new derived/output locations, never over source files.
- Derived outputs need provenance when feasible: input reference, script, command,
  config, timestamp, and code version.
- Do not log subject-identifiable data, secrets, raw data dumps, ignored data contents,
  or massive stdout.

## Environment Boot Rule

Start every non-trivial task with lightweight state checks:

```bash
pwd
git status --short
python --version
```

Then:

- Identify the repository root before editing or running project code.
- Normal project work must use the CopCo environment.
- Prefer a repo boot script if present:

```bash
source ./start_copco.sh
```

- If no repo boot script exists, try documented CopCo environment activation in this
  order, using only commands/environments that exist locally:

```bash
conda activate copco
micromamba activate copco
mamba activate copco
source ./copco_env/bin/activate
source ./.venv/bin/activate
```

- After activation, verify:

```bash
which python
python --version
python -c "import sys; print(sys.executable)"
```

- Do not run project code in base Python unless the task is only repository inspection
  or tiny scaffold validation.
- Do not silently switch to another environment to avoid dependency problems.
- If a package is missing, first try to install or fix it inside the CopCo environment.
- Never install packages system-wide.
- Prefer project-declared dependency management when available:
  - if `pyproject.toml` exists, update/use it when the dependency is project-level
  - if `requirements.txt` exists, update/use it when appropriate
  - otherwise use `python -m pip install <package>` inside the active CopCo environment
    for local runtime needs
- After installing or fixing dependencies, rerun the failing command and the
  lightweight validation.
- Log dependency changes in the AI run log.
- Do not create, delete, rewrite, or commit local environments unless the task
  explicitly requires environment maintenance.

## Project-wide UZH Slurm Resource Policy

- This repo inherits `~/.codex/AGENTS.md` if present, but the following rules are active
  directly in this repo.
- The user's time is more expensive than compute.
- For GPU work, CPU-only heavy work, memory-intensive jobs, large data processing,
  dense analysis, embedding generation or post-processing, LLM inference, model
  training, hyperparameter sweeps, bootstrap evaluation, repeated cross-validation, or
  large statistical analysis, request the strongest immediately available valid
  resource configuration first.
- On this UZH Slurm cluster, V100 is the strongest available GPU type. Do not search
  for, wait for, or request unavailable A100, H100, or newer GPUs.

### Known Tested High-power GPU Profile

For large GPU-capable work, the known tested strong profile is:

```bash
srun \
  --partition=teaching \
  --account=mlnlp2.pilot.s3it.uzh \
  --qos=normal \
  --gres=gpu:8 \
  --cpus-per-task=32 \
  --mem=128G \
  --time=04:00:00 \
  --pty bash
```

When immediate/no-wait behavior is supported, prefer:

```bash
srun \
  --partition=teaching \
  --account=mlnlp2.pilot.s3it.uzh \
  --qos=normal \
  --gres=gpu:8 \
  --cpus-per-task=32 \
  --mem=128G \
  --time=04:00:00 \
  --immediate=120 \
  --pty bash
```

This 8-GPU, 32-CPU, 128G, 4-hour profile has been tested by the user and can often
start immediately. It is known to have started immediately many times and should not be
replaced by weak `balanced` defaults. For large GPU-capable jobs, try this strongest
tested immediate profile first unless the workload clearly cannot use GPUs or the user
requests otherwise.

Strongest tested immediate GPU profile values:

- partition: `teaching`
- account: `mlnlp2.pilot.s3it.uzh`
- QoS: `normal`
- GPU: `--gres=gpu:8`
- CPUs: `--cpus-per-task=32`
- memory: `--mem=128G`
- time: `--time=04:00:00`

If the strongest tested profile cannot start immediately, cancel the waiting attempt,
log the failed stronger attempt, and try the next strongest tested immediate profile.
Do not step down to weaker resources to be `balanced`. Step down only when the stronger
immediate request fails or the workload cannot use those resources.

Never silently downgrade. Every downgrade must record:

- attempted command
- failure reason or immediate-allocation failure
- next profile selected
- why the next profile is still appropriate

Never submit jobs that wait indefinitely by default. Never silently choose weaker
resources.

### General Heavy-resource Rules

- For heavy GPU/CPU/memory/data/inference/training tasks, use
  `~/bin/claim_best_immediate_resource.sh` unless the user explicitly requests a
  different strategy.
- Use `--mode gpu` for GPU tasks.
- Use `--mode cpu` for CPU-only heavy, memory-heavy, preprocessing, feature extraction,
  statistics, cross-validation, bootstrap, and large data-processing tasks.
- Use immediate/no-wait allocation behavior by default.
- Use `--immediate=120` when claiming resources through Slurm unless the user
  explicitly asks for a queued job.
- Log every failed stronger attempt and why a weaker configuration was selected.
- Never use CPU fallback for GPU tasks.
- Never use weak local/login-node fallback for CPU-heavy tasks.
- If a task requires GPU, abort if PyTorch/CUDA sees zero GPUs or fewer GPUs than
  requested.
- If multiple GPUs are requested, verify the code actually uses them.
- If the code is CPU-only, do not request GPUs. Instead request the strongest immediate
  CPU/memory resources appropriate to the workload.
- For CPU-only heavy jobs, request the strongest immediate CPU/memory profile known from
  local policy, scripts, or previous logs.
- If no tested CPU profile is known, use
  `~/bin/claim_best_immediate_resource.sh --mode cpu` or inspect local policy rather
  than inventing weak defaults.
- For CPU-heavy work, verify CPU count, memory, and actual parallelism before scaling.
- If CPU-heavy code is single-threaded, parallelize across shards/files/folds/seeds/
  configs rather than only requesting more CPUs.
- Do not under-request CPU or memory for heavy jobs.
- For memory-heavy jobs, increase memory before reducing CPU count when possible.
- Do not kill productive running jobs.
- Long runs must be checkpointed, resumable, and have manifests, config hashes, logs,
  and status files.

## Tested Resource Profiles and Resource Discovery

- Prefer known tested resource commands and numbers over generic guesses.
- Directly inspect these before heavy work:
  - `~/.codex/AGENTS.md`
  - `~/bin/claim_best_immediate_resource.sh`
  - repo Slurm scripts
  - repo run scripts
  - previous AI run logs
  - recent Slurm logs
  - user-provided commands, tested numbers, or launch notes
- If tested CPU/GPU/memory/wall-time/account/partition/QoS values are found, reuse them
  exactly unless the workload clearly requires a documented change.
- If no tested values are found, use the strongest-immediate discovery path through
  `claim_best_immediate_resource.sh`; do not invent exact resource numbers.
- The known tested 8-GPU profile is not merely an example; it is the first strong GPU
  profile to try for large GPU-capable jobs unless the workload cannot use GPUs or the
  user requests otherwise.
- If it cannot start immediately, cancel the waiting request and try the next strongest
  tested immediate profile.
- Do not leave an interactive `srun` request waiting indefinitely.
- A smaller profile is acceptable only after the stronger immediate request fails, or
  when the workload cannot use the stronger resources.
- Never choose generic resource defaults for heavy work when tested stronger options
  exist.
- Never downgrade from a tested/requested resource profile without logging the reason.
- If a strong tested profile cannot start immediately, log the failed attempt and try
  the next strongest tested immediate candidate.
- Do not submit a waiting job unless the user explicitly asks.

## Slurm Submission Rules

- For ad hoc heavy commands and resource-sensitive work, prefer:

```bash
~/bin/claim_best_immediate_resource.sh
```

- For production runs, prefer `sbatch` only after the resource request has been selected
  according to the strongest-immediate policy.
- Do not use ordinary queued `sbatch` by default unless the user explicitly requests a
  queued job or immediate allocation is unsuitable and the reason is logged.
- Before submitting any existing `sbatch` script for a heavy job, inspect its `#SBATCH`
  resource lines.
- If the script uses weaker or fixed allocation, patch/wrap it or log why it is
  appropriate.
- A successful `sbatch --test-only` or immediate-start prediction is not enough; Codex
  must justify CPU/GPU/memory/wall-time choice.
- Production `sbatch` scripts are acceptable only when they encode the selected strong
  resource request, include required preflight checks, and write logs/status outputs.

## Required Slurm Preflight and Post-run Checks

For every allocated Slurm job, run this preflight before the actual command:

```bash
hostname
echo SLURM_JOB_ID=$SLURM_JOB_ID
echo SLURM_STEP_ID=$SLURM_STEP_ID
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
echo SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
echo SLURM_MEM_PER_NODE=$SLURM_MEM_PER_NODE
echo SLURM_MEM_PER_CPU=$SLURM_MEM_PER_CPU
nproc
free -h
ulimit -a
nvidia-smi || true
python - <<'PY'
import os
print("python preflight")
print("cpu_count_os", os.cpu_count())
try:
    import psutil
    print("cpu_count_psutil", psutil.cpu_count())
    print("virtual_memory", psutil.virtual_memory())
except Exception as e:
    print("psutil_unavailable", e)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(i, torch.cuda.get_device_name(i), p.total_memory)
except Exception as e:
    print("torch_preflight_failed", e)
PY
```

- For GPU tasks, abort if CUDA/GPU visibility is wrong.
- For CPU-only tasks, verify CPU count and memory before running.
- After Slurm jobs, always run and record:

```bash
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW
```

- Also run and record when available:

```bash
seff <JOBID>
```

- Do not trust outputs until `sacct` confirms successful completion.
- Low CPU efficiency means the code may not be using allocated CPUs; do not blindly
  request more CPUs without inspecting parallelism/vectorization.

## Login-node Protection

- No heavy work on a login node.
- Login nodes are only for inspection, editing, tiny smoke tests, and job submission.
- Do not run heavy extraction, batch feature generation, large joins, large text
  processing, embedding generation, large NLP/LLM inference, model training,
  hyperparameter sweeps, bootstrap evaluation, cross-validation, or long-running
  analysis on a login node.
- If `CPU time limit exceeded` appears at a login-node prompt, check for local/
  background processes before blaming Slurm allocation.
- Use `squeue`, `sacct`, `ps`, and `jobs -l` to distinguish Slurm jobs from login-node
  processes.
- Do not leave background compute terminals running on the login node.

## Heavy Compute / Long Runtime Policy

Before launching a computationally expensive task, estimate the workload size. Estimate,
as relevant:

- number of files, participants, texts, tokens, windows, rows, or records
- number of grid combinations, folds, seeds, candidate batches, or bootstrap samples
- expected runtime and memory from a small representative sample
- expected output count, size, and merge behavior
- whether the workload is CPU-bound, memory-bound, I/O-bound, GPU-bound, or blocked by
  uncached recomputation

Rules:

- If a full run is expected to take more than 30 minutes, first determine whether it can
  be split across files, participants, texts, folds, seeds, grid chunks, candidate
  batches, or time windows.
- Prefer parallel shards over a single serial loop when units are independent.
- On Slurm/HPC systems, prefer job arrays when work can be split cleanly.
- Do not use `simple but slow` serial execution when Slurm/job arrays are available.
- Do not use generic or minimal resources for large work when tested stronger immediate
  resources exist.
- Do not assume that a Slurm allocation makes code parallel.
- Requesting many CPUs only helps when the code uses multiprocessing, threading,
  vectorized libraries, job arrays, or another tested parallel method.
- If code is single-threaded, parallelize across shards/files/folds/seeds/configs rather
  than only requesting more CPUs.
- Each shard must write to a separate output file or directory.
- Sharded programs must support resume/restart:
  - skip completed shards
  - preserve partial results
  - avoid overwriting successful outputs
  - merge outputs only after all required shards finish
- Cache and reuse expensive extraction, tokenization, feature, embedding, and window
  computations when possible.
- Cache keys/manifests should include enough information to prevent stale reuse: input
  reference, config, code version, and feature version when feasible.
- Before scaling up, run a small test shard to verify correctness, runtime estimate,
  memory use, output format, logging, and merge behavior.
- Every heavy run must have:
  - config
  - manifest
  - log path
  - output path
  - resume behavior
  - validation plan
- For GPU-heavy jobs, also log GPU utilization evidence when available.
- Do not overwrite completed results. Use a new timestamped or configured output path.
- Do not launch duplicate jobs that write to the same output path as an active job.
- Do not kill running productive jobs without explicit user authorization.
- When a job is slow or appears stalled, distinguish Slurm queue wait time,
  reconnect/session issues, actual compute time, single-core bottlenecks, GPU underuse,
  memory pressure, I/O bottlenecks, repeated uncached computation, and excessive grid
  size.

Serial anti-pattern:

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

## Long-Running Job Triage and Safe Acceleration Policy

- Do not passively wait for long-running jobs that are slow, stalled,
  under-resourced, not producing expected outputs, or likely to miss useful turnaround.
- Analyze progress, estimate ETA, diagnose bottleneck, and decide whether safe
  acceleration is possible.
- Never kill a productive running job unless the user explicitly authorizes it.
- Never modify raw/source data.
- Never corrupt, overwrite, race, or duplicate an active output path.
- Safe acceleration must:
  - leave the original job untouched
  - preserve the same input/output contract
  - write to a separate staging path such as `results/<task>/accelerated_<timestamp>/`
  - use stronger immediate Slurm resources when appropriate
  - log why it was launched
  - validate outputs before treating them as final
- Unsafe acceleration includes:
  - rerunning the same command against the same output path
  - overwriting active outputs
  - editing raw data
  - switching GPU work to CPU
  - silently reducing resources
  - claiming success from incomplete/different schema
- When triaging a long-running job, write or update `logs/current_status.md` with:
  - job id
  - command
  - working directory
  - elapsed time
  - requested resources
  - observed usage
  - expected/current outputs
  - last output modification time
  - progress estimate
  - ETA
  - bottleneck hypothesis
  - whether original job is left running
  - acceleration recommendation
  - exact accelerated resource request
  - staging output path
  - validation plan
  - assumptions/uncertainties
- If an accelerated attempt finishes first, validate schema, row counts, checksums or
  deterministic fields where applicable, compare against partial/expected outputs where
  possible, do not delete original outputs, and do not kill the original job without
  explicit approval.

## Output / Reproducibility Policy

- Results go under `results/`.
- Runtime logs go under `logs/`.
- New nontrivial outputs must have configs or manifests.
- Do not overwrite completed results.
- Decision records go into `docs/decisions.md` or `docs/adr/`.
- Record command, input reference, script, config, environment, resource profile,
  timestamp, code version, and validation for nontrivial generated outputs.
- Benchmark artifacts must document scoring definitions, split definitions, leakage
  controls, and claim language before results are treated as publishable.

## AI Run Logging Policy

For every non-trivial Codex task, create
`logs/ai_runs/YYYY-MM-DD_HHMM_<slug>.md` and update `logs/ai_runs/INDEX.md`.

Log:

- prompt/request summary
- plan
- files inspected
- files modified
- commands run
- validation results
- final response summary
- commit/push status

For heavy jobs, also log:

- selected resource profile
- failed stronger attempts
- exact Slurm command
- job id
- preflight summary
- post-run `sacct` / `seff` summary
- output path
- progress evidence
- bottleneck hypothesis
- validation result
- whether any original job was left running

If package installation occurs, log:

- active environment
- install command
- dependency file changed, if any
- validation after install

Do not log massive stdout, raw data, subject-identifiable data, ignored file contents,
secrets, tokens, or credentials.

## Git Commit / Push Policy

- This GitHub repository is a research checkpoint system.
- At the end of every completed non-trivial task, Codex should commit and push unless
  the user explicitly says not to.
- Never use broad staging commands:

```bash
git add .
git add -A
git add *
```

- Stage only explicit safe paths.
- Never stage:
  - `data/`
  - `raw/`
  - `external/`
  - `extracted/`
  - `derived/`
  - `results/`
  - local environments
  - model checkpoints
  - large scientific artifacts
  - local configs
  - caches
  - secrets
  - files greater than or equal to 100 MB
- Before commit/push run:

```bash
git status --short
git diff --stat
git diff --check
```

- Also run a large-file check over staged/tracked files.
- If the working tree contains unrelated local files, leave them unstaged.
- If push fails, record the exact failure and do not retry blindly.

## Commit / Push Failure Behavior

- If commit or push fails, do not keep retrying blindly.
- Record the failure in the AI run log.
- Show the exact failure summary to the user.
- Leave the repository in a safe, inspectable state.
- Do not rewrite Git history unless the user explicitly asks.

## Lightweight Test Commands

Run only commands that apply to the current repository state. Run validation through the
CopCo environment.

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

## Agent Behavior When Unsure

- Inspect before assuming.
- Prefer the smallest safe change that advances the task.
- Do not invent data availability.
- Do not invent benchmark results.
- Do not invent tested cluster numbers.
- Do not silently change scientific framing.
- Do not silently weaken resources.
- Do not silently switch environments.
- Report blockers and uncertainty explicitly.
- Ask only when the decision cannot be made safely from repository evidence.
