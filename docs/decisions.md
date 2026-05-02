# Decision Record

## 2026-05-02: Phase 0 Initialization

This repository starts as a Phase 0 research-code checkpoint for CopCo / Eye Bench.
Phase 0 is reserved for freezing target definitions, scoring rules, dataset splits,
leakage controls, and claim language before benchmark results are generated.

Initial decisions:

- No raw data, copied datasets, derived participant-level tables, large artifacts, or
  local environments are committed.
- No benchmark results or benchmark claims are included at initialization.
- CopCo is treated as a Danish natural-reading eye-tracking corpus, not as a dyslexia
  dataset.
- Dyslexia-related and reading-difficulty-related language must remain operational and
  research-focused unless future project data explicitly support stronger claims.
- Reference labels are operational research labels, not biological or clinical ground
  truth.

## 2026-05-02: Operational Agent Policy Strengthened

`AGENTS.md` was strengthened to directly encode CopCo environment usage,
missing-package behavior, V100 strongest-immediate GPU policy, strongest-immediate
CPU/memory policy for CPU-only heavy work, required Slurm preflight/post-run checks,
long-job triage, safe acceleration, AI-run logging, and git safety.
