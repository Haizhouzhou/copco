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

## 2026-05-02: Dyslexia-Labeled Reader Pipeline Scaffold

The research program in `plan/copco_dyslexia_research_plan.md` is implemented as
committed package code and CLI entrypoints. Generated feature tables, LM features,
model outputs, mixed-effects reports, and validation reports are written under ignored
`results/` run directories.

Operational decisions:

- Stable IDs are source-derived strings, with final word observations asserted unique
  by `(participant_id, word_id)`.
- Practice speech `1327` and participant `P14` are excluded by default.
- The local `derived57` package is the preferred source. If unavailable, manifests
  record the missing input and the pipeline can read the ignored legacy
  `copco-processing/` schema.
- Base language models are allowed for surprisal and entropy; instruction-tuned LLMs
  are restricted to optional ablation annotations and never used for surprisal.
- Sequence models remain exploratory because participant N is small.
- Slurm command helpers prepend the tested 8-GPU teaching profile for GPU scoring and
  include CopCo environment activation and the required preflight block.
- Smoke configs are honored by the CLI without repeating sample flags. Sampled
  feature builds also restrict IA-report cross-check reads to sampled participants to
  avoid login-node CPU timeouts during validation.
