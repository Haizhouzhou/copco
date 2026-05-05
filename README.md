# CopCo Eye Bench

Research-code checkpoint for CopCo / Eye Bench work on Danish natural-reading
eye-tracking analysis and benchmarking.

## Status

Phase 0 / repository initialization. This repository does not contain benchmark
results yet.

## Purpose

This project will support reproducible CopCo preprocessing, corpus audit, eye-tracking
feature analysis, Danish reading modeling, benchmark construction, and later carefully
scoped comparisons to other reading-related datasets where appropriate.

## Scientific Caution

CopCo is a Danish natural-reading eye-tracking corpus. It should not be described as a
dyslexia dataset. Dyslexia-related or reading-difficulty-related work must use careful
operational language and must not imply formal diagnostic status, medical validation, or
medical screening unless explicitly supported by the project data.

## Planned Layout

```text
.
  AGENTS.md                  project rules for Codex and other agents
  pyproject.toml             minimal Python project metadata
  src/copco_eye_bench/       lightweight package namespace
  scripts/                   maintenance and validation scripts
  tests/                     lightweight tests
  docs/                      decisions and protocol notes
  logs/ai_runs/              AI run logs and index
  results/                   generated outputs, not committed
  data/                      local data, not committed
```

## Setup

Use the existing `copco` conda environment for project commands on the UZH cluster:

```bash
conda activate copco
python scripts/validate_env.py
```

For a fresh local development environment outside the cluster, use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Lightweight Validation

```bash
python scripts/validate_env.py
python -m pytest tests/ -q
```

## Dyslexia-Labeled Reader Pipeline

The committed CLI scaffold implements the reproducible research program in
`plan/copco_dyslexia_research_plan.md` while keeping generated outputs ignored under
`results/`.

```bash
copco-build-features --config configs/copco_dyslexia_full.yaml
copco-run-lm-features --output-dir results/<run> --dry-run
copco-run-models --output-dir results/<run>
copco-fit-mixed-effects --output-dir results/<run>
copco-validate-run --output-dir results/<run>
```

LM surprisal and entropy features use base/pretrained causal language models. The
primary Danish LM is `danish-foundation-models/dfm-decoder-open-v0-7b-pt`; the
sensitivity LM is the base `google/gemma-2-9b`. Instruction-tuned models such as
Gemma-it, Gemma Instruct, Mistral-Instruct, and `gemma-2-9b-it` are not used for
surprisal or entropy.

Run a dry LM check without loading a model:

```bash
copco-run-lm-features \
  --config configs/copco_dyslexia_smoke.yaml \
  --output-dir results/<run> \
  --dry-run \
  --limit-items 1 \
  --max-word-tokens 32
```

Run a tiny real LM scoring job only on a GPU allocation, keeping the word budget small:

```bash
copco-run-lm-features \
  --config configs/copco_dyslexia_smoke.yaml \
  --output-dir results/<run> \
  --model-id danish-foundation-models/dfm-decoder-open-v0-7b-pt \
  --real-run \
  --require-gpu \
  --limit-items 1 \
  --max-word-tokens 32
```

On the UZH V100 cluster, the `copco` environment must use a PyTorch build with `sm_70`
support. This run validated `torch==2.7.1+cu126`; `torch 2.11.0+cu130` did not support
V100 execution.

For heavy LM scoring or CPU-heavy modeling, print the Slurm launcher command first:

```bash
copco-build-features --config configs/copco_dyslexia_full.yaml --print-slurm-command
copco-run-lm-features \
  --config configs/copco_dyslexia_smoke.yaml \
  --output-dir results/<run> \
  --model-id danish-foundation-models/dfm-decoder-open-v0-7b-pt \
  --real-run \
  --limit-items 1 \
  --max-word-tokens 32 \
  --print-slurm-command
copco-run-models --output-dir results/<run> --print-slurm-command
```

The pipeline uses the local `derived57` package when it is importable. If it is absent,
the run manifest records that missing input and falls back to the ignored local
`copco-processing/` schema when available. Scientific wording remains
"dyslexia-labeled reader"; outputs are not formal diagnostic status, screening, or medical
validation.

## Feature Release V1

Use `configs/feature_release_v1.yaml` for the full feature release. This config forbids
smoke participant or speech limits, uses stable word identifiers, keeps participant
labels at participant scope, and records Slurm CPU/GPU settings for the UZH cluster.

Create a timestamped release directory and run the stages with the `copco` environment:

```bash
export RELEASE_DIR=results/feature_release_v1_$(date +%Y%m%d_%H%M)

copco-build-features \
  --config configs/feature_release_v1.yaml \
  --output-dir "$RELEASE_DIR" \
  --print-slurm-command

copco-write-release-features \
  --config configs/feature_release_v1.yaml \
  --output-dir "$RELEASE_DIR" \
  --print-slurm-command

copco-run-parser-features \
  --config configs/feature_release_v1.yaml \
  --output-dir "$RELEASE_DIR" \
  --print-slurm-command
```

Run full DFM LM scoring only with a GPU allocation. Surprisal and entropy must use
base/pretrained causal LMs, not instruction-tuned models:

```bash
copco-run-lm-features \
  --config configs/feature_release_v1.yaml \
  --output-dir "$RELEASE_DIR" \
  --model-id danish-foundation-models/dfm-decoder-open-v0-7b-pt \
  --model-label dfm_decoder_7b \
  --real-run \
  --require-gpu \
  --print-slurm-command
```

Gemma is a non-blocking sensitivity model. Use the base model only:

```bash
copco-run-lm-features \
  --config configs/feature_release_v1.yaml \
  --output-dir "$RELEASE_DIR" \
  --model-id google/gemma-2-9b \
  --model-label gemma2_9b \
  --real-run \
  --require-gpu \
  --print-slurm-command
```

Build embeddings, joins, modeling outputs, mixed-effects outputs, and reports:

```bash
copco-run-embeddings \
  --config configs/feature_release_v1.yaml \
  --output-dir "$RELEASE_DIR" \
  --print-slurm-command

copco-build-modeling-tables \
  --config configs/feature_release_v1.yaml \
  --output-dir "$RELEASE_DIR" \
  --print-slurm-command

copco-run-models \
  --config configs/feature_release_v1.yaml \
  --output-dir "$RELEASE_DIR" \
  --print-slurm-command

copco-fit-mixed-effects \
  --config configs/feature_release_v1.yaml \
  --output-dir "$RELEASE_DIR" \
  --print-slurm-command

copco-run-analysis-package \
  --config configs/feature_release_v1.yaml \
  --output-dir "$RELEASE_DIR" \
  --print-slurm-command

copco-finalize-feature-release \
  --config configs/feature_release_v1.yaml \
  --output-dir "$RELEASE_DIR"
```

Validate the release:

```bash
copco-validate-run --output-dir "$RELEASE_DIR"
copco-validate-feature-release \
  --config configs/feature_release_v1.yaml \
  --output-dir "$RELEASE_DIR"
```

Expected output layout includes `features/`, `linguistic_features/`, `lm_features/`,
`embedding_features/`, `modeling_tables/`, `analysis/`, `feature_dictionary_v1.json`,
`label_provenance_report.md`, and `feature_release_report.md`. Generated Parquet,
embedding, model-output, and result files stay under ignored `results/`; commit only
code, configs, docs, AI logs, and small release summaries when appropriate.

## Label Release v1.1 And Prepared Dataset Freeze

Use `configs/label_release_v1_1.yaml` to derive deterministic labels and freeze the
analysis-ready dataset from Feature Release V1. The config points to
`results/feature_release_v1_20260505_2155`, forbids smoke mode, forbids LLM-generated
core labels, and permits only participant-grouped split labels.

Build the release through the CPU Slurm launcher:

```bash
copco-build-label-release \
  --config configs/label_release_v1_1.yaml \
  --print-slurm-command
```

Or write to an explicit release directory:

```bash
copco-build-label-release \
  --config configs/label_release_v1_1.yaml \
  --output-dir results/label_release_v1_1_<timestamp> \
  --print-slurm-command
```

Validate the generated label release:

```bash
copco-validate-label-release \
  --config configs/label_release_v1_1.yaml \
  --output-dir results/label_release_v1_1_<timestamp>
```

If label files already exist and only the prepared tables need to be rebuilt:

```bash
copco-freeze-prepared-dataset \
  --config configs/label_release_v1_1.yaml \
  --output-dir results/label_release_v1_1_<timestamp> \
  --print-slurm-command
```

Generated files are under:

```text
results/label_release_v1_1_<timestamp>/
  labels/
    participant_labels_v1.parquet
    segmentation_boundary_labels_v1.parquet
    segmentation_word_labels_v1.parquet
    segmentation_sentence_labels_v1.parquet
    quality_labels_v1.parquet
    split_labels_v1.parquet
  prepared_dataset/
    analysis_ready_word_level_v1_1.parquet
    analysis_ready_sentence_level_v1_1.parquet
    analysis_ready_participant_level_v1_1.parquet
    analysis_ready_manifest.json
  analysis/label_analysis/
  manifest.json
  label_release_report.md
  label_release_validation_report.json
```

Segmentation-opacity labels are deterministic orthographic stimulus labels based on
Danish C/V boundaries using `a e i o u y æ ø å` and uppercase variants. They are not
pronunciation-aware labels and are not target labels. The current parser status
`surface_heuristic_fallback` means parser features are surface/morpho-orthographic
heuristics, not true syntactic annotations.

Legal split labels are `leave_one_participant_out`, `participant_grouped_kfold`, and
`sensitivity_exclude_uncertain_labels`. Random word-level train/test splits are
prohibited because word rows are not independent and participant-level labels would
leak across folds.

Label-card and policy documents are committed under `docs/`, and small label-analysis
reports are committed under `analysis/label_analysis/`. Large Parquet prepared-dataset
files remain under ignored `results/` and should not be committed.

## Data Policy

Raw data, copied datasets, derived participant-level tables, large artifacts, model
checkpoints, and local environments must not be committed. Keep local data under
ignored paths such as `data/`, `raw/`, `external/`, `extracted/`, or `derived/`, and
document how to obtain or regenerate it.

## Reproducibility Policy

Benchmark results must be reproducible from committed code, documented external data
inputs, frozen target definitions, scoring rules, dataset splits, leakage controls, and
claim language. Phase 0 must freeze those choices before generating benchmark results.

## Citations

Citation entries will be added before public benchmark release:

- CopCo first-release paper: TBD
- CopCo dataset record: TBD
