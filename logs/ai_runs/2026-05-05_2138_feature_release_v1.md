# Feature Release V1 And Initial Analysis Package

## Request Summary

Prepare a full CopCo feature-release package and first professional analysis package
for the next research-planning stage. Scope includes full feature tables, parser and
embedding features, full DFM LM scoring, optional Gemma sensitivity, joined modeling
tables, feature dictionary, label provenance report, validation reports,
psycholinguistic validation, dyslexia-labeled reader analysis, predictive models,
mixed-effects analysis, research-plan markdown, final release report, validation, commit,
and push.

## Plan

1. Inspect existing configs, source modules, CLIs, tests, validation, Slurm launchers,
   and docs.
2. Add `feature_release_v1` config and missing release-oriented CLIs/modules while
   reusing existing build, LM, model, mixed-effects, validation, and Slurm primitives.
3. Install missing project runtime dependencies inside `copco` only if needed.
4. Run full feature build with no smoke limits using immediate Slurm CPU resources.
5. Run full DFM LM scoring with immediate Slurm GPU resources and shard controls.
6. Run Gemma sensitivity only after DFM validates; document as pending if access or
   environment blocks it.
7. Generate parser, embeddings, joined tables, reports, analyses, model summaries, and
   feature dictionary.
8. Run all tests, ruff, and release validations.
9. Commit/push safe code, configs, docs, and logs only; leave large generated outputs
   ignored under `results/`.

## Environment Findings

- `conda run -n copco python` resolves to `/home/haizhe/conda/envs/copco/bin/python`.
- Repository worktree was clean at start.
- Missing before release installs: `dacy`, `spacy`, `stanza`, `sentence_transformers`,
  `lightgbm`, `xgboost`, `matplotlib`, and `seaborn`.

## Files Inspected

- `README.md`
- `pyproject.toml`
- `configs/copco_dyslexia_full.yaml`
- `configs/copco_dyslexia_smoke.yaml`
- `src/copco_eye_bench/cli.py`
- `src/copco_eye_bench/features.py`
- `src/copco_eye_bench/lm_features.py`
- `src/copco_eye_bench/modeling.py`
- `src/copco_eye_bench/mixed_effects.py`
- `src/copco_eye_bench/resources.py`
- `src/copco_eye_bench/slurm.py`
- `src/copco_eye_bench/validation.py`
- `tests/test_ids_features_splits_validation.py`
- `tests/test_lm_alignment.py`

## Commands Run

- `pwd`
- `git status --short`
- `python --version`
- `conda run -n copco python -c ...`

## Dependency Changes

- Repaired the existing `copco` environment after spaCy/DaCy import failed with a
  `blis` shared-object mapping error.
- Commands:
  - `conda run -n copco python -m pip install --force-reinstall --no-cache-dir 'blis==0.7.11'`
  - This temporarily installed incompatible `numpy 2.4.4`.
  - Corrected immediately with `conda run -n copco python -m pip install --force-reinstall --no-cache-dir 'numpy==1.26.4' 'blis==0.7.11'`.
- Follow-up validation:
  - `conda run -n copco python scripts/validate_env.py` passed.
  - `conda run -n copco python -m pytest tests/ -q` passed.
  - `conda run -n copco python -m ruff check .` passed.
- DaCy/spaCy still failed in this login-node context due
  `libtorch_cuda.so: failed to map segment from shared object`, so parser outputs are
  explicitly marked as `surface_heuristic` fallback in parser diagnostics.

## Validation Results

- `conda run -n copco python scripts/validate_env.py`: passed.
- `conda run -n copco python -m pytest tests/ -q`: `22 passed, 1 warning`.
- `conda run -n copco python -m ruff check .`: passed.
- `conda run -n copco copco-validate-run --output-dir results/feature_release_v1_20260505_2155`: passed.
- `conda run -n copco copco-validate-feature-release --config configs/feature_release_v1.yaml --output-dir results/feature_release_v1_20260505_2155`: passed.
- `git diff --check`: passed.

## Slurm Jobs

- `2722141`: base feature build, CPU teaching profile, completed `00:00:39`, exit `0:0`.
- `2722149`: release feature export and parser fallback features, CPU teaching profile,
  completed `00:00:18`, exit `0:0`.
- `2722155`: DFM decoder 7B causal-LM surprisal/entropy, 8-GPU teaching profile,
  completed `00:01:40`, exit `0:0`.
- `2722189`: Gemma 2 9B sensitivity LM attempt, 8-GPU teaching profile, failed
  `00:00:18`, exit `1:0`; Hugging Face gated model access denied.
- `2722194`: DFM sentence encoder embeddings, 8-GPU teaching profile, completed
  `00:00:37`, exit `0:0`.
- `2722203`: multilingual E5-large embeddings, 8-GPU teaching profile, completed
  `00:00:37`, exit `0:0`.
- `2722217`: modeling table joins, CPU teaching profile, completed `00:00:08`, exit `0:0`.
- `2722218`: predictive model ladder, CPU teaching profile, completed `00:05:26`, exit `0:0`.
- `2722264`: mixed-effects analysis, CPU teaching profile, completed `00:00:08`, exit `0:0`.
- `2722266`: analysis package first attempt, CPU teaching profile, failed `00:00:06`,
  exit `1:0`; completed rerun is `2722274`.
- `2722274`: analysis package reports, CPU teaching profile, completed `00:00:10`, exit `0:0`.
- `2722278`: final feature-release report, CPU teaching profile, completed `00:00:05`, exit `0:0`.
- Post-run accounting checked with `sacct -u "$USER" --starttime 2026-05-05T20:00 ...`.
- `seff` is not installed on this cluster.

## Outputs

- Release directory: `results/feature_release_v1_20260505_2155`.
- Full feature rows:
  - `features/word_level_gaze.parquet`: 335,203 rows.
  - `features/word_level_classical.parquet`: 31,986 rows.
  - `features/sentence_level.parquet`: 1,986 rows.
  - `features/paragraph_level.parquet`: 452 rows.
  - `features/participant_level.parquet`: 57 rows.
  - `features/text_level.parquet`: 32 rows.
- Label counts: 38 typical/control participants, 19 dyslexia-labeled participants.
- Parser outputs:
  - `linguistic_features/parser_word_level.parquet`: 31,986 rows.
  - `linguistic_features/parser_sentence_level.parquet`: 1,986 rows.
  - Backend: `surface_heuristic`; DaCy preferred but unavailable/stable import failed.
- DFM LM outputs:
  - `lm_features/dfm_decoder_7b/surprisal/surprisal_shard0000_of_0001.parquet`: 31,986 rows.
  - Alignment contexts: 452; status counts `ok: 8`, `warning: 444`; errors: none.
  - Warning: `non_special_token_unassigned` in 444 context reports.
- Gemma sensitivity:
  - Attempted `google/gemma-2-9b` base model.
  - Marked pending/aborted because Hugging Face gated model access returned 401.
- Embeddings:
  - `KennethEnevoldsen/dfm-sentence-encoder-large`: 1,986 sentence rows, 452 paragraph rows.
  - `intfloat/multilingual-e5-large`: 1,986 sentence rows, 452 paragraph rows.
- Joined modeling tables:
  - `modeling_tables/word_level_full.parquet`: 335,203 rows.
  - `modeling_tables/word_level_full_with_dfm_lm.parquet`: 335,203 rows.
  - `modeling_tables/sentence_level_full.parquet`, `paragraph_level_full.parquet`,
    `participant_level_full.parquet`, and `participant_aggregates.parquet` produced.
  - Join validation: duplicate participant-word keys `0`, unexpected row loss `0`,
    missing DFM LM rate `0.005265466001199273`, missing parser rate `0.0`, missing
    embedding rate `0.04202528020333947`.
- Feature dictionary:
  - `docs/feature_dictionary_v1.md`: generated and tracked.
  - `results/feature_release_v1_20260505_2155/feature_dictionary_v1.json`: 4,533 entries.
- Reports:
  - `label_provenance_report.md`
  - `modeling_tables/join_validation_report.md/json`
  - `analysis/psycholinguistic_validation/psycholinguistic_validation_report.md`
  - `analysis/dyslexia_group_analysis/dyslexia_labeled_reader_report.md`
  - `analysis/predictive_models/participant_level_model_report.md`
  - `analysis/predictive_models/word_level_model_ladder_report.md`
  - `mixed_effects/mixed_effects_report.md`
  - `analysis/research_plan_next_stage.md`
  - `feature_release_report.md`

## Commit / Push Status

- Safe paths staged only; generated `results/` artifacts were left ignored and unstaged.
- Commit message: `feat: produce feature release v1 and initial research analysis`.
- Push status is recorded in the final assistant response after the commit is pushed.
