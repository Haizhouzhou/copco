# Phase 3 Research Exploration

## Request Summary

Build Phase 3 controlled research exploration from the prepared Label Release v1.1
dataset without generating new labels, without LLM-generated annotations, and without
random word-level predictive splits.

Prepared input:

- Feature release: `results/feature_release_v1_20260505_2155`
- Label release: `results/label_release_v1_1_20260506_0041`
- Prepared dataset: `results/label_release_v1_1_20260506_0041/prepared_dataset`

## Plan

1. Add a Phase 3 config and research-exploration implementation using existing CLI style.
2. Validate the prepared dataset, participant labels, segmentation labels, quality labels, and split labels.
3. Produce text exposure, LM warning, segmentation, reader-group interaction, residualization,
   participant prediction, word-level ladder, and decision reports.
4. Add tests for leakage controls, residualization predictors, ablation definitions, and report generation.
5. Run full validation, commit, and push safe files only.

## Files Inspected

- `pyproject.toml`
- `src/copco_eye_bench/cli.py`
- `src/copco_eye_bench/research_exploration.py`
- `tests/test_research_exploration.py`
- `analysis/research_exploration/*`
- `results/research_exploration_v1_20260506_0149/manifest.json`

## Files Modified

- `configs/research_exploration_v1.yaml`
- `src/copco_eye_bench/research_exploration.py`
- `src/copco_eye_bench/cli.py`
- `pyproject.toml`
- `tests/test_research_exploration.py`
- `analysis/research_exploration/*.md`
- `analysis/research_exploration/*.csv`
- `analysis/research_exploration/figures/segmentation_effects.png`
- `logs/ai_runs/2026-05-06_0159_phase3_research_exploration.md`
- `logs/ai_runs/INDEX.md`

## Heavy Job

Full Phase 3 build:

- Command: `copco-run-research-exploration --config configs/research_exploration_v1.yaml --output-dir results/research_exploration_v1_20260506_0149`
- Launcher: `~/bin/claim_best_immediate_resource.sh --mode cpu`
- Candidate: `--partition=teaching --account=mlnlp2.pilot.s3it.uzh --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G --time=04:00:00`
- Slurm job: `2724341`
- Host: `u24-chiivm0-603`
- CPU/memory visible: 32 allocated CPUs, 128G requested, 755Gi node memory visible
- GPU visibility: no CUDA devices, expected for CPU-only analysis
- State: `COMPLETED`
- Elapsed: `00:05:25`
- Exit code: `0:0`
- MaxRSS: `2846368K`
- AveCPU: `00:21:08`
- `seff`: unavailable on this system

Earlier exploratory run `2724182` completed but used the initial association-model path
with empty coefficient summaries. It was superseded by the validated output above.

## Output Summary

Final output directory:

- `results/research_exploration_v1_20260506_0149`

Prepared rows:

- Word-level rows: 335,203
- Sentence-level rows: 1,986
- Participant-level rows: 57
- Participant sensitivity profiles: 57

Decision categories:

- Segmentation opacity beyond controls: `not_supported`
- Reader-group interactions: `promising_signal`
- Participant-level prediction after ablation: `strong_signal`

Text-exposure audit:

- Flagged exposure-count variables excluded from primary prediction:
  `n_speeches`, `n_word_rows`, `n_words_read`, `total_word_rows`, `word_observation_count`

LM warning audit:

- Dyslexia-labeled rows: 82,179; warning rate 0.9979; LM missing rate 0.0054
- Typical/control rows: 253,024; warning rate 0.9980; LM missing rate 0.0052
- `non_special_token_unassigned` rows: 334,516; LM missing rate 0.0052

Segmentation analysis:

- Coefficient rows: 288
- Significant boundary terms: 0
- Consistent opacity direction: false

Reader-group interactions:

- Interaction terms: 14
- Significant/stable interaction terms: 3
- Stable terms: `reader_group_x_word_length_chars_z`,
  `reader_group_x_dfm_lm_word_surprisal_z`,
  `reader_group_x_prev_boundary_opacity_score_z`

Participant prediction:

- Best feature group: `D_dfm_exposure_and_sensitivity`
- Best model/split: logistic regression with leave-one-participant-out
- ROC-AUC: 0.9058
- PR-AUC: 0.8686
- Balanced accuracy: 0.8816
- Macro F1: 0.8667
- Brier score: 0.1080
- Usable LOPO folds: 57
- Permutation p-value: 0.0099 with 100 permutations
- Bootstrap ROC-AUC CI: [0.8162, 0.9798]

Word-level secondary ladder:

- Best stage: `full_validated_feature_set`
- ROC-AUC: 0.6143
- PR-AUC: 0.3662
- Balanced accuracy: 0.5927
- Usable participant-grouped folds: 5
- Marked secondary because labels are participant-level.

## Commands Run

- `pwd && git status --short && python --version`
- `conda run -n copco python --version`
- `conda run -n copco python -c "import sys; print(sys.executable)"`
- `conda run -n copco python -m pip install -e .`
- `conda run -n copco python -m pytest tests/test_research_exploration.py -q`
- `conda run -n copco python -m ruff check src/copco_eye_bench/research_exploration.py tests/test_research_exploration.py`
- `~/bin/claim_best_immediate_resource.sh --mode cpu ... copco-run-research-exploration ...`
- `conda run -n copco copco-validate-research-exploration --config configs/research_exploration_v1.yaml --output-dir results/research_exploration_v1_20260506_0149`
- `sacct -j 2724341 --format=JobID,State,Elapsed,ExitCode,ReqTRES,AllocTRES,MaxRSS,AveCPU,CPUTimeRAW`
- `seff 2724341 || true`
- `conda run -n copco python scripts/validate_env.py`
- `conda run -n copco python -m pytest tests/ -q`
- `conda run -n copco python -m ruff check .`
- `conda run -n copco copco-validate-label-release --config configs/label_release_v1_1.yaml --output-dir results/label_release_v1_1_20260506_0041`
- `conda run -n copco python -c "import copco_eye_bench; import copco_eye_bench.research_exploration; print('imports ok')"`
- `git diff --check`

## Validation Results

- Environment validation: passed
- Import validation: passed
- Label Release v1.1 validation: passed
- Research exploration validation: passed
- Pytest: 27 passed, 3 warnings
- Ruff: passed
- `git diff --check`: passed

Pytest originally raised a `MemoryError` only while writing `.pytest_cache` after all
27 tests had passed. The project Pytest config now disables the cache provider so the
required command exits on the actual test result in this memory-limited login shell.

## Large Outputs Not Committed

- `results/research_exploration_v1_20260506_0149/`
- `analysis/research_exploration/participant_sensitivity_profiles.parquet`
- Any generated prepared/feature/label release Parquet artifacts under `results/`

## Final Response Summary

Phase 3 should deep-dive two directions: participant-level DFM predictability with
residualized gaze-cost profiles, and reader-group sensitivity interactions for word
length, DFM surprisal, and boundary opacity. Standalone segmentation-opacity main
effects are not supported in the controlled Phase 3 analyses and should be deferred.

## Commit / Push Status

Committed as part of the Phase 3 research-exploration change set. Final commit hash
and push status are reported in the assistant final response.
