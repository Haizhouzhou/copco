# MasterResearchRecord v1

Internal project document for CopCo / Eye Bench research-code evidence tracking.

- Build timestamp: `2026-05-24T08:32:12`
- Repository branch: `codex/d3-model-evidence-v1-1`
- Repository commit: `6ce219edfe344efa13ddf99c2b605a5c823605f3`
- Primary output path: `analysis/master_research_record_v1/MASTER_EXPERIMENT_RECORD.md`
- Scope: existing evidence only; no new training, feature search, metric optimization, paper figures, final paper tables, manuscript rewrite, or new scientific claim.

## SECTION 1 — Executive project map

The project objective is to preserve the complete factual route for CopCo Danish natural-reading eye-tracking analysis and its EyeBench-related benchmark comparisons. The target task throughout the main predictive work is operational reader-group classification between dyslexia-labeled readers and typical/control readers. The full prepared dataset records 57 participants, 19 dyslexia-labeled readers, 38 typical/control readers, and 335203 participant-word gaze rows.

The target label is reader-level because the operational label belongs to the participant, not to a word, fixation, sentence, or trial. Word-level observations are repeated evidence from the same reader and must not be treated as independent target labels. The full-record reader-profile model and fixed-budget online evidence model are therefore separated: the full-record model uses the complete reader record for retrospective profiling, while the online model records what can be inferred from prefix-limited evidence without future rows.

The public-facing umbrella method name used in this record is the residualized predictability-sensitive gaze-profile method. Internally this family appears as D3, D3 offline, D3 online, D3_Lite, BenchmarkBridge, OfficialEyeBenchAlignment, OperatingPointAdaptation, OnlineTargetedOptimization, and D3ModelEvidenceVault. Each internal term is paired with a public description below.

Internal term mapping:

| internal | public | description |
| --- | --- | --- |
| D3 | residualized predictability-sensitive gaze-profile method | umbrella name for residualized DFM gaze-profile rows. |
| D3 offline | full-record reader-profile model | participant-level model using the full reading record. |
| D3 online | fixed-budget sequential reader-evidence model | online prefix model using only evidence available up to a prefix. |
| D3_Lite | reduced official-protocol-compatible trial-level variant | trial-level reduced feature variant for official-compatible stress tests. |
| BenchmarkBridge | internal EyeBench-style benchmark comparison | full-data reader-aggregated benchmark-relative comparison. |
| OfficialEyeBenchAlignment | official protocol and data-alignment audit | audit of fold/data/evaluator alignment with EyeBench. |
| OperatingPointAdaptation | probability-first operating-point diagnostic | threshold, calibration, and aggregation analysis. |
| OnlineTargetedOptimization | fixed-budget online and stopping-policy evaluation | online prefix, accumulator, and stopping-policy evaluation. |
| D3ModelEvidenceVault | curated model evidence vault | source-traced internal evidence package for D3 results. |

Result status map:

| status | result_family | recorded_value | source |
| --- | --- | --- | --- |
| primary | Full-record reader-profile result | AUROC 0.8947; BA 0.8421; PR-AUC 0.8641 | analysis/autoresearch_v1/tables/final_model_metrics_table.csv |
| secondary | Fixed-budget online reader evidence | late/mid/early rows separated from full-evidence rows | analysis/d3_online_targeted_optimization_v2/strict_final_models.csv |
| diagnostic | Operating point, oracle threshold, and stopping diagnostics | oracle/test-label thresholds excluded from clean evidence | analysis/operating_point_adaptation_v1/ |
| blocked | Official EyeBench leaderboard result | blocked by missing official processed data/environment | analysis/official_eyebench_sota_check_v1/ |
| unresolved | Unseen-text specialist value | rescue_04 and rescue_05 differ by criterion/context | analysis/d3_online_targeted_optimization_v2/unseen_text_rescue_candidates.csv |

Metric inventory used by this master record:

| metric_table | rows |
| --- | --- |
| canonical_metrics | 486 |
| external_baselines | 56 |
| online_prefix | 2477 |
| online_stopping | 2133 |
| oracle | 3827 |
| number_registry | 108 |
| unresolved_conflicts | 1 |
## SECTION 2 — Timeline of completed research stages

Chronological route. Dates come from timestamped output directories or source reports where available. Missing branch/commit fields are explicitly recorded rather than inferred.

| stage | phase | date_time | branch_commit | purpose | input_data | outputs | commands_validation | key_results | status | stored_at | role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Initial scaffold and environment validation | repository scaffold / environment checks | historical; exact timestamp not centralized | available through logs/ai_runs and git history | establish package, validation script, split policy, and safe data framing | repository source plus local CopCo derived data references | README, docs, scripts/validate_env.py, package scaffold | scripts/validate_env.py and pytest smoke tests where logged | CopCo environment and package validation became available | historical record | src/, tests/, docs/, logs/ai_runs/ | historical record |
| Real LM scoring enablement | Feature Release preparation | 2026-05-05 21:55 output | 205cb7d465105a54caa439c5e182e4e4ac11f04d | enable real Danish causal-LM surprisal/entropy and embeddings | feature-release word/stimulus tables | DFM decoder 7B word-level LM features and embedding features | Slurm jobs 2722155, 2722194, 2722203; validation passed | DFM decoder completed; Gemma 2 9B blocked by gated access | completed with Gemma blocked/deferred | results/feature_release_v1_20260505_2155/ | main feature source and historical LM-status record |
| Feature Release v1 | feature_release_v1 | 2026-05-05 21:55 | 205cb7d465105a54caa439c5e182e4e4ac11f04d | freeze gaze, text, parser-fallback, LM, embedding, and modeling tables | CopCo derived57 normalized source layers | feature tables, modeling tables, validation reports, feature dictionary | feature release validation passed | 57 participants, 335203 word observations, 31986 stimulus words | complete | results/feature_release_v1_20260505_2155/ | main analysis input |
| Label Release v1.1 | label_release_v1_1 | 2026-05-06 00:41 | d7a89ecd12992203dde91b5be17fd22629e4338a | freeze operational reader labels, quality labels, split labels, and prepared dataset | Feature Release v1 | participant, quality, segmentation, split, and analysis-ready tables | label release validation passed | 57 participants, 19 dyslexia-labeled, 38 typical/control | complete | results/label_release_v1_1_20260506_0041/ | main analysis input |
| Segmentation / boundary-opacity label generation | label_release_v1_1 segmentation layer | 2026-05-06 00:41 | d7a89ecd12992203dde91b5be17fd22629e4338a | derive orthographic C/V boundary descriptors | stimulus word sequence | segmentation_boundary, segmentation_word, segmentation_sentence labels | label release validation and segmentation reports | 31986 boundary/word labels; standalone main-effect support not retained | complete as secondary interpretability feature | results/label_release_v1_1_20260506_0041/labels/ | secondary/diagnostic analysis |
| Phase 3 controlled research exploration | research_exploration_v1 | 2026-05-06 01:49 | not centralized in indexed manifest | explore controlled participant profiles, residualization, interactions, and ablations | Label Release v1.1 prepared dataset | ablation metrics, residual profiles, segmentation/group interaction reports | research exploration validation report passed | DFM exposure+sensitivity exploratory AUROC 0.9058; segmentation main effect not supported | complete | results/research_exploration_v1_20260506_0149/ | historical and secondary source |
| Phase 4 confirmatory sensitivity analysis | phase4_confirmatory_sensitivity_v1 | 2026-05-06 07:15 | not centralized in indexed manifest | lock cross-fitted residualized DFM gaze-profile model and robustness tests | Phase 3 selected feature family and Label Release v1.1 | confirmatory metrics, bootstrap/permutation, feature stability, interactions | phase4 confirmatory validation passed | D3 residual gaze AUROC 0.8947, BA 0.8421, p about 0.001 | complete | results/phase4_confirmatory_sensitivity_v1_20260506_0715/ | main analysis |
| AutoResearch v1 | autoresearch_v1 | 2026-05-06 09:17 | not centralized in indexed manifest | assemble final selection, stress tests, and source-traced paper material | feature release, label release, Phase 3, Phase 4 outputs | final model metrics, DFM ablation, stress-test tables, decision report | autoresearch validation passed | selected D3_dfm_residual_gaze_only logistic regression LOPO | complete | results/autoresearch_v1_20260506_0917/ | main analysis source |
| SubmissionSprint v1 | submission_v1 | 2026-05-06 09:36 | not centralized in indexed manifest | package submission-era material | AutoResearch and manuscript source artifacts | submission package, reproducibility records, supplement sources | submission package validation where present | historical packaging; not a new result family | complete as historical source | results/submission_v1_20260506_0936/ and paper/submission_v1/ | historical record |
| Final Manuscript Audit v1 | final_manuscript_audit_v1 | 2026-05-06 14:38 | not centralized in indexed manifest | audit manuscript/result consistency | submission package and evidence outputs | audit reports | audit validation report where present | claim and blocker status preserved | complete as audit source | results/final_manuscript_audit_v1_20260506_1438/ | diagnostic/historical record |
| BenchmarkBridge v1 | benchmark_bridge_v1 | 2026-05-06 18:36 | not centralized in indexed manifest | evaluate full-data D3 in internal EyeBench-style split regimes | prepared full CopCo feature data and D3 residual profiles | TYP/RCS metrics, split diagnostics, residualization diagnostics | benchmark bridge validation passed | reader-aggregated full-data D3 AUROC 0.8961 unseen_reader | complete | results/benchmark_bridge_v1_20260506_1836/ | secondary benchmark-relative analysis |
| OfficialEyeBenchAlignment v1 | official_eyebench_alignment_v1 | 2026-05-06 22:32 | EyeBench submodule ce87f38a3083aeed029c255716a1a51e6ae51167 | audit official EyeBench data/fold/evaluator compatibility | EyeBench submodule metadata and CopCo prepared data | alignment audit, official/fold/full-data comparison rows | official alignment validation passed | official subset blocked; EyeBench-fold full-feature intersection complete | complete with official blocker | results/official_eyebench_alignment_v1_20260506_2232/ | diagnostic and benchmark framing |
| OfficialEyeBenchSOTACheck v1 | official_eyebench_sota_check_v1 | 2026-05-06 23:41 | not centralized in indexed manifest | test whether an official EyeBench leaderboard claim can be made | official alignment audit and EyeBench submodule | official environment/data/baseline blocker reports | official SOTA check validation passed | official processed data and environment blocked; no official claim allowed | blocked official result; complete blocker record | results/official_eyebench_sota_check_v1_20260506_2341/ | blocked/diagnostic result |
| D3 EyeBench own-method score-max v2 | d3_eyebench_own_method_score_max_v2 | 2026-05-22 analysis sync | not centralized in indexed manifest | evaluate reduced official-compatible trial-level D3_Lite candidates | official-compatible feature intersection | trial metrics and candidate leaderboard | score-max validation passed in analysis record | candidate_0000 anchor retained for no-improvement decision | complete | analysis/d3_eyebench_own_method_score_max_v2/ | official-compatible stress test |
| OperatingPointAdaptation v1 | operating_point_adaptation_v1 | 2026-05-23 analysis sync | not centralized in indexed manifest | separate probability, calibration, threshold, reader aggregation, and oracle diagnostics | existing prediction outputs | fixed, legal, oracle, calibration, and aggregation metrics | operating-point validation passed | test-oracle thresholds marked diagnostic and not clean evidence | complete | analysis/operating_point_adaptation_v1/ | diagnostic analysis |
| D3OnlineTargetedOptimization v1 | online_targeted_optimization_v1 | 2026-05-23 analysis sync | not centralized in indexed manifest | evaluate prefix datasets, online probabilities, accumulation, and stopping | Label Release v1.1 prepared dataset | prefix data, nested predictions, online probabilities, locked rows | v1 validation passed | selected no_stop/full-sequence candidate; later audited as offline-like | complete but deprecated/fast for online claim | analysis/d3_online_targeted_optimization_v1/ | historical and diagnostic online source |
| D3OnlineTargetedOptimization v2 or audit-rerun | online_targeted_optimization_v2 | 2026-05-23 analysis sync | not centralized in indexed manifest | audit v1 and rerun strict online selection categories | v1 artifacts and prepared online prefix data | strict final models, per-prefix curves, legal calibration, error analysis | v2 validation passed | offline/all evidence remains strongest; online rows separated by budget | complete | analysis/d3_online_targeted_optimization_v2/ | secondary online/offline analysis |
| D3ModelEvidenceVault v1 | d3_model_evidence_v1 | 2026-05-23 07:47 output | recorded in vault source trace | curate model evidence into source-traced internal vault | all previous result directories | v1 evidence vault | vault validation passed | first canonical D3 evidence collection | complete | analysis/d3_model_evidence_v1/ | evidence source |
| D3ModelEvidenceVault v1.1 | d3_model_evidence_v1_1 | 2026-05-23 10:10 output | repository commit 4d6604eeb04a8fe64cfca434b9fe2ff247a71373 in status | expand canonical metrics, source trace, result scope, and claim status | v1 vault and all listed source artifacts | v1.1 evidence vault and machine-readable manifests | vault validation passed | 486 canonical metric rows and 1 unresolved discrepancy | complete | analysis/d3_model_evidence_v1_1/ | primary source for this master record |
| Deep literature review if present | deep_literature_review | not present at build time | not applicable | source related-work details if the directory exists | analysis/deep_literature_review/ | none indexed because directory is missing | missing source recorded | missing; no values fabricated | missing source | analysis/deep_literature_review/ | missing/historical placeholder |
## SECTION 3 — Data inventory and dataset versions

### A. Full prepared CopCo / CopCo-Dyslexia-style dataset

The full prepared data are the project-specific CopCo derived57 prepared dataset with operational reader labels. Source paths include `results/feature_release_v1_20260505_2155/`, `results/label_release_v1_1_20260506_0041/`, and `results/label_release_v1_1_20260506_0041/prepared_dataset/`.

| item | value | source |
| --- | --- | --- |
| participant count | 57 | label validation |
| dyslexia-labeled count | 19 | participant label report |
| typical/control count | 38 | participant label report |
| word-level gaze rows | 335203 | feature table summary |
| stimulus word rows | 31986 | feature table summary |
| sentence rows | 1986 | feature table summary |
| paragraph rows | 452 | feature table summary |
| participant-level rows | 57 | prepared manifest |
| DFM LM rows | 335203 | join validation |
| segmentation boundary rows | 31986 | label validation |
| segmentation word rows | 31986 | label validation |
| quality label rows | 335203 | label validation |
| split label rows | 3591 | label validation |
| LM missing rate | 0.0053 | label validation |
| embedding missing rate | 0.0420 | label validation |
| parser missing rate | 0.0000 | label validation |

Boundary-label distribution:

| boundary_type | count |
| --- | --- |
| C#C | 14055 |
| C#V | 6280 |
| V#C | 6412 |
| V#V | 2406 |
| unknown | 847 |

DFM alignment and parser status: DFM alignment status is `passed` with warning counts {'non_special_token_unassigned': 444}. Parser backend is `surface_heuristic` with preferred backend `dacy` and no true syntax claim.

Embedding models recorded:

| label | model_id | sentence_rows | paragraph_rows | embedding_dim |
| --- | --- | --- | --- | --- |
| dfm_sentence_encoder | KennethEnevoldsen/dfm-sentence-encoder-large | 1986 | 452 | 1024 |
| e5_large | intfloat/multilingual-e5-large | 1986 | 452 | 1024 |

### B. EyeBench-related data/protocol settings

Official EyeBench status is separated from EyeBench-style and official-compatible internal results. The EyeBench submodule commit recorded by the alignment audit is `ce87f38a3083aeed029c255716a1a51e6ae51167`. Official processed CopCo data were not present, the official environment was not import-ready, and the official evaluator did not run. EyeBench-fold full-feature intersection rows completed, but they are not official processed-data results. Full-data EyeBench-style rows are internal benchmark-relative rows.

| mode | model | claim_type | official_mode | exact_folds | exact_processed_data |
| --- | --- | --- | --- | --- | --- |
| official_eyebench_subset | D3_EyeBench_Lite | official_attempt_failed | False | False | False |
| eyebench_folds_full_feature_intersection | D3_FullFeature_EyeBenchFolds | EyeBench-fold-aligned_full-feature_non-official | False | True | False |
| full_data_eyebench_style | D3_FullData_EyeBenchStyle | internal_EyeBench-style_benchmark-relative | False | False | False |

### C. Full-data versus reduced official-compatible data

Full data include participant-level aggregates, residual gaze summaries, DFM surprisal/entropy, DFM sensitivity profiles, segmentation features, parser fallback features, and embedding-derived compact semantic features. The official-compatible reduced variant is trial-level and constrained by the intersection with EyeBench fold/protocol structure; it lacks the full reader profile scope and is not equivalent to the full-record reader-profile model.

### D. Online prefix data

Online prefix data use cumulative evidence only. Prefix types include `word_count_prefix`, `chronological_prefix`, `trial_or_text_prefix`, `speech_prefix`, and sequence stopping rows. Budgets recorded in v1/v2 include 50, 100, 250, 500, 1000, one to three trials/texts/speeches, `all`, and sequence-stop rows. Nested split roles are train_fit, inner_oof, calibration, and outer_test; legal threshold/calibration rows are selected without outer-test label thresholds.

| item | value | source |
| --- | --- | --- |
| v1 prefix rows | 1145 | d3_online_targeted_optimization_validation_report.json |
| v1 nested prediction rows | 306376 | d3_online_targeted_optimization_validation_report.json |
| v1 online probability rows | 243656 | run_manifest.json |
| v1 accumulation rows | 1232 | run_manifest.json |
| v1 stopping rows | 2128 | run_manifest.json |
| v1 oracle rows | 3785 | run_manifest.json |
| v1 error trajectory rows | 4222 | run_manifest.json |
| v2 per-prefix rows | 1232 | run_manifest.json |
| v2 candidate rows | 52 | run_manifest.json |
| v2 final model rows | 24 | run_manifest.json |
| v2 locked rows | 23 | run_manifest.json |
| v2 audit rows | 17 | run_manifest.json |
| v2 error rows | 7076 | run_manifest.json |

Additional source detail: feature join validation reports 335203 full word rows and 335203 word rows with DFM LM columns joined.
## SECTION 4 — Split protocols and evaluation regimes

All clean predictive evaluations use participant-aware or protocol-aware split roles. Random word-level train/test splitting is excluded because the target is reader-level.

| split | what_it_tests | train_test_disjoint | participant_disjoint | text_disjoint | completed | folds_skipped | source_file | role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leave_one_participant_out / LOPO | reader-level generalization to each held-out participant | participant-disjoint | True | False | True | 0 | results/label_release_v1_1_20260506_0041/analysis/label_analysis/split_label_report.md; analysis/autoresearch_v1/tables/final_model_metrics_table.csv | main |
| participant_grouped_kfold | participant-grouped cross-validation without participant leakage | participant-disjoint by fold | True | False | True | not recorded as skipped in v2 final rows | feature release splits and online v2 strict_final_models.csv | secondary |
| unseen_reader | held-out readers with seen text distribution allowed | reader-disjoint | True | False | True | 0 | BenchmarkBridge, OfficialEyeBenchAlignment, online v2 | secondary/benchmark |
| unseen_text | held-out speeches/texts with reader overlap allowed by protocol | text-disjoint | False | True | True | 0 | BenchmarkBridge, OfficialEyeBenchAlignment, online v2 | secondary/diagnostic; unresolved specialist rows |
| unseen_reader_and_text | simultaneous held-out readers and texts | reader-disjoint and text-disjoint | True | True | True | 0 | BenchmarkBridge, OfficialEyeBenchAlignment, online v2 | secondary/benchmark |
| text_balanced_unseen_reader | reader-disjoint split with deterministic text-exposure balancing | reader-disjoint | True | False | True | not recorded as skipped in v2 final rows | BenchmarkBridge and online v2 | diagnostic/secondary |
| leave_one_speech_out | speech/text holdout sensitivity | speech/text-disjoint | False | True | available in split sources where enabled | not centralized in canonical metric table | results/feature_release_v1_20260505_2155/splits/leave_one_speech_out.csv | diagnostic/historical |
| online prefix splits | same outer regimes under evidence prefixes | outer-test rows excluded from fit/threshold/calibration | depends on outer regime | depends on outer regime | True | not collapsed; see online manifests | analysis/d3_online_targeted_optimization_v1/ and v2/ | secondary/diagnostic online |
| official-compatible split handling | EyeBench fold metadata compatibility | official fold roles where available | official dependent | official dependent | fold-aligned intersection completed; official subset blocked | official subset skipped | analysis/official_eyebench_alignment_v1/ | blocked official / diagnostic |

Nested online roles are train_fit for fitting model parameters, inner_oof for candidate selection and legal thresholds, calibration for fitted calibration where available, and outer_test for final clean evaluation only. Oracle/test-label threshold rows are explicitly diagnostic.
## SECTION 5 — Feature families and how they were computed

This section records feature construction families at the level needed for future method explanation. It does not add new features or perform feature selection.

| family | details | source |
| --- | --- | --- |
| A. Gaze features | first fixation duration, first-pass duration, go-past time, total fixation duration, fixation count, skipping/fixated indicator, landing-position fields where available, saccade/source-derived features where present, participant aggregates, residual gaze features, and online cumulative gaze summaries. | Feature Release v1; participant sensitivity dictionaries; residualization reports |
| B. Classical text features | word length, sentence length, word position, punctuation, capitalization/digit flags, frequency/log frequency, readability/surface components, and text-level exposure controls. | Feature Release v1 feature dictionary and modeling tables |
| C. Parser or parser-fallback features | parser_status is surface_heuristic_fallback. These features are surface and morpho-orthographic heuristics, not true syntactic parses. Parser syntax claims are not supported. | parser_diagnostics.json; quality_label_card_v1.md |
| D. Segmentation / boundary-opacity features | C#C, C#V, V#C, V#V, other/unknown; deterministic orthographic vowel/consonant logic using Danish vowels a/e/i/o/u/y/ae/oe/aa equivalents in source spelling plus Danish letters; previous-boundary, next-boundary, sentence-level rates, and V#V indicators. These are stimulus-level linguistic labels and secondary interpretability features. | segmentation_label_card_v1.md and label release reports |
| E. DFM language-model features | danish-foundation-models/dfm-decoder-open-v0-7b-pt with tokenizer from the causal LM; causal scoring produces word-level surprisal and entropy; subword values are aligned and aggregated to words with warning/missingness tracked. DFM exposure features summarize average stimulus difficulty. DFM sensitivity features and residualized DFM gaze features summarize each reader's gaze-cost slope after residualizing gaze against stimulus/text covariates. | DFM alignment report, dfm_feature_summary, residualization reports |
| F. Embedding features | KennethEnevoldsen/dfm-sentence-encoder-large and intfloat/multilingual-e5-large sentence/paragraph embeddings, compact semantic features, and missingness indicators. These are context features, not the main D3 residual-gaze signal. | embedding manifest and feature dictionary |
| G. Online prefix features | cumulative residual gaze features, cumulative DFM exposure, cumulative DFM residual summaries, prefix stability, uncertainty features, evidence budgets, and stable_enough_for_prediction flags. Prefix features do not use future evidence. | docs/d3_online_targeted_optimization_v1.md and v2 artifacts |
| H. Prohibited or excluded features | participant_id as predictor, speech_id/text_id as direct predictors, future online evidence, exposure-count variables in primary models, and test-label thresholds in clean metrics are excluded. Oracle rows are diagnostic only. | prohibited_feature_policy.md, leakage controls, split policy |

Residual gaze construction: residualizers are fit inside the relevant training fold using stimulus/text predictors such as word length, log frequency, DFM surprisal, DFM entropy, sentence length, word position, segmentation labels, and missingness flags. Reader group, participant ID, speech ID, text ID, labels, and targets are not residualizer predictors. The resulting participant features are aggregates and slopes of residual gaze costs against DFM predictability features.
## SECTION 6 — Language models and NLP tools used

| model_tool | role | input_output | status | source_phase | notes |
| --- | --- | --- | --- | --- | --- |
| danish-foundation-models/dfm-decoder-open-v0-7b-pt | primary causal LM for surprisal and entropy | Danish text context to subword/word-level surprisal and entropy | completed | Feature Release v1 | base/pretrained causal LM; instruction tuning not used for token likelihood |
| google/gemma-2-9b | attempted sensitivity LM comparison | would have produced alternative causal-LM features | blocked | Feature Release v1 | blocked by gated Hugging Face access; no values fabricated |
| KennethEnevoldsen/dfm-sentence-encoder-large | Danish sentence/paragraph embeddings | sentences/paragraphs to 1024-dimensional embeddings and semantic summaries | completed | Feature Release v1 | embedding context features, not causal surprisal |
| intfloat/multilingual-e5-large | multilingual sentence/paragraph embeddings | sentences/paragraphs to 1024-dimensional embeddings and semantic summaries | completed | Feature Release v1 | secondary semantic feature source |
| DaCy / spaCy | preferred parser backend | sentence tokens to linguistic features | attempted/fallback | Feature Release v1 | preferred backend unavailable; backend error recorded as libtorch_cuda.so: failed to map segment from shared object |
| surface_heuristic parser fallback | parser-fallback feature generation | surface/token features and heuristic covariates | completed | Feature Release v1 | not true syntax; parser-syntax claims prohibited |

Base/pretrained causal language models are used for surprisal because token-level likelihood under left context is the needed quantity. Instruction-tuned/chat models are not used for surprisal because instruction alignment changes the task objective and does not provide the same stable next-token probability interpretation.

DFM exposure differs from DFM sensitivity: exposure summarizes the DFM predictability profile of the text a reader saw, while sensitivity summarizes how that reader's residual gaze costs vary with DFM surprisal/entropy. DFM residual gaze features are produced by fitting fold-local residualizers for gaze outcomes against stimulus/text covariates, then aggregating residual means and residual slopes with respect to DFM predictability features at the participant level.
## SECTION 7 — Model family taxonomy

| model_name_internal | public_description | data_scope | evaluation_level | split_regime | feature_family | calibrator | threshold_policy | accumulator | stopping_policy | key_metrics | source_files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D3_dfm_residual_gaze_only | full-record residualized predictability-sensitive reader-profile model | full prepared CopCo | reader | LOPO | DFM residual gaze sensitivity | none/fixed probability output | fixed 0.5 in final metrics | not applicable | not applicable | AUROC 0.8947; BA 0.8421; PR-AUC 0.8641 | analysis/autoresearch_v1/tables/final_model_metrics_table.csv |
| D1_dfm_exposure_only | language-model exposure-only ablation | full prepared CopCo | reader | LOPO/ablation | DFM exposure only | not recorded | fixed/evaluation default | not applicable | not applicable | AUROC 0.4238 | dfm_exposure_vs_sensitivity_table.csv |
| D2_dfm_sensitivity_only | language-model sensitivity-only ablation | full prepared CopCo | reader | LOPO/ablation | DFM sensitivity | not recorded | fixed/evaluation default | not applicable | not applicable | AUROC 0.8892 | dfm_exposure_vs_sensitivity_table.csv |
| D3_FullData_EyeBenchStyle | full-data internal EyeBench-style reader-aggregated model | full prepared CopCo | reader aggregated | unseen_reader/unseen_text/unseen_reader_and_text | D3 residual gaze profile | not primary | fixed/evaluation default | reader aggregation | not applicable | see BenchmarkBridge internal rows | analysis/benchmark_bridge_v1/tables/copco_typ_benchmark_comparison.csv |
| D3_EyeBench_Lite candidate_0000 | reduced official-protocol-compatible trial-level variant | official-compatible feature/fold subset | official trial-level fold mean | unseen_reader/unseen_text/unseen_reader_and_text | reduced D3_Lite exact features | none | fixed 0.5 | not applicable | not applicable | anchor rows; no locked candidate improved anchor | analysis/d3_eyebench_own_method_score_max_v2/trial_metrics.csv |
| OperatingPointAdaptation rows | probability-first operating-point diagnostic | existing prediction outputs | reader/trial depending source | multiple | probability outputs | fixed, fitted where legal, or oracle diagnostic | fixed 0.5, legal inner, or test-oracle diagnostic | reader probability aggregation where available | not applicable | 20 fixed-threshold rows | analysis/operating_point_adaptation_v1/ |
| best_online_late_accumulation / mid / early | fixed-budget sequential reader-evidence models | online prefix data | reader | participant/text online regimes | DFM residual plus uncertainty prefix features | identity, isotonic, sigmoid depending candidate | fixed 0.5 or inner_cv_regime_specific | mean_probability or learned_meta_aggregator | fixed_budget | 24 strict final rows | analysis/d3_online_targeted_optimization_v2/strict_final_models.csv |
| best_online_stopping_detector | adaptive stopping diagnostic | online sequence/prefix data | reader | participant/text online regimes | DFM residual plus uncertainty prefix features | identity | inner_cv_global | learned_meta_aggregator | coverage_constrained_stop | stopping_not_ready status | online v2 strict_final_models and stopping result summaries |
| unseen_text_rescue_04 / unseen_text_rescue_05 | unseen-text specialist diagnostic/rescue variants | online unseen_text split | reader | unseen_text | all_allowed_strict_online | identity or sigmoid | inner_cv_regime_specific | entropy_weighted or mean_probability | fixed_budget | 4 rescue candidate rows; discrepancy preserved | unseen_text_rescue_candidates.csv |

Reduced D3_Lite trial-level rows are not the full method because their data scope, feature scope, evaluation unit, and official-compatible constraints differ from the full-record reader-profile model.

Representative source snippets:

D3_Lite anchor trial rows:

| candidate_id | split_name | evaluation_level | n_predictions | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_0000 | unseen_reader | official_trial_level_fold_mean | 3554 | 0.8085 | 0.5614 | 0.7274 | 0.6767 | 0.1904 |
| candidate_0000 | unseen_text | official_trial_level_fold_mean | 3554 | 0.8319 | 0.5434 | 0.7341 | 0.6751 | 0.1871 |
| candidate_0000 | unseen_reader_and_text | official_trial_level_fold_mean | 1228 | 0.7154 | 0.5650 | 0.6342 | 0.6223 | 0.2191 |

BenchmarkBridge full-data row:

| model | unseen_reader_balanced_accuracy | unseen_text_balanced_accuracy | unseen_reader_text_balanced_accuracy | unseen_reader_AUROC | unseen_text_AUROC | unseen_reader_text_AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| D3_dfm_residual_gaze_only | 0.8158 | 0.7444 | 0.7458 | 0.8961 | 0.8285 | 0.8542 |
## SECTION 8 — Full prepared CopCo results

### A. Offline full-record reader-profile result

| metric | value | source |
| --- | --- | --- |
| AUROC | 0.8947 | final_model_metrics_table.csv |
| PR-AUC | 0.8641 | final_model_metrics_table.csv |
| balanced accuracy | 0.8421 | final_model_metrics_table.csv |
| macro F1 | 0.8421 | final_model_metrics_table.csv |
| Brier | 0.1159 | final_model_metrics_table.csv |
| permutation p-value | 0.0010 | phase4_confirmatory/permutation_results.csv |
| bootstrap AUROC CI | [0.7765, 0.9841] | phase4_confirmatory/bootstrap_results.csv |
| predictions | 57 | final_model_metrics_table.csv |
| usable folds | 57 | final_model_metrics_table.csv |

### B. DFM exposure vs sensitivity ablation

| feature_group | n_features | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score | n_predictions | skipped_folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1_dfm_exposure_only | 3 | 0.4238 | 0.3685 | 0.4474 | 0.4389 | 0.2684 | 57 | 0 |
| D2_dfm_sensitivity_only | 16 | 0.8892 | 0.8611 | 0.8421 | 0.8421 | 0.1130 | 57 | 0 |
| D3_dfm_residual_gaze_only | 12 | 0.8947 | 0.8641 | 0.8421 | 0.8421 | 0.1159 | 57 | 0 |
| D4_dfm_exposure_plus_sensitivity | 19 | 0.8726 | 0.8561 | 0.8158 | 0.8074 | 0.1206 | 57 | 0 |

Interpretation boundary: exposure-only rows are an ablation against text/LM exposure, not a clinical or causal test. The recorded pattern is that DFM sensitivity/residual gaze rows substantially exceed exposure-only rows.

### C. Phase 3 exploration

Phase 3 records an exploratory best participant-level model with `D_dfm_exposure_and_sensitivity` logistic regression under LOPO AUROC 0.9058, permutation p-value 0.0099, and bootstrap AUROC interval [0.8162, 0.9798]. It also records word-level secondary ladder outputs, reader-group interactions as exploratory, and standalone segmentation main-effect support as not retained.

| split_name | feature_group | model | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score | n_predictions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leave_one_participant_out | A_raw_gaze_aggregates | logistic_regression | 0.8338 | 0.7622 | 0.7632 | 0.7632 | 0.1608 | 57 |
| leave_one_participant_out | A_raw_gaze_aggregates | linear_svm | 0.8172 | 0.7578 | 0.7500 | 0.7467 | 0.1778 | 57 |
| leave_one_participant_out | A_raw_gaze_aggregates | random_forest | 0.8573 | 0.8055 | 0.7500 | 0.7467 | 0.1452 | 57 |
| participant_grouped_kfold | A_raw_gaze_aggregates | logistic_regression | 0.8227 | 0.7602 | 0.7368 | 0.7304 | 0.1730 | 57 |
| participant_grouped_kfold | A_raw_gaze_aggregates | linear_svm | 0.8075 | 0.7612 | 0.7763 | 0.7799 | 0.1775 | 57 |
| participant_grouped_kfold | A_raw_gaze_aggregates | random_forest | 0.8463 | 0.8128 | 0.7500 | 0.7467 | 0.1467 | 57 |
| leave_one_participant_out | B_residual_gaze_aggregates | logistic_regression | 0.8380 | 0.7585 | 0.7632 | 0.7524 | 0.1577 | 57 |
| leave_one_participant_out | B_residual_gaze_aggregates | linear_svm | 0.8809 | 0.8127 | 0.7763 | 0.7799 | 0.1633 | 57 |

### D. Phase 4 confirmatory analysis

Phase 4 selected the cross-fitted residualized DFM gaze-profile model, checked bootstrap/permutation robustness, preserved fold-local residualization, and recorded feature stability. Mixed-effects/interaction summaries are retained as secondary interpretability sources.

| metric | feature_group | model | split_name | observed | n_bootstrap | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| roc_auc | D3_dfm_residual_gaze_only | logistic_regression | leave_one_participant_out | 0.8947 | 2000 | 0.7765 | 0.9841 |
| pr_auc | D3_dfm_residual_gaze_only | logistic_regression | leave_one_participant_out | 0.8641 | 2000 | 0.7083 | 0.9728 |

### E. AutoResearch final selection

AutoResearch records the selected model as D3_dfm_residual_gaze_only logistic regression, with final main support from the locked Phase 4 LOPO metrics, DFM exposure-vs-sensitivity ablation, stress tests, calibration metrics, feature-stability outputs, and reviewer-risk/limitation source tables. It does not convert operational labels into clinical diagnostic claims.

Number registry rows available for future source tracing: 108.
## SECTION 9 — EyeBench-related results

This section separates official reported baselines, internal full-data EyeBench-style comparisons, EyeBench-fold full-feature intersection rows, official-subset/evaluator blockers, and reduced official-compatible trial-level D3_Lite rows. These result types are not interchangeable.

### Subsection A — Published / provided CopCo TYP baselines

The following values are recorded as official reported reference rows, not direct reruns by this repository.

| model | unseen_reader_balanced_accuracy | unseen_reader_AUROC | unseen_text_balanced_accuracy | unseen_text_AUROC | unseen_reader_text_balanced_accuracy | unseen_reader_text_AUROC | average_balanced_accuracy | average_AUROC | metric_basis | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Majority Class / Chance | 0.5030 | 0.5030 | 0.4960 | 0.4960 | 0.5010 | 0.5010 | 0.5000 | 0.5000 | published_fold_mean | Published EyeBench formatted CopCo_TYP test table central value. |
| Reading Speed | 0.5770 | 0.6070 | 0.5490 | 0.5620 | 0.5060 | 0.5090 | 0.5440 | 0.5593 | published_fold_mean | Published EyeBench formatted CopCo_TYP test table central value. |
| Text-Only Roberta | 0.5000 | 0.4700 | 0.5000 | 0.5000 | 0.5000 | 0.5040 | 0.5000 | 0.4913 | published_fold_mean | Published EyeBench formatted CopCo_TYP test table central value. |
| Logistic Regression~\cite{meziere2023using} | 0.7550 | 0.8310 | 0.7660 | 0.8330 | 0.6350 | 0.6890 | 0.7187 | 0.7843 | published_fold_mean | Published EyeBench formatted CopCo_TYP test table central value. |
| SVM~\cite{hollenstein2023zuco} | 0.7070 | 0.7070 | 0.7740 | 0.7740 | 0.6470 | 0.6470 | 0.7093 | 0.7093 | published_fold_mean | Published EyeBench formatted CopCo_TYP test table central value. |
| Random Forest~\cite{makowski2024detection} | 0.6980 | 0.8010 | 0.8150 | 0.9150 | 0.5970 | 0.6590 | 0.7033 | 0.7917 | published_fold_mean | Published EyeBench formatted CopCo_TYP test table central value. |
| AhnRNN~\citep{ahn2020towards} | 0.5000 | 0.5010 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5003 | published_fold_mean | Published EyeBench formatted CopCo_TYP test table central value. |
| AhnCNN~\citep{ahn2020towards} | 0.7770 | 0.8530 | 0.7750 | 0.8570 | 0.6560 | 0.7490 | 0.7360 | 0.8197 | published_fold_mean | Published EyeBench formatted CopCo_TYP test table central value. |
| BEyeLSTM~\citep{reich_inferring_2022} | 0.7190 | 0.7940 | 0.7680 | 0.8500 | 0.6470 | 0.6920 | 0.7113 | 0.7787 | published_fold_mean | Published EyeBench formatted CopCo_TYP test table central value. |
| PLM-AS~\citep{Yang2023PLMASPL} | 0.5520 | 0.5760 | 0.5730 | 0.5850 | 0.5590 | 0.5940 | 0.5613 | 0.5850 | published_fold_mean | Published EyeBench formatted CopCo_TYP test table central value. |
| PLM-AS-RM~\citep{haller2022eye} | 0.6090 | 0.6390 | 0.7160 | 0.8010 | 0.5460 | 0.5500 | 0.6237 | 0.6633 | published_fold_mean | Published EyeBench formatted CopCo_TYP test table central value. |
| RoBERTEye-W~\citep{Shubi2024Finegrained} | 0.7000 | 0.7830 | 0.6850 | 0.7670 | 0.6190 | 0.6820 | 0.6680 | 0.7440 | published_fold_mean | Published EyeBench formatted CopCo_TYP test table central value. |
| RoBERTEye-F~\citep{Shubi2024Finegrained} | 0.6060 | 0.7190 | 0.6030 | 0.7470 | 0.5400 | 0.6330 | 0.5830 | 0.6997 | published_fold_mean | Published EyeBench formatted CopCo_TYP test table central value. |
| MAG-Eye~\citep{Shubi2024Finegrained} | 0.4720 | 0.4590 | 0.4970 | 0.5470 | 0.5140 | 0.5610 | 0.4943 | 0.5223 | published_fold_mean | Published EyeBench formatted CopCo_TYP test table central value. |
| PostFusion-Eye~\citep{Shubi2024Finegrained} | 0.6470 | 0.7310 | 0.6890 | 0.7810 | 0.5700 | 0.6550 | 0.6353 | 0.7223 | published_fold_mean | Published EyeBench formatted CopCo_TYP test table central value. |

### Subsection B — Internal EyeBench-style full-data reader-aggregated comparison

| model | unseen_reader_balanced_accuracy | unseen_text_balanced_accuracy | unseen_reader_text_balanced_accuracy | average_balanced_accuracy | unseen_reader_AUROC | unseen_text_AUROC | unseen_reader_text_AUROC | average_AUROC | evaluation_level | official_mode | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D3_dfm_residual_gaze_only | 0.8158 | 0.7444 | 0.7458 | 0.7687 | 0.8961 | 0.8285 | 0.8542 | 0.8596 | reader_aggregated | False | BenchmarkBridge internal EyeBench-style split. |
| Chance | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | EyeBench reported central value | True | Analytic chance reference. |
| Reading Speed | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | EyeBench reported central value | True | Central value not present in frozen BenchmarkBridge prompt/config. |
| Logistic Regression | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | EyeBench reported central value | True | Central value not present in frozen BenchmarkBridge prompt/config. |
| SVM | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | EyeBench reported central value | True | Central value not present in frozen BenchmarkBridge prompt/config. |
| Random Forest | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | EyeBench reported central value | True | Central value not present in frozen BenchmarkBridge prompt/config. |
| AhnCNN | 0.7770 | not recorded | 0.6560 | 0.7165 | 0.8530 | not recorded | 0.7490 | 0.8010 | EyeBench reported central value | True | Gate central values supplied in BenchmarkBridge v1 request. |
| BEyeLSTM | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | EyeBench reported central value | True | Central value not present in frozen BenchmarkBridge prompt/config. |
| RoBERTEye-W | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | EyeBench reported central value | True | Central value not present in frozen BenchmarkBridge prompt/config. |
| best_reported_baseline | 0.7770 | not recorded | 0.6560 | 0.7165 | 0.8530 | not recorded | 0.7490 | 0.8010 | EyeBench reported central value | True | Gate central values supplied in BenchmarkBridge v1 request. |

The full-data D3 row is internal EyeBench-style and benchmark-relative. It is not an official leaderboard row because exact processed EyeBench data and the official evaluator were not used.

### Subsection C — EyeBench-fold full-feature intersection

| model | mode | claim_type | official_mode | exact_folds | exact_processed_data | unseen_reader_balanced_accuracy | unseen_text_balanced_accuracy | unseen_reader_text_balanced_accuracy | average_balanced_accuracy | unseen_reader_AUROC | unseen_text_AUROC | unseen_reader_text_AUROC | average_AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D3_FullFeature_EyeBenchFolds | eyebench_folds_full_feature_intersection | EyeBench-fold-aligned_full-feature_non-official | False | True | False | 0.7387 | 0.6976 | 0.7110 | 0.7158 | 0.8123 | 0.8141 | 0.7240 | 0.7835 |
| D3_FullData_EyeBenchStyle | full_data_eyebench_style | internal_EyeBench-style_benchmark-relative | False | False | False | 0.8158 | 0.7444 | 0.7458 | 0.7687 | 0.8961 | 0.8285 | 0.8542 | 0.8596 |

Alignment audit overlap: 57 common participants, 19 common dyslexia-labeled participants, 38 common typical/control participants, 32 common texts, 4782 common trials, and 31986 common word rows. The official subset was blocked; the EyeBench-fold full-feature intersection completed with exact folds but not exact processed data.

### Subsection D — Official EyeBench subset/evaluator

Official environment status: blocked_by_environment. Official processed data status: blocked_by_data. Official evaluator status: not run. Baseline reproduction status: skipped. Final blocker category: environment/data. No official leaderboard result was produced because official processed CopCo data and an import-ready official EyeBench environment were absent.

### Subsection E — Reduced official-protocol-compatible trial-level model

D3_Lite is the reduced official-compatible trial-level variant. It is not the full reader-profile method. Candidate_0000 is the anchor; the locked candidate search recorded no-improvement relative to that anchor for the intended decision.

| candidate_id | family | feature_recipe | model_type | threshold_method | calibration_method | split_name | evaluation_level | n_features | n_predictions | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_reader | official_trial_level_fold_mean | 12 | 3554 | 0.8085 | 0.5614 | 0.7274 | 0.6767 | 0.1904 | complete |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_text | official_trial_level_fold_mean | 12 | 3554 | 0.8319 | 0.5434 | 0.7341 | 0.6751 | 0.1871 | complete |
| candidate_0000 | d3_lite_anchor | d3_lite_exact | official_lite_logistic | fixed_0_5 | none | unseen_reader_and_text | official_trial_level_fold_mean | 12 | 1228 | 0.7154 | 0.5650 | 0.6342 | 0.6223 | 0.2191 | complete |
| candidate_0014_3a7538097b | d3_lite_plus_full_official_extension | d3_lite_all | logistic_l2 | inner_balanced_accuracy | sigmoid_cv3 | unseen_reader | official_trial_level_fold_mean | 128 | 3554 | 0.7285 | 0.4768 | 0.6825 | 0.6289 | 0.1491 | complete |
| candidate_0014_3a7538097b | d3_lite_plus_full_official_extension | d3_lite_all | logistic_l2 | inner_balanced_accuracy | sigmoid_cv3 | unseen_text | official_trial_level_fold_mean | 128 | 3554 | 0.8557 | 0.6466 | 0.7765 | 0.7262 | 0.1316 | complete |
| candidate_0014_3a7538097b | d3_lite_plus_full_official_extension | d3_lite_all | logistic_l2 | inner_balanced_accuracy | sigmoid_cv3 | unseen_reader_and_text | official_trial_level_fold_mean | 128 | 1228 | 0.6420 | 0.5056 | 0.5612 | 0.5294 | 0.1961 | complete |
| candidate_0013_936f0c9788 | d3_lite_plus_full_official_extension | d3_lite_all | logistic_elasticnet | inner_balanced_accuracy | sigmoid_cv3 | unseen_reader | official_trial_level_fold_mean | 128 | 3554 | 0.7309 | 0.4798 | 0.6796 | 0.6243 | 0.1487 | complete |
| candidate_0013_936f0c9788 | d3_lite_plus_full_official_extension | d3_lite_all | logistic_elasticnet | inner_balanced_accuracy | sigmoid_cv3 | unseen_text | official_trial_level_fold_mean | 128 | 3554 | 0.8541 | 0.6464 | 0.7779 | 0.7327 | 0.1323 | complete |
| candidate_0013_936f0c9788 | d3_lite_plus_full_official_extension | d3_lite_all | logistic_elasticnet | inner_balanced_accuracy | sigmoid_cv3 | unseen_reader_and_text | official_trial_level_fold_mean | 128 | 1228 | 0.6480 | 0.5103 | 0.5762 | 0.5530 | 0.1952 | complete |
| candidate_0011_c51e744a96 | d3_lite_plus_full_official_extension | d3_lite_all | logistic_l2 | inner_balanced_accuracy | sigmoid_cv3 | unseen_reader | official_trial_level_fold_mean | 128 | 3554 | 0.7178 | 0.4680 | 0.6731 | 0.6194 | 0.1503 | complete |
| candidate_0011_c51e744a96 | d3_lite_plus_full_official_extension | d3_lite_all | logistic_l2 | inner_balanced_accuracy | sigmoid_cv3 | unseen_text | official_trial_level_fold_mean | 128 | 3554 | 0.8566 | 0.6492 | 0.7742 | 0.7123 | 0.1313 | complete |
| candidate_0011_c51e744a96 | d3_lite_plus_full_official_extension | d3_lite_all | logistic_l2 | inner_balanced_accuracy | sigmoid_cv3 | unseen_reader_and_text | official_trial_level_fold_mean | 128 | 1228 | 0.6302 | 0.4910 | 0.5589 | 0.5302 | 0.1991 | complete |

Candidate leaderboard anchor and evaluated test rows:

| candidate_id | family | selection_score | test_evaluated | test_internal_simple_mean_ba | test_internal_simple_mean_auroc | unseen_reader_test_ba | unseen_reader_test_auroc | unseen_text_test_ba | unseen_text_test_auroc | unseen_reader_and_text_test_ba | unseen_reader_and_text_test_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate_0000 | d3_lite_anchor | 0.6985 | True | 0.6985 | 0.7852 | 0.7274 | 0.8085 | 0.7341 | 0.8319 | 0.6342 | 0.7154 |
| candidate_0001_4e51158850 | d3_lite_plus_raw_gaze | 0.7770 | False | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded |
| candidate_0002_c4123ca2fa | d3_lite_plus_raw_gaze | 0.7743 | False | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded |
| candidate_0003_491661e525 | d3_lite_plus_robust_residuals | 0.7494 | False | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded |
| candidate_0004_3f1cbc883e | d3_lite_plus_raw_gaze | 0.7853 | True | 0.6744 | 0.7553 | 0.6978 | 0.7522 | 0.7794 | 0.8451 | 0.5460 | 0.6686 |
| candidate_0005_5f87e66d6c | d3_lite_plus_text_gaze_interactions | 0.7559 | False | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded |
| candidate_0006_ba85f5af0b | d3_lite_calibration_variant | 0.7365 | False | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded |
| candidate_0007_d8cb2de706 | d3_lite_plus_robust_residuals | 0.7334 | False | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded | not recorded |
## SECTION 10 — Online and offline evaluation results

### Subsection A — Offline full-record model

| analysis | split_name | feature_group | model | n_features | n_predictions | usable_folds | skipped_folds | roc_auc | pr_auc | balanced_accuracy | macro_f1 | brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase4_confirmatory_participant_prediction | leave_one_participant_out | D3_dfm_residual_gaze_only | logistic_regression | 12 | 57 | 57 | 0 | 0.8947 | 0.8641 | 0.8421 | 0.8421 | 0.1159 |

### Subsection B — Online fixed-budget evaluation

Prefix types include chronological, word-count, trial/text, speech, all-evidence, and sequence-stop rows. Budgets include 50/100/250/500/1000 words, one to three texts/speeches, all evidence, and learned stopping decisions. v2 per-prefix curves contain 1232 rows.

| split_regime | prefix_type | prefix_value | feature_family | calibrator | threshold | accumulator | n_readers | n_prefix_rows | AUROC | PR-AUC | BA | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| participant_grouped_kfold | chronological_prefix | 100 | all_allowed_strict_online | identity | fixed_0_5 | beta_binomial_posterior | 57 | 57 | 0.7659 | 0.6245 | 0.6842 | 0.1950 |
| participant_grouped_kfold | chronological_prefix | 1000 | all_allowed_strict_online | identity | fixed_0_5 | beta_binomial_posterior | 57 | 57 | 0.8684 | 0.7930 | 0.7895 | 0.1515 |
| participant_grouped_kfold | chronological_prefix | 250 | all_allowed_strict_online | identity | fixed_0_5 | beta_binomial_posterior | 57 | 57 | 0.8338 | 0.7488 | 0.7237 | 0.1755 |
| participant_grouped_kfold | chronological_prefix | 50 | all_allowed_strict_online | identity | fixed_0_5 | beta_binomial_posterior | 57 | 57 | 0.7188 | 0.5083 | 0.6974 | 0.2070 |
| participant_grouped_kfold | chronological_prefix | 500 | all_allowed_strict_online | identity | fixed_0_5 | beta_binomial_posterior | 57 | 57 | 0.8573 | 0.7806 | 0.7632 | 0.1608 |
| participant_grouped_kfold | chronological_prefix | all | all_allowed_strict_online | identity | fixed_0_5 | beta_binomial_posterior | 57 | 57 | 0.8864 | 0.8283 | 0.7763 | 0.1345 |
| participant_grouped_kfold | speech_prefix | 1 | all_allowed_strict_online | identity | fixed_0_5 | beta_binomial_posterior | 57 | 57 | 0.8670 | 0.7949 | 0.7632 | 0.1480 |
| participant_grouped_kfold | speech_prefix | 2 | all_allowed_strict_online | identity | fixed_0_5 | beta_binomial_posterior | 55 | 55 | 0.8619 | 0.7829 | 0.7673 | 0.1474 |
| participant_grouped_kfold | speech_prefix | 3 | all_allowed_strict_online | identity | fixed_0_5 | beta_binomial_posterior | 43 | 43 | 0.7970 | 0.5636 | 0.7136 | 0.1690 |
| participant_grouped_kfold | speech_prefix | all | all_allowed_strict_online | identity | fixed_0_5 | beta_binomial_posterior | 57 | 57 | 0.9003 | 0.8561 | 0.8289 | 0.1271 |
| participant_grouped_kfold | trial_or_text_prefix | 1 | all_allowed_strict_online | identity | fixed_0_5 | beta_binomial_posterior | 57 | 57 | 0.8643 | 0.7939 | 0.7500 | 0.1502 |
| participant_grouped_kfold | trial_or_text_prefix | 2 | all_allowed_strict_online | identity | fixed_0_5 | beta_binomial_posterior | 55 | 55 | 0.8544 | 0.7708 | 0.7395 | 0.1479 |
| participant_grouped_kfold | trial_or_text_prefix | 3 | all_allowed_strict_online | identity | fixed_0_5 | beta_binomial_posterior | 43 | 43 | 0.7970 | 0.5536 | 0.7136 | 0.1722 |
| participant_grouped_kfold | trial_or_text_prefix | all | all_allowed_strict_online | identity | fixed_0_5 | beta_binomial_posterior | 57 | 57 | 0.8947 | 0.8409 | 0.8289 | 0.1303 |
| participant_grouped_kfold | word_count_prefix | 100 | all_allowed_strict_online | identity | fixed_0_5 | beta_binomial_posterior | 57 | 57 | 0.7424 | 0.5937 | 0.7237 | 0.1976 |
| participant_grouped_kfold | word_count_prefix | 1000 | all_allowed_strict_online | identity | fixed_0_5 | beta_binomial_posterior | 57 | 57 | 0.8643 | 0.7891 | 0.7632 | 0.1547 |

### Subsection C — Online targeted optimization

v1 selected `online_d3_0021` with `no_stop`, which the v2 audit records as offline-like because it consumes final sequence evidence. v2 separates best_offline_all_full_evidence, best_online_late_accumulation, best_online_mid_detection, best_online_early_detection, best_online_stopping_detector, and best_unseen_text_specialist.

| final_model | split_regime | n_readers | coverage | mean_words_to_decision | mean_texts_to_decision | evidence_cost | AUROC | PR-AUC | BA | macro_F1 | Brier | candidate_id | calibrator | threshold_policy | accumulator | stopping_policy | prefix_type | prefix_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_offline_all_full_evidence | participant_grouped_kfold | 57 | 1.0000 | 5880.7544 | 4.2456 | 1.0000 | 0.8989 | 0.8271 | 0.8158 | 0.8324 | 0.1186 | v2_candidate_0000 | identity | fixed_0_5 | learned_meta_aggregator | no_stop | trial_or_text_prefix | all |
| best_offline_all_full_evidence | text_balanced_unseen_reader | 57 | 1.0000 | 5880.7544 | 4.2456 | 1.0000 | 0.8989 | 0.8271 | 0.8158 | 0.8324 | 0.1186 | v2_candidate_0000 | identity | fixed_0_5 | learned_meta_aggregator | no_stop | trial_or_text_prefix | all |
| best_offline_all_full_evidence | unseen_reader | 57 | 1.0000 | 5880.7544 | 4.2456 | 1.0000 | 0.8989 | 0.8271 | 0.8158 | 0.8324 | 0.1186 | v2_candidate_0000 | identity | fixed_0_5 | learned_meta_aggregator | no_stop | trial_or_text_prefix | all |
| best_online_late_accumulation | participant_grouped_kfold | 57 | 1.0000 | 1000.0000 | 1.0877 | 0.2780 | 0.7784 | 0.5930 | 0.6842 | 0.6842 | 0.1784 | v2_candidate_0012 | isotonic | fixed_0_5 | mean_probability | fixed_budget | word_count_prefix | 1000 |
| best_online_late_accumulation | text_balanced_unseen_reader | 57 | 1.0000 | 1000.0000 | 1.0877 | 0.2780 | 0.7784 | 0.5930 | 0.6842 | 0.6842 | 0.1784 | v2_candidate_0012 | isotonic | fixed_0_5 | mean_probability | fixed_budget | word_count_prefix | 1000 |
| best_online_late_accumulation | unseen_reader | 57 | 1.0000 | 1000.0000 | 1.0877 | 0.2780 | 0.7784 | 0.5930 | 0.6842 | 0.6842 | 0.1784 | v2_candidate_0012 | isotonic | fixed_0_5 | mean_probability | fixed_budget | word_count_prefix | 1000 |
| best_online_late_accumulation | unseen_reader_and_text | 17 | 1.0000 | 1000.0000 | 1.0000 | 0.7566 | 0.7014 | 0.6299 | 0.5833 | 0.5825 | 0.2581 | v2_candidate_0012 | isotonic | fixed_0_5 | mean_probability | fixed_budget | word_count_prefix | 1000 |
| best_online_late_accumulation | unseen_text | 52 | 1.0000 | 1000.0000 | 1.0000 | 0.7568 | 0.7647 | 0.5804 | 0.7387 | 0.7273 | 0.1801 | v2_candidate_0012 | isotonic | fixed_0_5 | mean_probability | fixed_budget | word_count_prefix | 1000 |
| best_online_mid_detection | participant_grouped_kfold | 57 | 1.0000 | 500.0000 | 1.0175 | 0.2094 | 0.7950 | 0.6504 | 0.7763 | 0.7573 | 0.1596 | v2_candidate_0019 | identity | inner_cv_regime_specific | learned_meta_aggregator | fixed_budget | word_count_prefix | 500 |
| best_online_mid_detection | text_balanced_unseen_reader | 57 | 1.0000 | 500.0000 | 1.0175 | 0.2094 | 0.7950 | 0.6504 | 0.7763 | 0.7573 | 0.1596 | v2_candidate_0019 | identity | inner_cv_regime_specific | learned_meta_aggregator | fixed_budget | word_count_prefix | 500 |
| best_online_mid_detection | unseen_reader | 57 | 1.0000 | 500.0000 | 1.0175 | 0.2094 | 0.7950 | 0.6504 | 0.7763 | 0.7573 | 0.1596 | v2_candidate_0019 | identity | inner_cv_regime_specific | learned_meta_aggregator | fixed_budget | word_count_prefix | 500 |
| best_online_mid_detection | unseen_reader_and_text | 17 | 1.0000 | 500.0000 | 1.0000 | 0.6063 | 0.7639 | 0.6337 | 0.7014 | 0.7018 | 0.2283 | v2_candidate_0019 | identity | inner_cv_regime_specific | learned_meta_aggregator | fixed_budget | word_count_prefix | 500 |
| best_online_mid_detection | unseen_text | 56 | 1.0000 | 500.0000 | 1.0000 | 0.6221 | 0.7696 | 0.6965 | 0.6828 | 0.6937 | 0.1765 | v2_candidate_0019 | identity | inner_cv_regime_specific | learned_meta_aggregator | fixed_budget | word_count_prefix | 500 |
| best_online_early_detection | participant_grouped_kfold | 57 | 1.0000 | 1663.0702 | 1.0000 | 0.3235 | 0.7770 | 0.6489 | 0.7632 | 0.7409 | 0.1788 | v2_candidate_0031 | identity | inner_cv_regime_specific | learned_meta_aggregator | fixed_budget | trial_or_text_prefix | 1 |
| best_online_early_detection | text_balanced_unseen_reader | 57 | 1.0000 | 1663.0702 | 1.0000 | 0.3235 | 0.7770 | 0.6489 | 0.7632 | 0.7409 | 0.1788 | v2_candidate_0031 | identity | inner_cv_regime_specific | learned_meta_aggregator | fixed_budget | trial_or_text_prefix | 1 |
| best_online_early_detection | unseen_reader | 57 | 1.0000 | 1663.0702 | 1.0000 | 0.3235 | 0.7770 | 0.6489 | 0.7632 | 0.7409 | 0.1788 | v2_candidate_0031 | identity | inner_cv_regime_specific | learned_meta_aggregator | fixed_budget | trial_or_text_prefix | 1 |
| best_online_early_detection | unseen_reader_and_text | 17 | 1.0000 | 1659.7059 | 1.0000 | 0.9234 | 0.8333 | 0.7314 | 0.8194 | 0.8211 | 0.2035 | v2_candidate_0031 | identity | inner_cv_regime_specific | learned_meta_aggregator | fixed_budget | trial_or_text_prefix | 1 |
| best_online_early_detection | unseen_text | 57 | 1.0000 | 1663.0702 | 1.0000 | 0.9390 | 0.6884 | 0.6099 | 0.6447 | 0.6531 | 0.1920 | v2_candidate_0031 | identity | inner_cv_regime_specific | learned_meta_aggregator | fixed_budget | trial_or_text_prefix | 1 |
| best_online_stopping_detector | participant_grouped_kfold | 57 | 0.8772 | 214.2800 | 1.0400 | 0.1742 | 0.5177 | 0.3107 | 0.4958 | 0.4876 | 0.2465 | v2_candidate_0043 | identity | inner_cv_global | learned_meta_aggregator | coverage_constrained_stop | sequence | sequence_stop |
| best_online_stopping_detector | text_balanced_unseen_reader | 57 | 0.8772 | 214.2800 | 1.0400 | 0.1742 | 0.5177 | 0.3107 | 0.4958 | 0.4876 | 0.2465 | v2_candidate_0043 | identity | inner_cv_global | learned_meta_aggregator | coverage_constrained_stop | sequence | sequence_stop |
| best_online_stopping_detector | unseen_reader | 57 | 0.8772 | 214.2800 | 1.0400 | 0.1742 | 0.5177 | 0.3107 | 0.4958 | 0.4876 | 0.2465 | v2_candidate_0043 | identity | inner_cv_global | learned_meta_aggregator | coverage_constrained_stop | sequence | sequence_stop |
| best_online_stopping_detector | unseen_reader_and_text | 17 | 0.6471 | 172.7273 | 1.0000 | 0.4920 | 0.7857 | 0.6083 | 0.5000 | 0.3889 | 0.2591 | v2_candidate_0043 | identity | inner_cv_global | learned_meta_aggregator | coverage_constrained_stop | sequence | sequence_stop |
| best_online_stopping_detector | unseen_text | 57 | 0.7544 | 544.7907 | 1.0233 | 0.5762 | 0.6165 | 0.4193 | 0.5597 | 0.5559 | 0.1953 | v2_candidate_0043 | identity | inner_cv_global | learned_meta_aggregator | coverage_constrained_stop | sequence | sequence_stop |
| best_unseen_text_specialist | unseen_text | 52 | 1.0000 | 1000.0000 | 1.0000 | 0.7568 | 0.8639 | 0.8361 | 0.7546 | 0.7204 | 0.2331 | unseen_text_rescue_04 | identity | inner_cv_regime_specific | entropy_weighted | fixed_budget | word_count_prefix | 1000 |

Unseen-text specialist/rescue rows:

| candidate_id | rescue_candidate | split_regime | AUROC | PR-AUC | BA | macro_F1 | Brier | calibrator | threshold_policy | accumulator | prefix_type | prefix_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unseen_text_rescue_00 | text_shift_calibrated | unseen_text | 0.7042 | 0.5018 | 0.7134 | 0.6345 | 0.2124 | sigmoid | inner_cv_regime_specific | learned_meta_aggregator | word_count_prefix | 1000 |
| unseen_text_rescue_03 | text_difficulty_residualized | unseen_text | 0.7597 | 0.6041 | 0.6529 | 0.6233 | 0.1929 | sigmoid | inner_cv_regime_specific | logit_mean | word_count_prefix | 1000 |
| unseen_text_rescue_04 | regime_specific_threshold | unseen_text | 0.8639 | 0.8361 | 0.7546 | 0.7204 | 0.2331 | identity | inner_cv_regime_specific | entropy_weighted | word_count_prefix | 1000 |
| unseen_text_rescue_05 | regime_specific_calibrator | unseen_text | 0.8555 | 0.7928 | 0.8261 | 0.8112 | 0.1488 | sigmoid | inner_cv_regime_specific | mean_probability | word_count_prefix | 1000 |

### Subsection D — Online stopping

Stopping policies include no_stop historical/full-evidence rows, fixed_budget rows, and coverage_constrained_stop sequence rows. v2 records stopping_not_ready as the status for adaptive stopping despite cost reductions in some rows.

| split_regime | stopping_policy | coverage | undecided_rate | mean_words_to_decision | AUROC | PR-AUC | BA | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| participant_grouped_kfold | no_stop_all_evidence | 1.0000 | 0.0000 | 6360.0667 | 0.8800 | 0.8211 | 0.8500 | 0.1271 |
| participant_grouped_kfold | fixed_0_5_at_each_prefix | 1.0000 | 0.0000 | 50.0000 | 0.7000 | 0.5367 | 0.6500 | 0.2101 |
| participant_grouped_kfold | two_sided_confidence_policy | 0.8667 | 0.1333 | 153.8462 | 0.7222 | 0.5701 | 0.6944 | 0.1766 |
| participant_grouped_kfold | inner_cv_balanced_accuracy_policy | 1.0000 | 0.0000 | 6360.0667 | 0.8800 | 0.8211 | 0.8500 | 0.1271 |
| participant_grouped_kfold | cost_sensitive_online_policy | 0.8667 | 0.1333 | 153.8462 | 0.7222 | 0.5701 | 0.6944 | 0.1766 |
| participant_grouped_kfold | target_sensitivity_policy | 1.0000 | 0.0000 | 6360.0667 | 0.8800 | 0.8211 | 0.8500 | 0.1271 |
| participant_grouped_kfold | target_specificity_policy | 1.0000 | 0.0000 | 6360.0667 | 0.8800 | 0.8211 | 0.8500 | 0.1271 |
| participant_grouped_kfold | coverage_constrained_policy | 0.8667 | 0.1333 | 153.8462 | 0.7222 | 0.5701 | 0.6944 | 0.1766 |
| participant_grouped_kfold | no_stop_all_evidence | 1.0000 | 0.0000 | 6360.0667 | 0.8800 | 0.8211 | 0.8000 | 0.1454 |
| participant_grouped_kfold | fixed_0_5_at_each_prefix | 1.0000 | 0.0000 | 50.0000 | 0.7000 | 0.5367 | 0.6500 | 0.2464 |
| participant_grouped_kfold | two_sided_confidence_policy | 1.0000 | 0.0000 | 63.3333 | 0.7000 | 0.5367 | 0.6000 | 0.2788 |
| participant_grouped_kfold | inner_cv_balanced_accuracy_policy | 1.0000 | 0.0000 | 6360.0667 | 0.8800 | 0.8211 | 0.8000 | 0.1454 |

### Subsection E — Error trajectory

v2 error-source analysis records 7076 error rows. Unseen-text errors concentrate in held-out text IDs including 7905, 1323, 7946, 11171, 1125, and 1165 in the source report. The analysis records persistent false positives/false negatives, insufficient-evidence errors, threshold candidates, calibration candidates, and distribution-shift candidates.

v1 locked rows retained for historical comparison:

| split_regime | n_readers | earliness_score | AUROC | PR-AUC | BA | macro_F1 | Brier | candidate_id | stopping_policy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| participant_grouped_kfold | 57 | 0.0000 | 0.8961 | 0.8296 | 0.8684 | 0.8636 | 0.1093 | online_d3_0021 | no_stop |
| text_balanced_unseen_reader | 57 | 0.0000 | 0.8961 | 0.8296 | 0.8684 | 0.8636 | 0.1093 | online_d3_0021 | no_stop |
| unseen_reader | 57 | 0.0000 | 0.8961 | 0.8296 | 0.8684 | 0.8636 | 0.1093 | online_d3_0021 | no_stop |
| unseen_reader_and_text | 17 | 0.0000 | 0.8611 | 0.8671 | 0.8264 | 0.8235 | 0.1506 | online_d3_0021 | no_stop |
| unseen_text | 57 | 0.0000 | 0.7078 | 0.4724 | 0.6842 | 0.6298 | 0.2292 | online_d3_0021 | no_stop |
## SECTION 11 — Result conflicts and unresolved values

Unresolved values are preserved when source rows differ by candidate, metric criterion, split, evaluation level, threshold policy, or diagnostic status. This record does not choose a canonical value for such conflicts.

Source discrepancy table:

| conflict_group_id | metric_name | model_name | candidate_id | split_regime | evaluation_level | source_values | source_files | discrepancy_type | canonical_value_chosen | resolution_status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unseen_text_specialist_v2_conflict | AUROC_and_balanced_accuracy | best_unseen_text_specialist | unseen_text_rescue_04;unseen_text_rescue_05 | unseen_text | reader_aggregated | rescue_04 AUROC 0.8638655462184874 / BA 0.7546218487394958; rescue_05 AUROC 0.8554621848739495 / BA 0.8260504201680672 | analysis/d3_online_targeted_optimization_v2/strict_final_models.csv; analysis/d3_online_targeted_optimization_v2/unseen_text_rescue_candidates.csv; logs/ai_runs/2026-05-23_0643_d3_online_targeted_optimization_v2.md | different_candidate | not recorded | not_resolved_by_design | The source identity differs by candidate and rescue role; v1.1 records both values and does not collapse them into one canonical specialist value. |

Explicit unseen_text specialist discrepancy rows requested for this record:

| source_file | candidate_model | split | evaluation_level | metric | source_value | not_collapsed_reason | paper_direct_use |
| --- | --- | --- | --- | --- | --- | --- | --- |
| analysis/d3_online_targeted_optimization_v2/unseen_text_rescue_candidates.csv | unseen_text_rescue_04 | unseen_text | reader | AUROC / BA | 0.8639 / 0.7546 | candidate, calibrator, accumulator, and criterion differ | no; use only with context and conflict note |
| analysis/d3_online_targeted_optimization_v2/unseen_text_rescue_candidates.csv | unseen_text_rescue_05 | unseen_text | reader | AUROC / BA | 0.8555 / 0.8261 | candidate, calibrator, accumulator, and criterion differ | no; use only with context and conflict note |

Known values from the source file: unseen_text_rescue_04 has AUROC 0.8638655462184874 and BA 0.7546218487394958. unseen_text_rescue_05 has AUROC 0.8554621848739495 and BA 0.8260504201680672. They are not collapsed into one canonical value because the first maximizes/ranks differently from the second and uses a different calibrator/accumulator context.
## SECTION 12 — What each result supports

| result_family | supports | does_not_support | allowed_context | prohibited_context |
| --- | --- | --- | --- | --- |
| full-record reader-profile result | a completed reader-level D3 result on full prepared CopCo under LOPO | clinical diagnosis, screening utility, or external generalization | main internal method/result source | clinical/medical claim or official EyeBench leaderboard claim |
| DFM exposure vs sensitivity ablation | sensitivity/residual gaze rows outperform exposure-only rows in recorded outputs | causal mechanism or complete removal of all text-assignment concerns | ablation/source of method explanation | claim that exposure confounds are impossible |
| BenchmarkBridge full-data comparison | internal EyeBench-style benchmark-relative comparison | official leaderboard status | benchmark-relative internal comparison | official EyeBench result wording |
| official-compatible trial-level stress test | D3_Lite reduced variant behavior under official-compatible constraints | full D3 equivalence or trial-level state-of-the-art claim | stress test/negative or no-improvement record | replace full reader-profile method |
| online fixed-budget evidence | secondary online prefix performance under fixed budgets | ready adaptive stopping detector or official benchmark result | online/offline separation and evidence-cost framing | full-record result equivalence without budget qualification |
| adaptive stopping result | diagnostic stopping-policy status and cost/coverage tradeoffs | stopping detector readiness | diagnostic limitation/status | deployment-ready stopping claim |
| unseen_text result | general unseen_text remains harder; specialist rows are diagnostic | general unseen_text solved by the main model | limitations/conflict section | canonical main result without caveat |
| segmentation/boundary opacity result | orthographic boundary features as secondary interpretability covariates | standalone segmentation-opacity main effect | feature description and secondary interactions | diagnostic label or core claim |
| parser fallback result | surface_heuristic fallback features with parser status recorded | true syntax claims | limitation and feature-missingness explanation | syntactic interpretation |
## SECTION 13 — Public-facing method language

| internal_term | public_facing_term | short_explanation | where_it_may_appear | where_it_should_not_appear |
| --- | --- | --- | --- | --- |
| D3 | residualized predictability-sensitive gaze-profile method | umbrella name for residualized DFM gaze-profile rows. | internal record, methods source map, appendix source notes | standalone paper prose without public description |
| D3 offline | full-record reader-profile model | participant-level model using the full reading record. | internal record, methods source map, appendix source notes | standalone paper prose without public description |
| D3 online | fixed-budget sequential reader-evidence model | online prefix model using only evidence available up to a prefix. | internal record, methods source map, appendix source notes | standalone paper prose without public description |
| D3_Lite | reduced official-protocol-compatible trial-level variant | trial-level reduced feature variant for official-compatible stress tests. | internal record, methods source map, appendix source notes | standalone paper prose without public description |
| BenchmarkBridge | internal EyeBench-style benchmark comparison | full-data reader-aggregated benchmark-relative comparison. | internal record, methods source map, appendix source notes | standalone paper prose without public description |
| OfficialEyeBenchAlignment | official protocol and data-alignment audit | audit of fold/data/evaluator alignment with EyeBench. | internal record, methods source map, appendix source notes | standalone paper prose without public description |
| OperatingPointAdaptation | probability-first operating-point diagnostic | threshold, calibration, and aggregation analysis. | internal record, methods source map, appendix source notes | standalone paper prose without public description |
| OnlineTargetedOptimization | fixed-budget online and stopping-policy evaluation | online prefix, accumulator, and stopping-policy evaluation. | internal record, methods source map, appendix source notes | standalone paper prose without public description |
| D3ModelEvidenceVault | curated model evidence vault | source-traced internal evidence package for D3 results. | internal record, methods source map, appendix source notes | standalone paper prose without public description |

Objective result-expression templates:

| template_name | template |
| --- | --- |
| method_dataset_split_metric_baseline | [method] + [dataset/protocol] + [split regime] + [metric] + [baseline] + [absolute improvement] + [additional advantage] |
| example_full_record | Using residualized predictability-sensitive reader profiles on CopCo TYP under the unseen-reader regime, the model achieved AUROC X and balanced accuracy Y, compared with baseline Z at AUROC A and balanced accuracy B. |
| official_status_guard | When exact official processed EyeBench data, official folds, and the official evaluator are absent, describe the row as internal EyeBench-style or fold-aligned, not official. |

State-of-the-art wording is restricted to objective baseline comparisons with scope labels. A standalone official leaderboard claim is not supported by the recorded official-subset status.
## SECTION 14 — Paper-writing source map

This is a source map only. It does not write final paper text.

| paper_section | master_subsections | evidence_vault_files | canonical_metrics | source_reports |
| --- | --- | --- | --- | --- |
| Introduction | Sections 1, 12, 13 | 05_claim_status, 09_appendix_source_material | main LOPO D3 metrics only as factual context | final_publication_decision_report; reviewer_risk_report |
| Related Work | Sections 9, 13 | canonical_external_baselines.csv | official reported CopCo TYP baseline rows | official alignment/SOTA check reports; deep review if later present |
| Data | Sections 3, 4, 5 | dataset_summary.md, participant_label_summary.md | participant and row counts | feature_release_report.md; label_release_report.md |
| Feature Extraction | Sections 5, 6 | dfm_predictability_features.md, residualization_algorithm.md | feature row counts and missingness | feature_dictionary_v1.md; parser/embedding/DFM reports |
| Method | Sections 5, 6, 7, 13 | 01_algorithm_details/* | model taxonomy rows | Phase 4 and D3 evidence vault algorithms |
| Experiments | Sections 2, 4, 7 | split_policy_summary.md, canonical_model_runs.csv | split/evaluation scope rows | BenchmarkBridge and online target docs |
| Results | Sections 8, 9, 10, 11 | canonical_metrics_long.csv, number registry | full-record, benchmark, online, and conflict rows | AutoResearch, BenchmarkBridge, online v2 |
| Ablations | Sections 8B, 12 | dfm_exposure_vs_sensitivity_summary.md | D1/D2/D3/D4 rows | dfm_exposure_vs_sensitivity_table.csv |
| Online Evaluation | Section 10 | canonical_online_prefix_results.csv, stopping results | v2 strict final and per-prefix rows | online v1/v2 reports |
| Benchmark Comparison | Section 9 | canonical_external_baselines.csv | official/reference/internal split rows | BenchmarkBridge, OfficialEyeBenchAlignment, SOTACheck |
| Limitations | Sections 11, 12, 15 | limitations_factual_notes.md | blocked/unresolved rows | reviewer risk and official blocker reports |
| Appendix | all sections as needed | source manifests and validation files | full metric registry with source paths | all source reports listed in source_trace_manifest.json |
## SECTION 15 — Validation and completeness status

| check | value |
| --- | --- |
| source directories inspected | 21 |
| source files indexed | 853 |
| metric rows used | 486 |
| online prefix metric rows indexed | 2477 |
| online stopping rows indexed | 2133 |
| oracle rows indexed | 3827 |
| source conflicts found | 1 |
| official claim status separated | True |
| full-data and EyeBench-related results separated | True |
| online/offline results separated | True |
| language-model features documented | True |
| model variants documented | True |
| large result files copied | False |
| large result files referenced only | True |
| figures generated | False |
| final paper tables generated | False |
| new experiments run | False |

Missing source directories:

| source_id | path | public_description | status |
| --- | --- | --- | --- |
| deep_literature_review | analysis/deep_literature_review | deep related-work source review, if present | missing |

This master record references large result artifacts by path and does not copy Parquet files, prediction CSVs, model artifacts, figures, or final paper tables.
