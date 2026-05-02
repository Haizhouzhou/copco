# CopCo-Based Dyslexia-Labeled Reader Detection: Research Plan

A scientifically defensible NLP + eye-tracking pipeline for analyzing Danish natural-reading data from CopCo, with the downstream goal of dyslexia-labeled reader detection (not clinical diagnosis).

---

## 1. Data inventory

### 1.1 CopCo stimulus texts — *essential*
- **Why**: every linguistic and LM-derived feature is text-anchored; reproducibility requires the exact stimuli with the exact tokenization CopCo used.
- **Required**: raw text per speech/article, paragraph and sentence boundaries, CopCo's own token spans and word IDs.
- **Quality checks**: Unicode normalization (NFC), whitespace consistency, alignment of CopCo tokenization with our parsing pipeline (DaCy may disagree on clitics, hyphenated forms, numerals).
- **Risks**: tokenization mismatches between CopCo and DaCy/Stanza will propagate as silent join failures; treat alignment as a first-class engineering task.

### 1.2 CopCo word-level gaze features — *essential*
- **Why**: the dependent variables (TRT) and the gaze-only baseline depend entirely on these.
- **Required columns**: `participant_id`, `text_id`, `paragraph_id`, `sentence_id`, `word_id`, `word_form`, FFD, GD, TRT, regression-in, regression-out, fixation count, refixation count, skip flag.
- **Quality checks**: drift correction quality, calibration error per trial, proportion of zero-fixation words, distributional sanity (FFD ≤ GD ≤ TRT, regression rates plausible).
- **Risks**: blinks, track loss, miscalibration; CopCo's own exclusion flags must be respected.

### 1.3 Character-level gaze features — *optional*
- Useful for landing position, within-word refixations, and orthographic neighborhood effects. Required only if we pursue fine-grained orthographic analyses in a later study.

### 1.4 Participant metadata — *essential*
- **Required**: age, sex, education, native-language status, reading habits, vision correction, hand dominance.
- **Risks**: incomplete or self-reported; may not be released with the public CopCo release for privacy reasons.

### 1.5 Dyslexia / control labels — *essential to the project*
- Without confirmed dyslexia-labeled participants in the CopCo subset, the project reduces to a methodology paper.
- **Required**: binary group label plus *provenance*. Use the term **"dyslexia-labeled"** throughout unless formal diagnosis is documented.

### 1.6 Diagnostic provenance — *essential when labels exist*
- **Required**: assessor type (clinician, school psychologist, self-report), instrument (e.g., the Danish national *Ordblindetest*, clinical assessment), date, score where available.
- **Why**: provenance becomes a covariate and a sensitivity-analysis exclusion criterion.

### 1.7 Reading-comprehension scores — *essential*
- Both as a **covariate** (controls for engagement and decoding success) and as a **sanity check** (extreme floor scores flag invalid trials).

### 1.8 Identifier columns — *essential*
- See Section 2.

### 1.9 External Danish lexical resources
- **SubtlexDK** (subtitle frequencies; primary psycholinguistic norm).
- **Danish Gigaword Corpus** or **KorpusDK** (secondary frequency, robustness).
- **DanNet** (Danish WordNet — semantic relations, sense inventory).
- **Den Danske Ordbog (DDO)** lemma list (orthographic neighborhood reference).
- **Pyphen** with `da_DK` dictionary (syllabification fallback).
- **DaCy / Stanza Danish-DDT / UDPipe Danish-DDT** (parsing).

---

## 2. Stable data identifiers

A hierarchical, never-reused scheme. All IDs are strings, not integers, to avoid silent type coercion.

| ID | Format | Scope |
|---|---|---|
| `participant_id` | `P###` | stable across all tables, anonymized |
| `speech_id` | source-derived slug | unique per stimulus text |
| `paragraph_id` | `{speech_id}_p{NN}` | unique within a speech |
| `sentence_id` | `{paragraph_id}_s{NN}` | unique within a paragraph |
| `word_id` | `{sentence_id}_w{NNN}` | unique within a sentence |
| `character_id` | `{word_id}_c{NN}` | only if needed |

**Join discipline**:
- Linguistic features join on `word_id` (text-only, participant-independent).
- Gaze features are keyed by `(participant_id, word_id)`.
- Participant covariates join on `participant_id`.
- Each (participant, word) pair must appear **exactly once** in the final table — enforce with an assertion before writing Parquet.

**Leakage control via IDs**:
- ML splits group on `participant_id` (always) and additionally on `speech_id` for cross-text generalization tests.
- Random word-level splits are forbidden — they will inflate AUC by 10–20 points through participant and text leakage.

---

## 3. Linguistic features

### 3.1 Classical linguistic (text-level)
- word length in characters (after NFC), word length in graphemes
- syllable count (Pyphen `da_DK`; fall back to vowel-cluster heuristic for OOV)
- lemma (DaCy primary)
- POS (Universal Dependencies tagset)
- morphological features (UD `FEATS`)
- dependency relation, head index, signed and absolute dependency distance
- sentence length (tokens, content tokens)
- paragraph position (raw and normalized to [0,1])
- word position in sentence and paragraph (raw and normalized)
- preceding / following punctuation flags

### 3.2 Psycholinguistic
- SubtlexDK Zipf log-frequency (primary)
- Gigaword log-frequency (sensitivity)
- lemma frequency (sum over inflected forms)
- orthographic neighborhood (Coltheart's *N* over a Danish lexicon)
- character bigram and trigram frequencies (Danish corpus)
- Danish compound features: compound depth, head/modifier lengths (Danish has heavy productive compounding — non-trivial and needs a segmenter)
- morphological complexity (count of `FEATS`, derivational depth)
- readability per sentence and paragraph: **LIX**, **LÆS**

### 3.3 LM-derived
- word-level surprisal (primary: dfm-decoder-7B; secondary: gemma-2-9b)
- word-level entropy
- Δ-surprisal: sentence-context vs paragraph-context
- sentence embedding (dfm-sentence-encoder-large; e5-large baseline)
- paragraph embedding (mean-pooled and length-weighted)
- semantic cohesion: cosine(sentence_i, sentence_{i−1})
- topic drift: cosine(sentence_i, paragraph_centroid); rolling drift over a 3-sentence window

### 3.4 Gaze (from CopCo)
FFD, GD, TRT, regression-in, regression-out, skip flag, refixation count.

### 3.5 Participant-level covariates
age, sex, education, vision correction, comprehension score, mean reading speed (words/min), label, label provenance.

---

## 4. Language models

### 4.1 Surprisal and entropy
- **Primary**: `danish-foundation-models/dfm-decoder-open-v0-7b-pt`. Native Danish pretraining; 7B sits cleanly on a single V100; *base* model.
- **Sensitivity**: `google/gemma-2-9b` (base). Different pretraining mix and tokenizer; convergence between the two is evidence that surprisal effects are not model artifacts.
- **Why base, not instruct**: instruction tuning, RLHF, and DPO shift the conditional next-token distribution toward an aligned chat policy. The cognitive-modeling literature on surprisal-as-reading-cost relies on base LMs precisely because their distributions reflect corpus statistics. Mixing instruction-tuned surprisals into this literature breaks comparability and biases effect estimates in unknown directions.

### 4.2 Embeddings
- **Primary**: `KennethEnevoldsen/dfm-sentence-encoder-large` (Danish-tuned).
- **Baseline**: `intfloat/multilingual-e5-large` (strong multilingual reference; demonstrates robustness to embedding space).

### 4.3 POS, dependency, morphology
- **Primary**: **DaCy large**. Best Danish benchmark performance, integrated lemma/POS/dep/morph, actively maintained, drop-in spaCy API.
- **Robustness checks**: Stanza Danish-DDT (more conservative tokenization, UD-trained) and UDPipe Danish-DDT (lighter, deterministic). Run on a 10–20% sample to quantify parser-induced variance in linguistic features. Main pipeline uses DaCy.

### 4.4 Instruction-tuned LLMs
- `mistralai/Mistral-Small-3.1-24B-Instruct` or `google/gemma-3-27b-it`: **optional**, restricted to (a) qualitative annotation of stimuli (register, topic, difficulty rubric) and (b) structured side-annotations (e.g., entity types, abstractness ratings).
- **Never used for surprisal.**
- Treat any instruction-tuned LLM annotations as **ablation features only**. They are not part of the primary scientific story, both because they are unstable across prompts and runs, and because they introduce a black-box dependency that complicates the methods section.
- Choice between Mistral-Small and Gemma-3-27B: prefer Mistral-Small if multilingual robustness matters; prefer Gemma-3-27B if Danish quality is decisive on a small pilot annotation. Run a 100-item agreement check first.

---

## 5. Feature computation plan

### 5.1 Word-level surprisal
1. Tokenize the **paragraph** with the LM tokenizer (`return_offsets_mapping=True`).
2. Forward-pass the paragraph in bf16. If it exceeds context, slide with stride = window/2 and discard predictions in the first half of each non-initial window.
3. For each subword token *t* at position *i*, compute `−log p(t_i | t_<i)`.
4. Use the offset mapping to attribute each subword to its source word (the word's character span).
5. **Aggregate to the word by summing subword `−log p`**, *not* averaging. Summing yields `−log p(word | context)` — a proper joint log-probability. Averaging changes the scale and discards multi-piece evidence; it has no probabilistic interpretation.
6. Persist both `surprisal_sentence_context` (LM sees only the current sentence) and `surprisal_paragraph_context` (LM sees the full paragraph up to that point).

### 5.2 Word-level entropy
- At the position immediately before the first subword of word *w*, compute Shannon entropy `H = −Σ p_v log p_v` over the LM vocabulary.
- This is predictive uncertainty *at word onset*, the quantity that aligns with the gaze-onset moment.
- Optionally also store per-subword entropy summed across the word's subwords as `entropy_sum`.

### 5.3 Subword-to-word aggregation
Always use fast tokenizers with `offset_mapping`. Match by character span, not by string equality (whitespace and casing will bite). Validate on a stratified sample: for 100 random words, manually inspect that the assigned subwords reconstruct the surface form.

### 5.4 Paragraph-context scoring
Run the full paragraph through the LM with BOS, then read off per-word surprisals. Δ-surprisal = `surprisal_sentence − surprisal_paragraph`. This captures discourse-level predictability and is a natural feature for sensitivity to context.

### 5.5 Sentence and paragraph embeddings
- Encode every sentence with both encoders.
- Paragraph embedding = mean-pool of sentence embeddings; also store a length-weighted variant.
- L2-normalize before cosine.

### 5.6 Semantic distance
- `cohesion_adjacent = cos(s_i, s_{i−1})`
- `centroid_similarity = cos(s_i, mean(s_1..s_N))`
- `rolling_drift = cos(s_i, mean(s_{i−3..i−1}))`

### 5.7 Linguistic parser features
Single DaCy pass per paragraph; persist UD-formatted output keyed by `word_id`.

### 5.8 Danish frequency
- SubtlexDK Zipf = `log10(count_per_million) + 3`.
- Gigaword: `log10(count + 1)`.
- Lemma frequency = sum over inflected forms.
- OOV smoothing: add-one per source; flag with `is_oov_*` columns so models can exploit OOV-ness.

---

## 6. Dyslexia-detection modeling plan

| Model | Inputs | Purpose |
|---|---|---|
| **A** | gaze only | baseline; everything must beat this |
| **B** | gaze + classical linguistic | tests whether psycholinguistic structure adds signal |
| **C** | gaze + surprisal/entropy | tests LM-derived predictability signal |
| **D** | gaze + embeddings | tests distributional-semantic signal |
| **E** | full (B+C+D) with feature selection | predictive ceiling |
| **F** | E + LLM-instruct annotations | optional ablation |

**Aggregation**: run each model both at (a) per-word with grouped CV, reporting participant-level AUC by averaging predictions per participant, and (b) per-participant by aggregating features (means, SDs, slopes from per-participant within-subject regressions).

**Classifiers**:
- *Interpretability*: L1-regularized logistic regression; mixed-effects logistic regression.
- *Predictive performance*: **LightGBM** or **XGBoost** — best on tabular gaze + linguistic features at CopCo-like sample sizes.
- Linear SVM and random forest as standard comparators.
- Sequence models (LSTM / small Transformer over fixation sequences): only if participant N is large; otherwise the variance dominates.

**Reporting model**: regularized logistic regression for the headline interpretable result + LightGBM for the performance ceiling. Mixed-effects logistic for hypothesis tests in Section 8.

---

## 7. Cross-validation and leakage control

- **Outer evaluation**: participant-grouped *k*-fold CV (k = 5, stratified on label).
- **Sensitivity**: leave-one-participant-out (LOPO).
- **Text-leakage check**: leave-one-speech-out (LOSO_text). If performance collapses here vs participant-grouped CV, the model is exploiting text-specific cues.
- **Forbidden**: random word- or sentence-level splits across participants.
- **Sensitivity exclusion**: rerun excluding participants with weak label provenance (self-report only).
- **Metrics**: ROC-AUC, **PR-AUC** (groups are usually imbalanced), Brier score, calibration plot, and effect-size estimates from the mixed-effects models.
- **Stability**: report 95% CIs over CV folds and over 5 random seeds for stochastic models.

---

## 8. Statistical analysis

The mixed-effects results are the **scientific contribution**. Classification AUC alone is insufficient at typical CopCo participant counts.

### 8.1 Preregistered hypotheses
- **H1** — dyslexia-labeled readers show a **larger word-length effect** on TRT.
- **H2** — dyslexia-labeled readers show a **steeper frequency effect** on TRT.
- **H3** — dyslexia-labeled readers show **altered surprisal sensitivity** on TRT (direction not preregistered; both larger and noisier are reported in the literature).
- **H4** — entropy effects on TRT differ by group.

### 8.2 Models
Primary specification (lme4 / pymer4 syntax):

```
log(TRT) ~ word_length * group
        + log_freq * group
        + surprisal * group
        + position
        + (1 + word_length + log_freq | participant_id)
        + (1 | word_id)
        + (1 | text_id)
```

- Fit with REML.
- Report fixed-effect estimates with 95% CIs and likelihood-ratio tests for the interaction terms.
- Apply **Holm correction** across the four hypotheses.
- Report random-effect SDs and ICCs to characterize between-participant variability.

### 8.3 Robustness
- Refit with surprisal from gemma-2-9b. Report convergence of interaction estimates.
- Refit excluding weak-provenance participants.
- Refit with linear (untransformed) and log-transformed TRT.

---

## 9. GPU and compute plan (8 × V100)

V100 = 16 GB or 32 GB HBM (verify your variant). 7–9B models in bf16 fit on a single V100; 24–27B models do not, without sharding.

- **dfm-decoder-7B**: bf16 inference, batch by paragraph, one model replica per GPU, paragraphs sharded across 8 GPUs. Use HF `transformers` + `accelerate`, or **vLLM** for throughput. Expected: full CopCo paragraph corpus in single-digit GPU-hours.
- **gemma-2-9b**: same pattern; bf16 single-GPU.
- **dfm-sentence-encoder-large** and **e5-large**: sub-1B params, batch on a single GPU; minutes to encode the entire corpus.
- **No full fine-tuning** of any large LM is required or justified.
- **LoRA / QLoRA**: only if a later study justifies a domain-adapted compact base LM (e.g., a 1–3B Danish base) for a surprisal robustness check. Skip for now.
- **Parsing and classical features**: CPU pipeline with `multiprocessing` across paragraphs; no GPU.
- **Storage**: Parquet, partitioned by `speech_id`. One feature table per granularity (word, sentence, paragraph, participant). Use Arrow-backed `polars` or `pandas` for joins; assert join cardinalities before write.

---

## 10. Expected outputs

| Artifact | Contents |
|---|---|
| `words.parquet` | one row per (participant, word) with all gaze + per-word linguistic + LM features |
| `sentences.parquet` | sentence-level linguistic, embedding, and cohesion features |
| `paragraphs.parquet` | paragraph-level features |
| `participants.parquet` | covariates + per-participant aggregates |
| `data_dictionary.md` | every column: source, units, missingness rule, provenance |
| `pipeline/` | reproducible feature code with config files (Hydra / YAML) |
| `modeling.ipynb` | staged Models A–F with grouped CV |
| `mixed_effects.{R,py}` | fits and reports for H1–H4 |
| `validation_report.md` | leakage checks, CV stability, parser-variance, dfm/gemma agreement |
| `ablation_results.md` | feature-family ablations, including the LLM-annotation ablation |
| `methods.md` | paper-ready methods section draft |

---

## 11. Risks and limitations

- **CopCo is normative Danish reading data** in its main release. The dyslexia-labeled subset (or a separate dyslexia extension) must be confirmed with the data provider, and its labeling protocol described accurately. Without confirmed dyslexia-labeled participants the project reduces to a feature-engineering and methodology paper — which is still publishable but should be framed honestly.
- **Diagnostic-label uncertainty**: distinguish clinician-diagnosed, school-tested (e.g., *Ordblindetest*), and self-reported. Treat provenance as a covariate and a sensitivity-exclusion criterion.
- **Small participant N** inflates variance in both classification metrics and mixed-effects estimates. LOPO must be reported alongside *k*-fold.
- **Participant or text leakage** is the most common error in this literature. Enforce grouped splits with assertions.
- **Overfitting on high-dimensional embedding-derived features**: control via L1, feature selection, and PCA compression to ≤ 64 dims for embedding aggregates.
- **LLM-generated annotations are unstable** across prompts, runs, and decoding seeds. Treat them as ablation only, report inter-run agreement, and never let them drive the headline result.
- **Danish resource ceiling**: SubtlexDK is smaller than English Subtlex; orthographic-neighborhood lexicons are limited; productive Danish compounding is non-trivially handled.
- **Clinical wording**: never claim diagnosis. Never frame the model as a screening tool without prospective clinical validation. Use **"dyslexia-labeled reader"** in all manuscripts and figures.

---

## 12. Final recommendation

### Implement first (in order)
1. **Lock identifier scheme and build the join layer** (Section 2). Highest leverage; everything downstream depends on it.
2. **Build `words.parquet`** with gaze + classical linguistic features (DaCy primary). Validate against published CopCo descriptive statistics.
3. **Compute surprisal and entropy** with dfm-decoder-7B in paragraph context. Replicate on a 10% sample with gemma-2-9b and check correlation.
4. **Fit Models A and B** with participant-grouped CV; establish the gaze-only baseline and the classical-linguistic uplift.
5. **Fit the H1–H3 mixed-effects models**. These are the scientific deliverable.

### Defer
- Embedding-based features beyond simple cohesion and centroid similarity.
- LLM-instruction annotations (Model F).
- Sequence models over fixation sequences.
- LoRA / QLoRA fine-tuning.

### Models to use first
- **Surprisal**: dfm-decoder-7B (primary), gemma-2-9b (sensitivity).
- **Parsing**: DaCy large.
- **Frequency**: SubtlexDK.
- **Classification**: regularized logistic regression + LightGBM.
- **Inference**: mixed-effects logistic and linear models with random effects for participant, word, and text.

### Minimum viable feature table (`words.parquet`)
`participant_id`, `word_id`, `sentence_id`, `paragraph_id`, `speech_id`, `word_form`, `lemma`, `pos`, `dep_rel`, `dep_distance`, `word_length`, `syllables`, `log_freq_subtlex`, `surprisal_dfm`, `entropy_dfm`, `FFD`, `GD`, `TRT`, `regression_in`, `skip`, plus participant covariates joined on `participant_id`.

### Result strong enough for a paper
A credible **group × linguistic-predictor interaction** (H1, H2, or H3) that:
- survives participant-grouped CV and LOPO,
- replicates across the dfm and gemma-2 surprisal pair,
- holds under the weak-provenance exclusion sensitivity,
- is reported with effect-size CIs as the primary evidence and classification AUC/PR-AUC as secondary.

A null classification result combined with a robust interaction effect is still a publishable contribution — it reframes the question from "can we detect dyslexia from gaze + linguistics?" to "how does linguistic processing differ in dyslexia-labeled readers?" — which is the more defensible scientific question given current sample sizes.
