# D3 Online Testing Standard v1

## Project-Specific Testing Regimes

Primary regimes:

- `unseen_reader`
- `unseen_reader_and_text`

Additional reported regimes:

- `unseen_text`
- `text_balanced_unseen_reader`
- `participant_grouped_kfold`
- `leave_one_speech_out`, if feasible for prefix rows without mixed evidence

## Project-Specific Online Evidence Budgets

Word-count prefixes:

- first 50 words
- first 100 words
- first 250 words
- first 500 words
- first 1000 words
- all available evidence

Trial/text prefixes:

- first 1 text/trial
- first 2 texts/trials
- first 3 texts/trials
- first 5 texts/trials
- first 10 texts/trials, if available
- all available evidence

Speech prefixes:

- first 1 speech
- first 2 speeches
- first 3 speeches
- first 5 speeches
- all available speeches

Chronological prefixes use the actual reading order when available and are
reported at the same word-count budgets.

## Evidence Cost

For each prefix:

```text
word_cost = n_words_observed / max_words_for_reader
text_cost = n_texts_observed / max_texts_for_reader
combined_evidence_cost = 0.5 * word_cost + 0.5 * text_cost
earliness_score = 1 - combined_evidence_cost
```

Costs are clipped to `[0, 1]`. Missing denominators block the affected row from
stopping-policy scoring.

## Primary Online Target Metric

Model selection uses validation/inner data only. The primary score is:

```text
online_primary_score =
  mean_over_primary_regimes(
    0.35 * reader_AUROC
  + 0.25 * reader_PR_AUC
  + 0.20 * reader_balanced_accuracy
  + 0.10 * (1 - Brier)
  + 0.10 * earliness_score
  )
```

Primary regimes are `unseen_reader` and `unseen_reader_and_text`.

Sensitivity reporting must include the same ranking with earliness removed:

```text
online_no_earliness_score =
  mean_over_primary_regimes(
    0.3888889 * reader_AUROC
  + 0.2777778 * reader_PR_AUC
  + 0.2222222 * reader_balanced_accuracy
  + 0.1111111 * (1 - Brier)
  )
```

## Claim Rules

Clean online results may support a project-specific deployment or supplement
claim. They do not create an official EyeBench SOTA claim unless the official
protocol chain is separately satisfied.

Oracle diagnostics are never used for official or benchmark-relative claims.
