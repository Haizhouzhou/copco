# Dataset Summary

## Source Paths

- `results/label_release_v1_1_20260506_0041/prepared_dataset/analysis_ready_manifest.json`
- `results/feature_release_v1_20260505_2155/feature_release_manifest.json`
- `analysis/d3_online_targeted_optimization_v1/run_manifest.json`
- `analysis/d3_online_targeted_optimization_v2/run_manifest.json`

## Recorded Row Counts

Prepared dataset row counts:

```json
{
  "analysis_ready_participant_level_v1_1": 57,
  "analysis_ready_sentence_level_v1_1": 1986,
  "analysis_ready_word_level_v1_1": 335203
}
```

Prepared join validation values:

```json
{
  "duplicate_participant_word_keys": 0,
  "duplicate_stimulus_word_keys": 0,
  "missing_participant_label_rate": 0.0,
  "missing_quality_label_rate": 0.0,
  "missing_segmentation_word_label_rate": 0.0,
  "source_word_rows": 335203,
  "target_labels_used_during_feature_generation": false,
  "unexpected_row_gain": 0,
  "unexpected_row_loss": 0,
  "word_rows_after_labels": 335203
}
```

Feature release manifest values:

```json
{
  "checksummed_files": 76,
  "run_type": "finalize_feature_release",
  "status": "complete"
}
```

## Connection to D3

The prepared word-level, sentence-level, and participant-level rows are the factual
source for D3 feature construction and participant labels. Word-level rows support
DFM predictability and gaze-feature construction. Participant-level rows support the
offline reader-profile evaluation. Online prefix rows derive from prepared evidence
but are recorded in the online targeted optimization artifacts rather than rebuilt in
this vault.
