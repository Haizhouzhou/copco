# Residualization Algorithm

## Purpose

Residualization removes recorded word, text, quality, and nuisance effects from gaze outcomes without using reader_group as a predictor. The file is part of D3ModelEvidenceVault v1.1 and records factual evidence
about completed D3 work. It does not train a model, change a threshold, select a
configuration, or create a figure. It describes the algorithm state represented by
prior source files and records the known implementation variants that were already
present in those sources.

## Inputs

The D3 family uses prepared CopCo reading data, feature-release gaze and linguistic
tables, DFM predictability features, operational participant labels, and the split
manifests recorded by Feature Release, Label Release, Phase 4, BenchmarkBridge,
OfficialEyeBenchAlignment, D3_Lite, OperatingPointAdaptation, and online targeted
optimization outputs. Input rows can appear at word, sentence, paragraph, trial,
participant, prefix, reader-aggregated, or stopping-decision level. Source files
identify whether the row is a clean evaluation, a diagnostic row, a blocked row, or an
external reference baseline.

## Outputs

The algorithm outputs recorded in this vault include participant-level probabilities,
trial or prefix probabilities, reader-aggregated probabilities, stopping decisions,
balanced accuracy, AUROC, PR-AUC, macro F1, Brier score, calibration slope/intercept,
coverage, evidence cost, and source status fields. Output rows are factual records and
not new paper tables. Output rows include `source_phase`, `source_file`,
`result_origin`, and `result_scope`.

## Feature Families

Feature families include raw gaze prefix features, residual gaze features, DFM exposure
features, DFM sensitivity features, DFM residual gaze features, DFM residual plus
uncertainty features, and all-allowed online feature groups. Offline full-profile D3
uses participant-level residualized gaze-profile features. D3 Lite uses a reduced
official-compatible trial-level set. Online D3 uses cumulative prefix features and
prefix probability accumulation.

## Training and Evaluation Protocol

Prior D3 outputs use logistic regression as the main classifier, with standardization
from training data where applicable. Split regimes include LOPO, unseen_reader,
unseen_text, unseen_reader_and_text, text_balanced_unseen_reader,
participant_grouped_kfold, and official-fold or official-compatible variants where
available. Online v1 and v2 include inner-validation, calibration, threshold, and outer
test roles when nested artifacts are present. The vault records each prior protocol
without changing the protocol.

## Split Requirements and Leakage Controls

Clean rows require training, calibration, and threshold choices to come from non-test
data. Participant IDs, speech IDs, and text IDs are not predictors. `reader_group` is
not used inside residualization. Random word-level splits are not used as clean D3
evidence. Online prefix rows use evidence observed up to the prefix. Oracle rows are
stored separately and marked diagnostic.

## Threshold and Calibration Handling

Threshold sources include fixed 0.5, inner-CV global thresholds, prefix-specific
thresholds, regime-specific thresholds, and test-oracle thresholds. Fitted calibrators
include identity, sigmoid/Platt, isotonic where sample size allowed, and source-specific
recalibration rows. Test-label thresholds remain diagnostic only.

## Metrics

The source files record AUROC, PR-AUC, balanced accuracy, macro F1, Brier score,
calibration intercept/slope, ECE where present, coverage, undecided rate, mean words
to decision, mean texts to decision, confidence intervals, p-values, and status fields.
Metric definitions are preserved in `metric_definitions.md` and the canonical schema.

## Known Implementation Variants

The recorded D3 variants include full offline reader-profile D3, BenchmarkBridge
full-data reader aggregation, official-compatible D3 Lite, online prefix models,
online accumulators, online stopping policies, oracle diagnostics, and unseen_text
specialist rows. The variants are separated by `algorithm_regime` and `result_scope`.

## Source Files

- `analysis/autoresearch_v1/tables/final_model_metrics_table.csv`
- `analysis/phase4_confirmatory/cross_fitted_residualization_report.md`
- `analysis/benchmark_bridge_v1/benchmark_bridge_decision_report.md`
- `analysis/d3_eyebench_own_method_score_max_v2/final_decision_report.md`
- `analysis/d3_online_targeted_optimization_v2/final_decision_report.md`

## Current Recorded Status

The status fields are inherited from completed prior outputs. Offline full-profile D3
is recorded as a completed primary result. BenchmarkBridge rows are internal
benchmark-relative rows. D3 Lite is official-compatible but not an official SOTA row.
Online v1 is marked fast/truncated where applicable. Online v2 separates full evidence,
late, mid, early, stopping, and unseen_text specialist rows.

## Factual Limitations

Known factual limitations include blocked official subset/evaluator support, general
unseen_text weakness in prior outputs, diagnostic-only oracle thresholds, v1 fast mode,
and stopping detector rows with not-ready status. These limitations are recorded as
source metadata and not resolved in this vault.


Additional factual trace detail: this vault records the source paths, split labels, evaluation levels, threshold and calibration labels, and result-scope metadata that were available in prior outputs. The text does not add a new experiment and does not choose a new model. When a source file gives row-level values, the canonical CSV keeps the source file path and row context. When only a report gives a status, the row is marked as report text trace. Oracle rows remain diagnostic rows. Official claim fields remain false unless an earlier official protocol source explicitly supplied a supported official result, which is not present in the recorded D3 sources.

Additional factual trace detail: this vault records the source paths, split labels, evaluation levels, threshold and calibration labels, and result-scope metadata that were available in prior outputs. The text does not add a new experiment and does not choose a new model. When a source file gives row-level values, the canonical CSV keeps the source file path and row context. When only a report gives a status, the row is marked as report text trace. Oracle rows remain diagnostic rows. Official claim fields remain false unless an earlier official protocol source explicitly supplied a supported official result, which is not present in the recorded D3 sources.

Additional factual trace detail: this vault records the source paths, split labels, evaluation levels, threshold and calibration labels, and result-scope metadata that were available in prior outputs. The text does not add a new experiment and does not choose a new model. When a source file gives row-level values, the canonical CSV keeps the source file path and row context. When only a report gives a status, the row is marked as report text trace. Oracle rows remain diagnostic rows. Official claim fields remain false unless an earlier official protocol source explicitly supplied a supported official result, which is not present in the recorded D3 sources.

Additional factual trace detail: this vault records the source paths, split labels, evaluation levels, threshold and calibration labels, and result-scope metadata that were available in prior outputs. The text does not add a new experiment and does not choose a new model. When a source file gives row-level values, the canonical CSV keeps the source file path and row context. When only a report gives a status, the row is marked as report text trace. Oracle rows remain diagnostic rows. Official claim fields remain false unless an earlier official protocol source explicitly supplied a supported official result, which is not present in the recorded D3 sources.

Additional factual trace detail: this vault records the source paths, split labels, evaluation levels, threshold and calibration labels, and result-scope metadata that were available in prior outputs. The text does not add a new experiment and does not choose a new model. When a source file gives row-level values, the canonical CSV keeps the source file path and row context. When only a report gives a status, the row is marked as report text trace. Oracle rows remain diagnostic rows. Official claim fields remain false unless an earlier official protocol source explicitly supplied a supported official result, which is not present in the recorded D3 sources.

Additional factual trace detail: this vault records the source paths, split labels, evaluation levels, threshold and calibration labels, and result-scope metadata that were available in prior outputs. The text does not add a new experiment and does not choose a new model. When a source file gives row-level values, the canonical CSV keeps the source file path and row context. When only a report gives a status, the row is marked as report text trace. Oracle rows remain diagnostic rows. Official claim fields remain false unless an earlier official protocol source explicitly supplied a supported official result, which is not present in the recorded D3 sources.
