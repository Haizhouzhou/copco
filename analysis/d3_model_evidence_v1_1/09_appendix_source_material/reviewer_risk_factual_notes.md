# Reviewer Risk Factual Notes

## Source Scope

Prior outputs record factual risk areas: official SOTA overclaim risk, unseen_text weakness, trial-level D3_Lite limitations, v1 fast/truncated online status, and stopping detector not-ready status. This summary records completed source output only. It does not create a new
experiment, does not add a new claim, and does not choose among conflicting source
values. Each number in the summary has a corresponding source file path in the
canonical metric files or source-value reconciliation files.

## Source Files

- `analysis/autoresearch_v1/reviewer_risk_report.md`
- `analysis/d3_online_targeted_optimization_v2/final_decision_report.md`

## Models and Variants Recorded

The rows connected to this summary may include full offline D3, BenchmarkBridge D3,
D3 Lite, operating-point diagnostics, online prefix rows, online accumulation rows,
online stopping rows, external reference baselines, oracle diagnostics, blocked rows,
or unseen_text specialist rows. The canonical CSV files keep these roles separate.

## Splits and Evaluation Levels

The relevant split labels are preserved exactly as source fields when available:
LOPO, unseen_reader, unseen_text, unseen_reader_and_text,
text_balanced_unseen_reader, participant_grouped_kfold, validation, test, or unknown.
Evaluation levels include trial_level, prefix_level, reader_aggregated, reader_level,
and stopping_decision. Prefix rows preserve prefix type and prefix value.

## Metrics and Values

Metric values are copied from source files into long-form canonical rows. Values may
include AUROC, PR-AUC, balanced accuracy, macro F1, Brier, calibration
slope/intercept, coverage, undecided rate, mean words to decision, p-values,
confidence intervals, row counts, and status values. When two source values describe
different candidates or contexts, v1.1 records both values in the reconciliation file.

## Result Scope Metadata

`result_origin` records clean evaluation, oracle diagnostic, external reference,
blocked/skipped, source summary, or validation summary. `result_scope` records
primary completed result, secondary completed result, diagnostic completed result,
external reference baseline, blocked result, deprecated or fast run, or unresolved
conflict. These are factual source-role labels.

## Factual Caveats

The source files preserve the known caveats for this result area. Official SOTA fields
remain false for D3 official rows. Oracle rows are not clean benchmark evidence. v1
online rows are marked fast/truncated where the v2 audit recorded fast mode and
truncation. General unseen_text weakness is preserved. Stopping detector status is
preserved as not ready where v2 recorded that status.


Additional factual trace detail: this vault records the source paths, split labels, evaluation levels, threshold and calibration labels, and result-scope metadata that were available in prior outputs. The text does not add a new experiment and does not choose a new model. When a source file gives row-level values, the canonical CSV keeps the source file path and row context. When only a report gives a status, the row is marked as report text trace. Oracle rows remain diagnostic rows. Official claim fields remain false unless an earlier official protocol source explicitly supplied a supported official result, which is not present in the recorded D3 sources.

Additional factual trace detail: this vault records the source paths, split labels, evaluation levels, threshold and calibration labels, and result-scope metadata that were available in prior outputs. The text does not add a new experiment and does not choose a new model. When a source file gives row-level values, the canonical CSV keeps the source file path and row context. When only a report gives a status, the row is marked as report text trace. Oracle rows remain diagnostic rows. Official claim fields remain false unless an earlier official protocol source explicitly supplied a supported official result, which is not present in the recorded D3 sources.

Additional factual trace detail: this vault records the source paths, split labels, evaluation levels, threshold and calibration labels, and result-scope metadata that were available in prior outputs. The text does not add a new experiment and does not choose a new model. When a source file gives row-level values, the canonical CSV keeps the source file path and row context. When only a report gives a status, the row is marked as report text trace. Oracle rows remain diagnostic rows. Official claim fields remain false unless an earlier official protocol source explicitly supplied a supported official result, which is not present in the recorded D3 sources.

Additional factual trace detail: this vault records the source paths, split labels, evaluation levels, threshold and calibration labels, and result-scope metadata that were available in prior outputs. The text does not add a new experiment and does not choose a new model. When a source file gives row-level values, the canonical CSV keeps the source file path and row context. When only a report gives a status, the row is marked as report text trace. Oracle rows remain diagnostic rows. Official claim fields remain false unless an earlier official protocol source explicitly supplied a supported official result, which is not present in the recorded D3 sources.
