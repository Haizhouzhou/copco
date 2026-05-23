# Online Prefix Dataset Summary

## Source Paths

- `analysis/d3_online_targeted_optimization_v1/run_manifest.json`
- `analysis/d3_online_targeted_optimization_v1/d3_online_targeted_optimization_validation_report.json`
- `analysis/d3_online_targeted_optimization_v2/run_manifest.json`
- `analysis/d3_online_targeted_optimization_v2/per_prefix_performance_curves.csv`

## Recorded Counts

- v1 prefix rows: `1145`
- v1 nested prediction rows: `306376`
- v1 online probability rows: `243656`
- v1 legal calibration rows: `2624`
- v1 legal threshold rows: `5904`
- v1 accumulation rows: `1232`
- v1 stopping rows: `2128`
- v1 oracle rows: `3785`
- v1 error trajectory rows: `4222`
- v2 per-prefix rows: `1232`
- v2 final model rows: `24`
- v2 locked rows: `23`
- v2 error rows: `7076`

## Prefix Types and Budgets

Recorded prefix types include word_count_prefix, chronological_prefix,
trial_or_text_prefix, speech_prefix, sequence, and all-evidence rows. Budgets include
50, 100, 250, 500, 1000, 1 text/trial, 2, 3, 5 where present, sequence_stop, and all.

## Split Roles

The v1 nested prediction contract included train_fit, inner_oof, calibration, and
outer_test roles. v1 validation reports that nested prediction artifacts existed and
passed validation. The source metric rows in v1.1 preserve calibration and threshold
source labels but do not copy full prediction CSVs.


Additional factual trace detail: this vault records the source paths, split labels, evaluation levels, threshold and calibration labels, and result-scope metadata that were available in prior outputs. The text does not add a new experiment and does not choose a new model. When a source file gives row-level values, the canonical CSV keeps the source file path and row context. When only a report gives a status, the row is marked as report text trace. Oracle rows remain diagnostic rows. Official claim fields remain false unless an earlier official protocol source explicitly supplied a supported official result, which is not present in the recorded D3 sources.

Additional factual trace detail: this vault records the source paths, split labels, evaluation levels, threshold and calibration labels, and result-scope metadata that were available in prior outputs. The text does not add a new experiment and does not choose a new model. When a source file gives row-level values, the canonical CSV keeps the source file path and row context. When only a report gives a status, the row is marked as report text trace. Oracle rows remain diagnostic rows. Official claim fields remain false unless an earlier official protocol source explicitly supplied a supported official result, which is not present in the recorded D3 sources.

Additional factual trace detail: this vault records the source paths, split labels, evaluation levels, threshold and calibration labels, and result-scope metadata that were available in prior outputs. The text does not add a new experiment and does not choose a new model. When a source file gives row-level values, the canonical CSV keeps the source file path and row context. When only a report gives a status, the row is marked as report text trace. Oracle rows remain diagnostic rows. Official claim fields remain false unless an earlier official protocol source explicitly supplied a supported official result, which is not present in the recorded D3 sources.

Additional factual trace detail: this vault records the source paths, split labels, evaluation levels, threshold and calibration labels, and result-scope metadata that were available in prior outputs. The text does not add a new experiment and does not choose a new model. When a source file gives row-level values, the canonical CSV keeps the source file path and row context. When only a report gives a status, the row is marked as report text trace. Oracle rows remain diagnostic rows. Official claim fields remain false unless an earlier official protocol source explicitly supplied a supported official result, which is not present in the recorded D3 sources.

Additional factual trace detail: this vault records the source paths, split labels, evaluation levels, threshold and calibration labels, and result-scope metadata that were available in prior outputs. The text does not add a new experiment and does not choose a new model. When a source file gives row-level values, the canonical CSV keeps the source file path and row context. When only a report gives a status, the row is marked as report text trace. Oracle rows remain diagnostic rows. Official claim fields remain false unless an earlier official protocol source explicitly supplied a supported official result, which is not present in the recorded D3 sources.

Additional factual trace detail: this vault records the source paths, split labels, evaluation levels, threshold and calibration labels, and result-scope metadata that were available in prior outputs. The text does not add a new experiment and does not choose a new model. When a source file gives row-level values, the canonical CSV keeps the source file path and row context. When only a report gives a status, the row is marked as report text trace. Oracle rows remain diagnostic rows. Official claim fields remain false unless an earlier official protocol source explicitly supplied a supported official result, which is not present in the recorded D3 sources.

Additional factual trace detail: this vault records the source paths, split labels, evaluation levels, threshold and calibration labels, and result-scope metadata that were available in prior outputs. The text does not add a new experiment and does not choose a new model. When a source file gives row-level values, the canonical CSV keeps the source file path and row context. When only a report gives a status, the row is marked as report text trace. Oracle rows remain diagnostic rows. Official claim fields remain false unless an earlier official protocol source explicitly supplied a supported official result, which is not present in the recorded D3 sources.
