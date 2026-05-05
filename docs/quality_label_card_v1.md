# Quality Label Card v1

Quality labels document whether each participant-word row has usable labels and feature coverage for primary and sensitivity analyses.

## Parser Status
`parser_status = surface_heuristic_fallback` means current parser-feature files are surface and morpho-orthographic heuristics. They are usable as heuristic covariates, not as true syntactic annotations.

## Missingness
Missing LM, embedding, parser, participant metadata, and segmentation fields are preserved as boolean flags. Rows are not dropped by the label release.

## Intended Use
Use `include_primary_analysis` and `include_sensitivity_analysis` to construct transparent analysis subsets. Report missingness by reader group before modeling.
