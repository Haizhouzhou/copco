# Participant Label Card v1

## Purpose
This card documents the participant-level target label used in Label Release v1.1.

## Allowed Labels
- `dyslexia_labeled`: participant is marked by project metadata as dyslexia-labeled.
- `typical_control`: participant is marked by project metadata as typical/control.
- `uncertain`: source metadata is insufficient or conflicting.

## Provenance
The v1.1 labels are deterministic transformations of existing project metadata. Current wording should remain dyslexia-labeled readers and typical/control readers.

## Intended Use
Use these labels for psycholinguistic group-difference analyses and exploratory participant-level prediction with participant-grouped splits.

## Prohibited Interpretations
Do not use the labels as formal diagnostic status, screening outcome, or biomarker. Do not interpret row-level word examples as independent participant labels.

## Missingness Policy
Participants with uncertain labels remain documented and can be excluded from primary analysis through `include_primary_analysis` while remaining available for sensitivity checks.
