# Segmentation Label Card v1

## Motivation
Segmentation-opacity labels describe stimulus-level orthographic boundary structure for Danish natural-reading analyses. They are not participant target labels.

## Deterministic Algorithm
For each within-sentence word boundary, the final alphabetic character of the previous word and the initial alphabetic character of the current word are classified as C or V. Leading and trailing punctuation or quotes are stripped only for classification; original word strings are preserved.

## Danish Vowel Set
`a e i o u y æ ø å` and uppercase variants.

## Boundary Types And Scores
- `C#C`: score 0, low opacity.
- `C#V`: score 1, medium-low opacity.
- `V#C`: score 2, medium-high opacity.
- `V#V`: score 3, high opacity.
- `other` and `unknown`: no numeric opacity score.

## Examples
- C#C: `tak for`.
- C#V: `kan ikke`.
- V#C: `de går`.
- V#V: `se efter`.

## Limitations
These are orthographic proxy labels, not pronunciation-aware syllabification or phonology. They should be interpreted as deterministic stimulus descriptors for exploratory psycholinguistic modeling.

## Planned Extension
A future pronunciation-aware layer can add phonological boundary labels from a Danish pronunciation lexicon. LLM-generated labels are not part of the core v1.1 release.

## Prohibited Interpretations
Do not treat segmentation-opacity labels as dyslexia labels, diagnostic measures, or evidence of individual reading status.
