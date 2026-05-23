# Residualization Specification

Residualization removes known word/text/quality effects from gaze outcomes before D3
reader sensitivity features are computed. Residualizers are fit without `reader_group`,
without participant IDs as predictors, and without held-out rows. Cross-fitting is used
where held-out evaluation is required so test evidence is not used to fit residual
models.
