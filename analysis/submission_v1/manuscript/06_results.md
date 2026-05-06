# Results
The locked participant-level model obtains ROC-AUC 0.8947, PR-AUC 0.8641, balanced accuracy 0.8421, macro F1 0.8421, Brier score 0.1159, calibration intercept -0.5321, and calibration slope 0.8693 with 57 predictions and zero
skipped folds (Table (tab:final-model-metrics), Figure (fig:final-roc),
Figure (fig:final-pr)). The selected feature group is
`D3_dfm_residual_gaze_only`, and the model is a standardized logistic
regression evaluated with LOPO.

The DFM exposure-versus-sensitivity comparison is the key ablation
(Table (tab:dfm-exposure-sensitivity), Figure (fig:dfm-exposure-sensitivity)).
DFM exposure-only is weak (D1 ROC-AUC 0.4238), while DFM sensitivity-only is strong
(D2 ROC-AUC 0.8892). The residual gaze-only model is strongest among the frozen
confirmatory candidates (D3 ROC-AUC 0.8947), and adding exposure variables does not
improve it (D4 ROC-AUC 0.8726). This ordering supports the interpretation that
participant-level predictability sensitivity, not text exposure, drives the result.

Robustness checks support the locked model. The permutation test uses 1,000 valid
permutations and gives p=0.000999. Bootstrap intervals are [0.7765, 0.9841] for
ROC-AUC and [0.7083, 0.9728] for PR-AUC (Table (tab:robustness),
Figure (fig:permutation-null), Figure (fig:bootstrap-auc)). Feature
stability is summarized in Table (tab:feature-stability) and
Figure (fig:feature-stability). Calibration and participant influence are
reported in Table (tab:calibration-influence), Figure (fig:calibration),
and Figure (fig:participant-error).
