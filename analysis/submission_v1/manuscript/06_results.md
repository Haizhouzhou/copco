# Results
The final participant-level model obtains ROC-AUC 0.8947, PR-AUC 0.8641, balanced accuracy 0.8421, macro F1 0.8421, Brier score 0.1159, calibration intercept -0.5321, and calibration slope 0.8693 with 57 predictions and zero
skipped folds (Table (tab:final-model-metrics), Figure (fig:final-roc),
Figure (fig:final-pr)). The DFM exposure-only model is weak, while sensitivity
and residual gaze models are strong (Table (tab:dfm-exposure-sensitivity),
Figure (fig:dfm-exposure-sensitivity)). This supports the interpretation that
the signal reflects participant-level predictability sensitivity rather than the text
assigned to a participant.

The permutation test uses 1,000 valid permutations and gives p=0.000999.
Bootstrap intervals are [0.7765, 0.9841] for ROC-AUC and [0.7083, 0.9728] for PR-AUC
(Table (tab:robustness), Figure (fig:permutation-null),
Figure (fig:bootstrap-auc)). Calibration and influence summaries are reported in
Table (tab:calibration-influence), Figure (fig:calibration), and
Figure (fig:participant-error). Feature stability is shown in
Table (tab:feature-stability) and Figure (fig:feature-stability).
