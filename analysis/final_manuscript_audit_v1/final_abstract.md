# Abstract
We analyze Danish natural-reading eye movements for participant-level prediction of
dyslexia-labeled versus typical/control readers. The main analysis uses Danish
Foundation Models (DFM) predictability features to build cross-fitted residualized
gaze profiles, then predicts the participant label with the
locked `D3_dfm_residual_gaze_only` logistic-regression model under
leave-one-participant-out validation. The final model gives ROC-AUC 0.8947, PR-AUC 0.8641, balanced accuracy 0.8421, macro F1 0.8421, Brier score 0.1159, calibration intercept -0.5321, and calibration slope 0.8693 over 57
participants, with 57 predictions and zero skipped folds. A DFM exposure-only model is
weak (ROC-AUC 0.4238), whereas DFM sensitivity-only and residual-gaze models are strong
(ROC-AUC 0.8892 and 0.8947), supporting predictability sensitivity rather than text
exposure as the central signal. Robustness checks include 1,000 valid label
permutations (p=0.000999) and bootstrap intervals of [0.7765, 0.9841] for ROC-AUC and
[0.7083, 0.9728] for PR-AUC. Boundary-opacity interactions are retained as secondary
interpretability evidence. The main limitations are the 57-participant sample,
operational label provenance, and the absence of independent external validation.
