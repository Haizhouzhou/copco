# Reviewer-risk summary and submission blocking status.

| risk | risk_level | evidence_available | remaining_weakness | where_to_discuss | blocks_submission |
| --- | --- | --- | --- | --- | --- |
| small participant count | high | 57 participants; bootstrap/influence available | limited external power | main text and limitations | False |
| label provenance | high | label release documents operational labels | not clinical diagnosis | main text | False |
| text/speech exposure imbalance | medium | exposure-only DFM is weak; exposure audits included | not randomized text assignment | main text and appendix | False |
| reading-speed confound | medium | D3 excludes raw speed/global duration | other speed-like residuals possible | main text | False |
| DFM exposure vs sensitivity | low | D1 weak, D2/D3 strong | same dataset only | main text | False |
| leakage risk | low | LOPO and cross-fitted residualization | implementation complexity | appendix | False |
| calibration | medium | Brier and calibration slope recorded | small calibration sample | appendix | False |
| participant influence | medium | leave-one-dyslexia minimum ROC-AUC 0.8801 | small N remains | appendix | False |
| LM alignment warnings | medium | missingness/warnings recorded | DFM warning details need appendix | appendix | False |
| segmentation proxy limitations | medium | orthographic proxy only | not pronunciation-aware | limitations | False |
| parser fallback | medium | surface_heuristic_fallback documented | no true syntax | limitations | False |
| Gemma pending | medium | DFM locked; Gemma deferred | missing LM sensitivity | future work | False |
| no external dataset | high | internal validation strong | external generalization unknown | limitations | False |
| generalization beyond Danish | high | Danish natural reading only | language specificity | limitations | False |
