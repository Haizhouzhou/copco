# Reviewer Revision Plan

## Reviewer 1: NLP/ML Leakage and Validation

| Criticism | Where Answered | Remaining Weak Point | Exact Revision | Status |
| --- | --- | --- | --- | --- |
| Leakage through held-out participants | Methods, Figure 2, claim ledger C04 | External code audit still useful | State that residual models are fit only on training participants in each LOPO fold | resolved |
| Text exposure confound | Results DFM comparison, text-exposure audit | Non-random text assignment remains | Keep D1 exposure-only failure prominent | partially resolved |
| Model selection after seeing results | Methods and decision framing | Frozen history must be trusted | Use locked model language throughout | resolved |
| Small sample reliability | Limitations, robustness table | No external dataset | Avoid clinical or deployment claims | partially resolved |
| Reproducibility | Reproducibility capsule | Full results dir not committed | Explain ignored generated artifacts | resolved |

## Reviewer 2: Psycholinguistics / Eye Tracking

| Criticism | Where Answered | Remaining Weak Point | Exact Revision | Status |
| --- | --- | --- | --- | --- |
| Word rows are not independent | Methods | None for main task | State participant-level target before LOPO | resolved |
| Residual slopes hard to interpret | Methods, interpretation | Feature names are technical | Keep feature dictionary in supplement | partially resolved |
| Mixed-effects evidence secondary | Interpretation | Some interactions use fallback models | Frame interactions as interpretability only | resolved |
| Calibration with small N | Limitations | Wide uncertainty | Treat calibration as diagnostic | partially resolved |
| Gaze outcome coverage | Methods and supplement | Full equations omitted | Point to supplement feature list | partially resolved |

## Reviewer 3: Danish / Dyslexia / Reading

| Criticism | Where Answered | Remaining Weak Point | Exact Revision | Status |
| --- | --- | --- | --- | --- |
| Operational label provenance | Data and limitations | Labels are not clinical | Avoid diagnosis/screening language | resolved |
| Danish-only scope | Introduction and limitations | No cross-lingual evidence | Keep generalization cautious | resolved |
| Boundary opacity proxy | Related work and limitations | Not pronunciation-aware | Keep as secondary interpretation | resolved |
| Parser fallback | Data and limitations | No syntax claims | Explicitly defer parser-syntax claims | resolved |
| Gemma pending | Limitations | No model-family sensitivity | Defer Gemma to future work | unresolved |
