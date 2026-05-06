# BenchmarkBridge Decision Report

- Decision category: `main_paper_comparison`
- Official EyeBench-compatible: False

## Questions
| question | answer | evidence |
| --- | --- | --- |
| Does D3 beat the strongest listed CopCo TYP Unseen Reader baseline? | True | D3 AUROC=0.8961, BA=0.8158 |
| Does D3 beat the strongest listed CopCo TYP Unseen Reader + Text baseline? | True | D3 AUROC=0.8542, BA=0.7458 |
| Is the result official EyeBench-compatible or internal EyeBench-style? | internal EyeBench-style | EyeBench package/repository/CLI not available in the local CopCo workspace. |
| Should the benchmark comparison enter the main manuscript? | True | main_paper_comparison |
| Should it remain appendix-only? | False | main_paper_comparison |
| Does RCS show useful signal? | False | best R2=0.0116 |
| Does RCS affect the main paper story? | False | RCS is auxiliary and not part of the frozen TYP main claim. |
| Does the bridge change the final title, abstract, or claims? | False | Exact official EyeBench mode was not run. |

## Exact Paper Text
In an internal EyeBench-style bridge, the frozen D3 DFM residual gaze-profile model was re-evaluated under unseen-reader, unseen-text, and strict unseen-reader-plus-text splits. Because exact official EyeBench folds were not available, these results are benchmark-relative and should be reported as supplementary validation rather than official leaderboard scores.
