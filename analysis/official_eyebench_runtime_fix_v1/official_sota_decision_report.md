# Official EyeBench SOTA Decision Report

- Final claim category: `blocked_by_baseline_reproduction`
- Official EyeBench SOTA allowed: False
- Environment kind: `python312_minimal_runtime_compatible`
- Exact `environment.yml` used: False
- Recommended wording: benchmark-relative state of the art under internal EyeBench-style reader-aggregated evaluation.

| gate | passed |
| --- | --- |
| official environment or compatible runtime | True |
| official processed CopCo data present | True |
| official folds validated | True |
| official evaluator or exact result format | True |
| official baseline reproduced | False |
| local/manual baseline completed | True |
| baseline used official command source | False |
| baseline within tolerance | True |
| D3_EyeBench_Lite complete | True |
| D3 beats strongest official baseline | False |
| no leakage | True |
| no prohibited predictors | True |
| no full-data substitution | True |

## Result Separation
- Official EyeBench trial-level result: only valid if official data/folds/evaluator gates pass.
- Reader-aggregated metrics are secondary and cannot define the official claim.
- BenchmarkBridge full-data metrics remain benchmark-relative.
