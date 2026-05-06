# Official EyeBench Decision Report

- Decision category: `benchmark_relative_sota_only`
- Recommended wording: benchmark-relative state of the art under internal EyeBench-style evaluation

## Decisions
| question | answer |
| --- | --- |
| Did official EyeBench mode run? | False |
| Did exact EyeBench folds run? | True |
| Did exact EyeBench processed data run? | False |
| Did D3_EyeBench_Lite beat strongest official CopCo_TYP baselines? | False |
| Did D3_FullFeature_EyeBenchFolds beat baselines? | False |
| Did full-data BenchmarkBridge remain consistent? | True |
| Does this change the manuscript main claim? | False |
| Does this permit an official SOTA claim? | False |

## Claim Labels
- official EyeBench result: False
- EyeBench-compatible result: False
- EyeBench-fold-aligned result: True
- benchmark-relative result: True
- internal-only result: True

## Exact Wording
Use: "benchmark-relative state of the art under internal EyeBench-style evaluation." Do not call the result official unless exact processed EyeBench data, folds, and evaluator are used.
