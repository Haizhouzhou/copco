# OfficialEyeBenchBaselineEvaluatorClosure v1

This closure phase checks the remaining official EyeBench gates without changing
the frozen CopCo D3 result.

Scope:

- reuse official CopCo processed data and folds already produced under `eyebench/`
- attempt the official CopCo_TYP ML command source from the vendored EyeBench repo
- repair only missing runtime imports in the minimal Python 3.12 prefix
- validate the official result-format evidence for D3_EyeBench_Lite
- keep local diagnostic baselines separate from official command-source evidence

The documented compatible runtime is:

`eyebench/.envs/eyebench_official_py312_minimal`

This is not an exact `environment.yml` reproduction.

The official SOTA claim remains disallowed unless the official command-source
baseline/evaluator gates pass and D3_EyeBench_Lite beats the strongest published
CopCo_TYP baseline under the official trial-level comparison.
