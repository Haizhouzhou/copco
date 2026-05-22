"""Resume D3 online targeted optimization from staged prediction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from copco_eye_bench.config import load_config
from copco_eye_bench.d3_online_targeted_optimization import (
    _analysis_dir,
    _update_subgoal,
    _write_json,
    error_trajectory_analysis,
    evaluate_stopping_policies,
    online_offline_comparison_and_decision,
    oracle_diagnostics,
    targeted_optimization,
    update_manuscript_if_valid,
)


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out
    config = load_config(args.config, repo_root=root)
    analysis = _analysis_dir(config, root)

    online_path = out / "online_probabilities" / "online_probabilities.csv"
    prefix_metrics_path = analysis / "online_prefix_model_metrics.csv"
    accumulation_metrics_path = analysis / "online_evidence_accumulation_metrics.csv"
    for path in [online_path, prefix_metrics_path, accumulation_metrics_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    _update_subgoal(
        root,
        "GOAL_0",
        "completed",
        [
            "docs/d3_online_targeted_optimization_v1.md",
            "docs/d3_online_detection_goal_contract_v1.md",
            "docs/d3_online_testing_standard_v1.md",
            "analysis/d3_online_targeted_optimization_v1/subgoal_status.md",
            "analysis/d3_online_targeted_optimization_v1/subgoal_status.json",
        ],
    )
    _update_subgoal(
        root,
        "GOAL_1",
        "completed",
        [
            _relative(out / "prefix_data" / "prefix_features.parquet", root),
            _relative(analysis / "prefix_feature_dictionary.md", root),
            _relative(analysis / "prefix_dataset_report.md", root),
        ],
    )
    _update_subgoal(
        root,
        "GOAL_2",
        "completed",
        [
            _relative(out / "nested_predictions", root),
            _relative(analysis / "nested_prediction_artifact_report.md", root),
        ],
    )
    _update_subgoal(
        root,
        "GOAL_3",
        "completed",
        [
            _relative(analysis / "online_prefix_model_metrics.csv", root),
            _relative(analysis / "online_prefix_model_report.md", root),
        ],
    )
    _update_subgoal(
        root,
        "GOAL_4",
        "completed",
        [
            _relative(analysis / "legal_calibration_metrics.csv", root),
            _relative(analysis / "legal_threshold_metrics.csv", root),
            _relative(analysis / "legal_thresholds_learned.csv", root),
            _relative(analysis / "calibration_threshold_report.md", root),
        ],
    )
    _update_subgoal(
        root,
        "GOAL_5",
        "completed",
        [
            _relative(online_path, root),
            _relative(accumulation_metrics_path, root),
            _relative(analysis / "online_evidence_accumulation_report.md", root),
        ],
    )

    online = pd.read_csv(online_path, low_memory=False)
    prefix_metrics = pd.read_csv(prefix_metrics_path)
    accumulation_metrics = pd.read_csv(accumulation_metrics_path)

    stopping_path = analysis / "online_stopping_policy_metrics.csv"
    curve_path = analysis / "online_earliness_performance_curve.csv"
    if stopping_path.exists() and curve_path.exists():
        stopping_metrics = pd.read_csv(stopping_path)
        print(f"reused stopping metrics from {stopping_path}")
    else:
        print("evaluating stopping policies")
        stopping_metrics, _curve = evaluate_stopping_policies(config, online, root)
    stopping_ok = not stopping_metrics.empty and stopping_metrics["stopping_policy"].nunique() >= 4
    _update_subgoal(
        root,
        "GOAL_6",
        "completed" if stopping_ok else "blocked",
        [
            _relative(analysis / "online_stopping_policy_metrics.csv", root),
            _relative(analysis / "online_stopping_policy_report.md", root),
            _relative(analysis / "online_earliness_performance_curve.csv", root),
        ],
        "" if stopping_ok else "fewer than four stopping policies evaluated",
    )

    print("running targeted optimization")
    search_space, _ranking, locked = targeted_optimization(config, online, root)
    optimization_ok = len(search_space) >= 12 and not locked.empty
    _update_subgoal(
        root,
        "GOAL_7",
        "completed" if optimization_ok else "blocked",
        [
            _relative(analysis / "online_candidate_search_space.csv", root),
            _relative(analysis / "online_candidate_validation_ranking.csv", root),
            _relative(analysis / "online_locked_test_results.csv", root),
            _relative(analysis / "online_targeted_optimization_report.md", root),
        ],
        "" if optimization_ok else "candidate search or locked test result missing",
    )

    print("running oracle diagnostics")
    oracle, _oracle_budget = oracle_diagnostics(config, online, root)
    oracle_ok = not oracle.empty and (oracle["official_claim_allowed"] == False).all()  # noqa: E712
    _update_subgoal(
        root,
        "GOAL_8",
        "completed" if oracle_ok else "blocked",
        [
            _relative(analysis / "oracle_upper_bound_metrics.csv", root),
            _relative(analysis / "oracle_information_budget.csv", root),
            _relative(analysis / "oracle_upper_bound_report.md", root),
        ],
        "" if oracle_ok else "oracle diagnostics missing or not diagnostic",
    )

    print("running error trajectory analysis")
    trajectories, _persistent = error_trajectory_analysis(config, online, locked, root)
    _update_subgoal(
        root,
        "GOAL_9",
        "completed" if not trajectories.empty else "blocked",
        [
            _relative(analysis / "reader_probability_trajectories.csv", root),
            _relative(analysis / "persistent_error_readers.csv", root),
            _relative(analysis / "error_trajectory_report.md", root),
        ],
        "" if not trajectories.empty else "no locked trajectory rows",
    )

    print("writing online/offline comparison")
    comparison, decision = online_offline_comparison_and_decision(
        config, prefix_metrics, accumulation_metrics, stopping_metrics, locked, oracle, root
    )
    _update_subgoal(
        root,
        "GOAL_10",
        "completed" if not comparison.empty else "blocked",
        [
            _relative(analysis / "online_offline_comparison_table.csv", root),
            _relative(analysis / "online_offline_comparison_table.md", root),
            _relative(analysis / "final_online_targeted_decision_report.md", root),
        ],
        "" if not comparison.empty else "comparison table missing",
    )

    changed = update_manuscript_if_valid(decision, root)
    _update_subgoal(
        root,
        "GOAL_11",
        "completed" if changed else "blocked",
        changed,
        "" if changed else "manuscript files unavailable or already contained the online update",
    )

    manifest = {
        "status": "complete",
        "output_dir": str(out),
        "analysis_dir": str(analysis),
        "online_probability_rows": int(len(online)),
        "accumulation_metric_rows": int(len(accumulation_metrics)),
        "stopping_metric_rows": int(len(stopping_metrics)),
        "candidate_rows": int(len(search_space)),
        "locked_test_rows": int(len(locked)),
        "oracle_rows": int(len(oracle)),
        "trajectory_rows": int(len(trajectories)),
        "official_sota_claim_changed": False,
        "manuscript_changed": changed,
    }
    _write_json(out / "run_manifest.json", manifest)
    _write_json(analysis / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
