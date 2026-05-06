from __future__ import annotations

from pathlib import Path

import pandas as pd

from copco_eye_bench.benchmark_bridge import PROHIBITED_FEATURES
from copco_eye_bench.official_eyebench_sota_check import (
    BASELINE_COLUMNS,
    COMPARISON_COLUMNS,
    SOTA_TYP_COLUMNS,
    run_official_eyebench_sota_check,
    validate_official_eyebench_sota_check,
    validate_official_eyebench_sota_check_config,
    validate_official_split_labels,
)
from tests.test_official_eyebench_alignment import _write_fake_eyebench


def _write_fake_processed_copco(eyebench: Path, participants: list[str], speeches: list[str]) -> None:
    processed = eyebench / "data" / "CopCo" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    ia_rows = []
    trial_rows = []
    for participant_index, participant_id in enumerate(participants):
        label = int(participant_index >= len(participants) // 2)
        for speech_index, speech_id in enumerate(speeches):
            unique_trial_id = f"{participant_id}_{speech_id}_0"
            unique_paragraph_id = f"{speech_id}_0"
            signal = 45 * label + 2 * speech_index
            trial_rows.append(
                {
                    "unique_trial_id": unique_trial_id,
                    "participant_id": participant_id,
                    "speech_id": speech_id,
                    "paragraph_id": 0,
                    "unique_paragraph_id": unique_paragraph_id,
                    "reading_speed": 0.2 + 0.03 * label + 0.001 * speech_index,
                    "first_pass_skip_rate": 0.1 + 0.02 * label,
                    "mean_FFD": 190 + signal,
                    "mean_GD": 210 + signal,
                    "mean_TFD": 260 + signal,
                    "mean_go_past_time": 280 + signal,
                    "CURRENT_FIX_DURATION_mean": 200 + signal,
                    "forward_saccade_length_mean": 3.0 - 0.2 * label,
                    "regression_rate": 0.1 + 0.04 * label,
                }
            )
            for word_id in range(12):
                ia_rows.append(
                    {
                        "unique_trial_id": unique_trial_id,
                        "participant_id": participant_id,
                        "speech_id": speech_id,
                        "paragraph_id": 0,
                        "unique_paragraph_id": unique_paragraph_id,
                        "dyslexia": label,
                        "RCS_score": float(participant_index + 1),
                        "word_id": word_id,
                        "word_length": 4 + (word_id % 5),
                        "wordfreq_frequency": 2.0 + 0.05 * word_id,
                        "gpt2_surprisal": 3.0 + 0.1 * word_id,
                        "TRIAL_IA_COUNT": 12,
                        "normalized_ID": word_id / 12,
                        "start_of_line": int(word_id == 0),
                        "end_of_line": int(word_id == 11),
                        "is_content_word": int(word_id % 2 == 0),
                        "IA_FIRST_FIXATION_DURATION": 180 + signal + word_id,
                        "IA_FIRST_RUN_DWELL_TIME": 200 + signal + word_id,
                        "IA_SELECTIVE_REGRESSION_PATH_DURATION": 230 + signal + word_id,
                        "IA_TOTAL_FIXATION_DURATION": 260 + signal + word_id,
                        "IA_DWELL_TIME": 260 + signal + word_id,
                        "IA_SKIP": int((word_id + label) % 5 == 0),
                        "IA_FIXATION_COUNT": 1 + label + (word_id % 2),
                    }
                )
    pd.DataFrame(ia_rows).to_feather(processed / "ia.feather")
    pd.DataFrame(trial_rows).to_feather(processed / "trial_level.feather")
    pd.DataFrame(
        {
            "feature_name": [
                "reading_speed",
                "first_pass_skip_rate",
                "mean_FFD",
                "mean_GD",
                "mean_TFD",
                "mean_go_past_time",
            ],
            "feature_type": ["LOGISTIC"] * 6,
        }
    ).to_csv(processed / "ia_trial_level_feature_keys.csv", index=False)
    pd.DataFrame(
        {
            "feature_name": ["CURRENT_FIX_DURATION_mean", "forward_saccade_length_mean", "regression_rate"],
            "feature_type": ["LOGISTIC", "LOGISTIC", "LOGISTIC"],
        }
    ).to_csv(processed / "fixation_trial_level_feature_keys.csv", index=False)


def _mini_sota_config(eyebench: Path, tmp_path: Path) -> dict:
    prohibited = sorted(
        PROHIBITED_FEATURES | {"unique_trial_id", "unique_paragraph_id", "dyslexia", "RCS_score"}
    )
    return {
        "run": {"name": "official_eyebench_sota_check_v1", "output_root": str(tmp_path / "results")},
        "official_eyebench_sota_check": {
            "version": "v1",
            "eyebench": {
                "path": str(eyebench),
                "global_processed_dir": str(eyebench / "data" / "processed"),
                "processed_copco_dir": str(eyebench / "data" / "CopCo" / "processed"),
                "allow_data_download": False,
                "run_preprocessing_if_environment_ok": False,
            },
            "environment": {
                "name": "missing_eyebench_official_test_env",
                "auto_create_in_cli": False,
            },
            "repo_analysis_dir": str(tmp_path / "analysis" / "official_eyebench_sota_check_v1"),
            "deterministic_seed": 173,
            "no_new_labels": True,
            "no_feature_engineering_search": True,
            "no_broad_model_search": True,
            "forbid_random_word_level_split": True,
            "tasks": ["CopCo_TYP"],
            "split_regimes": ["unseen_reader", "unseen_text", "unseen_reader_and_text"],
            "prohibited_features": prohibited,
            "residualization": {"reader_group_never_used": True},
            "decision_gates": {
                "CopCo_TYP": {
                    "formatted_table": str(
                        eyebench / "results" / "formatted_eyebench_benchmark_results" / "CopCo_TYP_test.csv"
                    )
                }
            },
        },
    }


def test_official_eyebench_sota_config_parses() -> None:
    import yaml

    config = yaml.safe_load(Path("configs/official_eyebench_sota_check_v1.yaml").read_text())
    assert validate_official_eyebench_sota_check_config(config)["status"] == "passed"


def test_official_eyebench_sota_end_to_end_with_fake_processed_data(tmp_path: Path) -> None:
    participants = [f"P{idx:02d}" for idx in range(1, 9)]
    speeches = ["S1", "S2", "S3", "S4"]
    eyebench = _write_fake_eyebench(tmp_path, participants, speeches)
    _write_fake_processed_copco(eyebench, participants, speeches)
    config = _mini_sota_config(eyebench, tmp_path)
    out = tmp_path / "results" / "official_eyebench_sota_check_v1_test"

    manifest = run_official_eyebench_sota_check(config, out, repo_root=tmp_path)
    report = validate_official_eyebench_sota_check(config, out, repo_root=tmp_path)

    assert manifest["status"] == "complete"
    assert report["status"] == "passed", report
    metrics = pd.read_csv(out / "typ" / "d3_eyebench_lite_metrics.csv")
    baseline = pd.read_csv(out / "baseline" / "official_baseline_reproduction_metrics.csv")
    comparison = pd.read_csv(
        tmp_path
        / "analysis"
        / "official_eyebench_sota_check_v1"
        / "tables"
        / "copco_typ_official_sota_comparison.csv"
    )
    assert set(SOTA_TYP_COLUMNS).issubset(metrics.columns)
    assert set(BASELINE_COLUMNS).issubset(baseline.columns)
    assert set(COMPARISON_COLUMNS).issubset(comparison.columns)
    assert metrics["status"].eq("complete").any()
    assert baseline["status"].eq("complete").any()
    assert not comparison["model"].isna().any()
    splits = pd.read_parquet(out / "splits" / "official_eyebench_sota_split_labels.parquet")
    split_errors, _ = validate_official_split_labels(splits)
    assert split_errors == []
    decision = (out / "official_eyebench_sota_decision_report.json").read_text(encoding="utf-8")
    assert "blocked_by_environment" in decision


def test_official_sota_prohibited_predictors_are_configured() -> None:
    import yaml

    config = yaml.safe_load(Path("configs/official_eyebench_sota_check_v1.yaml").read_text())
    prohibited = set(config["official_eyebench_sota_check"]["prohibited_features"])
    assert {"participant_id", "speech_id", "text_id", "dyslexia", "RCS_score"}.issubset(prohibited)
