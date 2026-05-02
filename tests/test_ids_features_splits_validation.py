from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from copco_eye_bench.cli import build_features_main
from copco_eye_bench.features import GAZE_METRIC_MAP, assert_unique_participant_word, build_feature_tables
from copco_eye_bench.ids import add_stable_ids, word_id_from_source
from copco_eye_bench.resources import syllable_count, vowel_cluster_syllable_count
from copco_eye_bench.slurm import launcher_command
from copco_eye_bench.splits import assert_no_group_leakage, participant_grouped_folds
from copco_eye_bench.validation import validate_metrics_schema, validate_run


def test_stable_word_id_construction() -> None:
    assert word_id_from_source(45.0, 3.0, 2.0, 11.0) == "45_p3_s2_w11"
    frame = pd.DataFrame(
        {
            "part": ["p1"],
            "speechId": [45],
            "paragraphId": [3],
            "sentenceId": [2],
            "wordId": [11],
        }
    )
    out = add_stable_ids(frame)
    assert out.loc[0, "participant_id"] == "P01"
    assert out.loc[0, "word_id"] == "45_p3_s2_w11"


def test_metric_mapping_and_join_cardinality_guard() -> None:
    assert GAZE_METRIC_MAP["FFD"] == "word_first_fix_dur"
    frame = pd.DataFrame({"participant_id": ["P01", "P01"], "word_id": ["a", "a"]})
    with pytest.raises(ValueError, match="one row per participant_id"):
        assert_unique_participant_word(frame)


def test_public_resource_syllable_fallback_contract() -> None:
    assert vowel_cluster_syllable_count("læser") >= 1
    count, source = syllable_count("læser")
    assert count >= 1
    assert source in {"pyphen_da_DK", "vowel_cluster_fallback"}


def test_participant_grouped_split_has_no_participant_leakage() -> None:
    participants = pd.DataFrame(
        {
            "participant_id": ["P01", "P02", "P03", "P04"],
            "dyslexia_labeled": [0, 1, 0, 1],
        }
    )
    splits = participant_grouped_folds(participants, n_splits=2, seed=17)
    for fold in sorted(splits["fold"].unique()):
        train = splits[(splits["fold"] == fold) & (splits["split"] == "train")]
        test = splits[(splits["fold"] == fold) & (splits["split"] == "test")]
        assert_no_group_leakage(train, test, ["participant_id"])


def test_report_metric_schema_validation() -> None:
    valid = pd.DataFrame(
        {
            "model": ["A_gaze"],
            "classifier": ["l1"],
            "cv_regime": ["participant_grouped_5fold"],
            "roc_auc": [0.5],
            "pr_auc": [0.5],
            "brier": [0.25],
        }
    )
    validate_metrics_schema(valid)
    with pytest.raises(ValueError, match="missing columns"):
        validate_metrics_schema(valid.drop(columns=["pr_auc"]))


def _write_minimal_legacy_source(root: Path) -> None:
    extracted = root / "ExtractedFeatures"
    extracted.mkdir(parents=True)
    rows = [
        ("P01", 1, 45, 1, 1, 1, "Hej", 120, 130, 150, 1),
        ("P01", 1, 45, 1, 1, 2, "verden", 121, 131, 151, 2),
        ("P02", 1, 45, 1, 1, 1, "Hej", 220, 230, 250, 1),
        ("P02", 1, 45, 1, 1, 2, "verden", 221, 231, 251, 2),
        ("P01", 2, 1327, -1, 1, 1, "Practice", 1, 1, 1, 1),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "part",
            "trialId",
            "speechId",
            "paragraphId",
            "sentenceId",
            "wordId",
            "word",
            "word_first_fix_dur",
            "word_first_pass_dur",
            "word_total_fix_dur",
            "number_of_fixations",
        ],
    )
    frame["char_IA_ids"] = ""
    frame["landing_position"] = 0
    frame["word_go_past_time"] = frame["word_total_fix_dur"]
    frame["word_mean_fix_dur"] = frame["word_total_fix_dur"]
    frame["word_mean_sacc_dur"] = 0
    frame["word_peak_sacc_velocity"] = 0
    frame[frame["part"] == "P01"].to_csv(extracted / "P01.csv", index=False)
    frame[frame["part"] == "P02"].to_csv(extracted / "P02.csv", index=False)
    pd.DataFrame({"subj": ["P01", "P02"], "dyslexia": [0, 1]}).to_csv(
        root / "participant_stats.csv", index=False
    )


def test_build_features_and_validate_minimal_run(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy"
    _write_minimal_legacy_source(legacy_root)
    config = {
        "dataset": {
            "legacy_root": str(legacy_root),
            "extracted_features_glob": "ExtractedFeatures/P*.csv",
            "participant_stats_path": "participant_stats.csv",
            "excluded_participants": ["P14"],
            "excluded_speech_ids": ["1327"],
        },
        "cv": {"participant_grouped_folds": 2, "random_seeds": [17]},
    }
    output_dir = tmp_path / "run"
    manifest = build_feature_tables(config, output_dir, repo_root=tmp_path)
    assert manifest["row_counts"]["word_observations"] == 4
    words = pd.read_parquet(output_dir / "tables" / "words.parquet")
    assert words["speech_id"].tolist() == ["45", "45"]
    report = validate_run(output_dir)
    assert report["status"] == "passed", json.dumps(report["errors"])


def test_build_features_cli_uses_sample_defaults(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy"
    _write_minimal_legacy_source(legacy_root)
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        f"""
run:
  name: test_smoke
  output_root: results
sample:
  participants: 1
  speeches: 1
dataset:
  legacy_root: {legacy_root}
  extracted_features_glob: ExtractedFeatures/P*.csv
  participant_stats_path: participant_stats.csv
  excluded_participants:
    - P14
  excluded_speech_ids:
    - "1327"
cv:
  participant_grouped_folds: 2
  random_seeds:
    - 17
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"
    exit_code = build_features_main(
        ["--config", str(config_path), "--repo-root", str(tmp_path), "--output-dir", str(output_dir)]
    )
    assert exit_code == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample"] == {"participants": 1, "speeches": 1}
    assert manifest["row_counts"]["participants"] == 1
    assert manifest["row_counts"]["word_observations"] == 2


def test_slurm_launcher_activates_copco_env_and_uses_cpu_mode(tmp_path: Path) -> None:
    command = launcher_command("copco-build-features --config config.yaml", repo_root=tmp_path, mode="cpu")
    assert "~/bin/claim_best_immediate_resource.sh --mode cpu" in command
    assert "conda activate copco" in command
    assert "SLURM_JOB_ID" in command
