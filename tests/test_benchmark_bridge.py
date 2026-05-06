from __future__ import annotations

from pathlib import Path

import pandas as pd

from copco_eye_bench.benchmark_bridge import (
    PROHIBITED_FEATURES,
    RCS_METRIC_COLUMNS,
    TYP_METRIC_COLUMNS,
    _feature_columns,
    run_benchmark_bridge,
    validate_benchmark_bridge,
    validate_benchmark_config,
    validate_split_labels,
)
from copco_eye_bench.config import load_config


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_bridge_inputs(root: Path) -> dict[str, Path]:
    label_dir = root / "label_release"
    prepared = label_dir / "prepared_dataset"
    labels = label_dir / "labels"
    participants = [f"P{idx:02d}" for idx in range(1, 9)]
    groups = {
        participant: ("dyslexia_labeled", 1) if idx >= 4 else ("typical_control", 0)
        for idx, participant in enumerate(participants)
    }
    texts = ["S1", "S2", "S3", "S4"]
    word_rows = []
    for participant_id in participants:
        reader_group, binary = groups[participant_id]
        for text_index, speech_id in enumerate(texts):
            for word_index in range(12):
                opacity = word_index % 4
                surprisal = 0.8 + 0.2 * opacity + 0.05 * text_index
                entropy = 0.4 + 0.1 * opacity
                base = 150 + 10 * opacity + 20 * binary + 8 * binary * surprisal
                word_id = f"{speech_id}_w{word_index}"
                word_rows.append(
                    {
                        "participant_id": participant_id,
                        "speech_id": speech_id,
                        "paragraph_id": f"{speech_id}_p0",
                        "sentence_id": f"{speech_id}_sent0",
                        "word_id": word_id,
                        "participant_word_key": f"{participant_id}::{word_id}",
                        "stimulus_word_key": word_id,
                        "word": f"word{word_index}",
                        "reader_group": reader_group,
                        "reader_group_binary": binary,
                        "dyslexia_labeled": binary,
                        "group_label": reader_group,
                        "FFD": float(base),
                        "GD": float(base + 20),
                        "TRT": float(base + 90),
                        "go_past_time": float(base + 40),
                        "fixation_count": 1 + opacity + binary,
                        "skip": int(opacity == 0 and binary == 0),
                        "refixation_count": opacity // 2,
                        "landing_position": 2.0,
                        "word_length_chars": 3 + opacity,
                        "log_corpus_frequency": 2.2 - 0.1 * opacity,
                        "dfm_lm_word_surprisal": surprisal,
                        "dfm_lm_word_entropy": entropy,
                        "sentence_length_words": 12,
                        "word_position_in_sentence_norm": word_index / 12,
                        "prev_boundary_opacity_score": float(opacity),
                        "prev_boundary_type_orth": ["C#C", "C#V", "V#C", "V#V"][opacity],
                        "gaze_missing": False,
                        "participant_label_missing": False,
                        "segmentation_label_missing": False,
                        "lm_missing": False,
                        "embedding_missing": False,
                        "parser_missing": False,
                        "include_primary_analysis": True,
                        "include_sensitivity_analysis": True,
                        "comprehension_score": 0.35 + 0.05 * int(participant_id[-2:]) + 0.1 * binary,
                    }
                )
    word = pd.DataFrame(word_rows)
    participant_rows = []
    label_rows = []
    for participant_id in participants:
        reader_group, binary = groups[participant_id]
        pword = word[word["participant_id"].eq(participant_id)]
        participant_rows.append(
            {
                "participant_id": participant_id,
                "reader_group": reader_group,
                "reader_group_binary": binary,
                "dyslexia_labeled": binary,
                "group_label": reader_group,
                "mean_ffd": pword["FFD"].mean(),
                "mean_trt": pword["TRT"].mean(),
                "comprehension_score": pword["comprehension_score"].iloc[0],
            }
        )
        label_rows.append(
            {
                "participant_id": participant_id,
                "reader_group": reader_group,
                "reader_group_binary": binary,
                "include_primary_analysis": True,
                "include_sensitivity_analysis": True,
                "comprehension_score": pword["comprehension_score"].iloc[0],
            }
        )
    participant = pd.DataFrame(participant_rows)
    labels_frame = pd.DataFrame(label_rows)
    quality = word[
        [
            "participant_id",
            "speech_id",
            "paragraph_id",
            "sentence_id",
            "word_id",
            "participant_word_key",
            "stimulus_word_key",
            "gaze_missing",
            "participant_label_missing",
            "segmentation_label_missing",
            "lm_missing",
            "parser_missing",
            "embedding_missing",
            "include_primary_analysis",
            "include_sensitivity_analysis",
        ]
    ].copy()
    segmentation_boundary = word[
        ["speech_id", "paragraph_id", "sentence_id", "word_id", "prev_boundary_type_orth"]
    ].drop_duplicates("word_id")
    split_labels = pd.DataFrame(
        [
            {
                "split_name": "leave_one_participant_out",
                "fold_id": 0,
                "participant_id": participants[0],
                "reader_group": "typical_control",
                "split_role": "test",
                "include_in_fold": True,
                "split_valid": True,
            }
        ]
    )
    _write_frame(word, prepared / "analysis_ready_word_level_v1_1.parquet")
    _write_frame(participant, prepared / "analysis_ready_participant_level_v1_1.parquet")
    _write_frame(labels_frame, labels / "participant_labels_v1.parquet")
    _write_frame(quality, labels / "quality_labels_v1.parquet")
    _write_frame(segmentation_boundary, labels / "segmentation_boundary_labels_v1.parquet")
    _write_frame(split_labels, labels / "split_labels_v1.parquet")
    return {"label_dir": label_dir, "prepared": prepared}


def _mini_config(paths: dict[str, Path], tmp_path: Path) -> dict:
    return {
        "run": {"name": "benchmark_bridge_v1", "output_root": str(tmp_path / "results")},
        "benchmark_bridge": {
            "version": "v1",
            "frozen_inputs": {
                "feature_release_dir": str(tmp_path / "feature_release"),
                "label_release_dir": str(paths["label_dir"]),
                "prepared_dataset_dir": str(paths["prepared"]),
                "phase3_dir": str(tmp_path / "phase3"),
                "phase3_analysis_dir": str(tmp_path / "phase3" / "analysis"),
                "phase4_dir": str(tmp_path / "phase4"),
                "phase4_analysis_dir": str(tmp_path / "phase4" / "analysis"),
            },
            "prepared_dataset_path": str(paths["prepared"]),
            "output_dir_pattern": str(tmp_path / "results" / "benchmark_bridge_v1_<timestamp>"),
            "repo_analysis_dir": str(tmp_path / "analysis" / "benchmark_bridge_v1"),
            "deterministic_seed": 97,
            "no_new_labels": True,
            "no_feature_engineering_search": True,
            "forbid_random_word_level_split": True,
            "tasks": [
                {"name": "CopCo_TYP", "enabled": True, "role": "primary"},
                {"name": "CopCo_RCS", "enabled": True, "role": "optional_auxiliary"},
            ],
            "primary_model": {
                "feature_group": "D3_dfm_residual_gaze_only",
                "model": "logistic_regression",
            },
            "comparison_feature_groups": [
                "D1_dfm_exposure_only",
                "D2_dfm_sensitivity_only",
                "D3_dfm_residual_gaze_only",
                "D4_dfm_exposure_plus_sensitivity",
                "raw_gaze_baseline",
                "reading_speed_baseline",
                "no_dfm_baseline",
                "no_segmentation_baseline",
            ],
            "typ_feature_groups": ["chance_majority", "D3_dfm_residual_gaze_only"],
            "rcs_feature_groups": ["D3_dfm_residual_gaze_only"],
            "split_regimes": [
                "unseen_reader",
                "unseen_text",
                "unseen_reader_and_text",
                "text_balanced_unseen_reader",
                "leave_one_speech_out",
                "participant_grouped_kfold",
            ],
            "split_policy": {
                "participant_grouped_kfold_folds": 2,
                "text_balanced_unseen_reader_folds": 2,
                "unseen_reader_and_text_folds": 2,
            },
            "evaluation_units": ["participant_text_trial", "reader_aggregated"],
            "prohibited_features": sorted(PROHIBITED_FEATURES),
            "residualization": {
                "mode": "cross_fitted_within_each_split",
                "min_words_for_slope": 8,
                "reader_group_never_used": True,
            },
            "decision_gates": {
                "CopCo_TYP": {
                    "unseen_reader": {"test_AUROC": 0.853, "test_balanced_accuracy": 0.777},
                    "unseen_reader_and_text": {"test_AUROC": 0.749, "test_balanced_accuracy": 0.656},
                },
                "CopCo_RCS": {"average_test_R2_approx": 0.08, "validation_R2_approx": 0.07},
            },
            "official_eyebench_mode": "enabled_if_available",
            "internal_matched_mode": "always_enabled",
        },
    }


def test_benchmark_config_parses() -> None:
    config = load_config("configs/benchmark_bridge_v1.yaml")
    report = validate_benchmark_config(config)
    assert report["status"] == "passed"


def test_benchmark_bridge_end_to_end_and_validation(tmp_path: Path) -> None:
    paths = _write_bridge_inputs(tmp_path)
    config = _mini_config(paths, tmp_path)
    out = tmp_path / "results" / "benchmark_bridge_v1_test"

    manifest = run_benchmark_bridge(config, out, repo_root=tmp_path)
    report = validate_benchmark_bridge(config, out, repo_root=tmp_path)

    assert manifest["status"] == "complete"
    assert report["status"] == "passed", report
    samples = pd.read_parquet(out / "data" / "participant_text_trial_features.parquet")
    assert {"participant_id", "reader_group_binary", "speech_id", "text_id", "n_words_in_sample"}.issubset(
        samples.columns
    )
    assert any(column.startswith("global_") for column in samples.columns)

    splits = pd.read_parquet(out / "splits" / "benchmark_split_labels.parquet")
    split_errors, _ = validate_split_labels(splits)
    assert split_errors == []
    assert not splits["split_name"].str.contains("random", case=False, na=False).any()
    for (_, _), fold in splits[splits["split_name"].eq("unseen_reader")].groupby(
        ["split_name", "fold_id"]
    ):
        train = set(fold[fold["split_role"].eq("train")]["participant_id"])
        test = set(fold[fold["split_role"].eq("test")]["participant_id"])
        assert train.isdisjoint(test)
    for split_name in ["unseen_text", "unseen_reader_and_text"]:
        for (_, _), fold in splits[splits["split_name"].eq(split_name)].groupby(
            ["split_name", "fold_id"]
        ):
            train_text = set(fold[fold["split_role"].eq("train")]["text_id"])
            test_text = set(fold[fold["split_role"].eq("test")]["text_id"])
            assert train_text.isdisjoint(test_text)

    groups = _feature_columns(samples.assign(crossfit_ffd_residual_dfm_surprisal_slope=0.0))
    for columns in groups.values():
        assert PROHIBITED_FEATURES.isdisjoint(columns)

    residual_report = (out / "residualization" / "crossfit_residualization_report.md").read_text(
        encoding="utf-8"
    )
    assert "Held-out reader rows used for residual fitting: False" in residual_report
    assert "Reader group used in residualization: False" in residual_report

    typ_metrics = pd.read_csv(out / "typ" / "typ_benchmark_metrics.csv")
    assert set(TYP_METRIC_COLUMNS).issubset(typ_metrics.columns)
    assert {
        "unseen_reader",
        "unseen_text",
        "unseen_reader_and_text",
    }.issubset(set(typ_metrics["split_name"]))

    rcs_metrics = pd.read_csv(out / "rcs" / "rcs_benchmark_metrics.csv")
    assert set(RCS_METRIC_COLUMNS).issubset(rcs_metrics.columns)

    typ_comparison = pd.read_csv(
        tmp_path / "analysis" / "benchmark_bridge_v1" / "tables" / "copco_typ_benchmark_comparison.csv"
    )
    assert {
        "model",
        "unseen_reader_balanced_accuracy",
        "unseen_reader_text_AUROC",
        "official_mode",
    }.issubset(typ_comparison.columns)
    assert (tmp_path / "analysis" / "benchmark_bridge_v1" / "benchmark_bridge_decision_report.md").exists()
