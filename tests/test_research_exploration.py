from __future__ import annotations

from pathlib import Path

import pandas as pd

from copco_eye_bench.research_exploration import (
    PARTICIPANT_METRIC_COLUMNS,
    PRIMARY_EXPOSURE_COUNT_FEATURES,
    RESIDUALIZATION_PREDICTORS,
    assert_primary_feature_sets_safe,
    participant_feature_groups,
    run_research_exploration,
    validate_preflight,
    validate_research_exploration,
)


def test_residualization_predictors_do_not_use_reader_group() -> None:
    forbidden = {"reader_group", "reader_group_binary", "dyslexia_labeled", "participant_id"}
    assert not forbidden.intersection(RESIDUALIZATION_PREDICTORS)


def test_participant_feature_sets_exclude_exposure_counts_and_labels() -> None:
    feature_sets = participant_feature_groups()
    assert_primary_feature_sets_safe(feature_sets)
    for columns in feature_sets.values():
        assert not PRIMARY_EXPOSURE_COUNT_FEATURES.intersection(columns)
        assert "reader_group" not in columns
        assert "dyslexia_labeled" not in columns


def _write_mini_prepared(root: Path) -> dict[str, Path]:
    label_dir = root / "label_release"
    prepared = label_dir / "prepared_dataset"
    labels = label_dir / "labels"
    prepared.mkdir(parents=True)
    labels.mkdir(parents=True)
    participants = [f"P{idx:02d}" for idx in range(1, 7)]
    groups = {
        "P01": ("typical_control", 0),
        "P02": ("typical_control", 0),
        "P03": ("typical_control", 0),
        "P04": ("dyslexia_labeled", 1),
        "P05": ("dyslexia_labeled", 1),
        "P06": ("dyslexia_labeled", 1),
    }
    word_rows = []
    for participant_id in participants:
        group, binary = groups[participant_id]
        for idx in range(12):
            speech_id = f"s{idx % 2}"
            word_id = f"{speech_id}_sent_w{idx}"
            opacity = idx % 4
            skip = 1 if idx % 5 == 0 and binary == 0 else 0
            base = 160 + 20 * opacity + 10 * binary
            word_rows.append(
                {
                    "participant_id": participant_id,
                    "speech_id": speech_id,
                    "paragraph_id": f"{speech_id}_p0",
                    "sentence_id": f"{speech_id}_sent",
                    "word_id": word_id,
                    "participant_word_key": f"{participant_id}::{word_id}",
                    "stimulus_word_key": word_id,
                    "word": f"word{idx}",
                    "reader_group": group,
                    "reader_group_binary": binary,
                    "dyslexia_labeled": binary,
                    "group_label": "dyslexia_labeled" if binary else "typical",
                    "FFD": float(base),
                    "GD": float(base + 20),
                    "TRT": float(base + 80),
                    "go_past_time": float(base + 40),
                    "fixation_count": 1 + opacity,
                    "skip": skip,
                    "refixation_count": opacity // 2,
                    "word_length_chars": 3 + opacity,
                    "log_corpus_frequency": 2.0 - 0.1 * opacity,
                    "dfm_lm_word_surprisal": 1.0 + 0.2 * opacity,
                    "dfm_lm_word_entropy": 0.5 + 0.1 * opacity,
                    "sentence_length_words": 8 + idx % 3,
                    "word_position_in_sentence_norm": idx / 12,
                    "prev_boundary_opacity_score": float(opacity),
                    "prev_boundary_type_orth": ["C#C", "C#V", "V#C", "V#V"][opacity],
                    "vocoid_run_cross_boundary": opacity,
                    "paragraph_cohesion": 0.8,
                    "local_semantic_drift": 0.2,
                    "embedding_missing": False,
                    "lm_missing": False,
                    "parser_status": "surface_heuristic_fallback",
                    "lm_alignment_status": "warning",
                    "lm_alignment_warning": "non_special_token_unassigned",
                    "lm_alignment_error": None,
                    "dfm_lm_alignment_status": "warning",
                    "dfm_lm_alignment_warning": "non_special_token_unassigned",
                    "dfm_lm_alignment_error": None,
                    "participant_label_missing": False,
                    "segmentation_label_missing": False,
                    "segmentation_label_version": "segmentation_orthographic_v1",
                    "age": 30 + binary,
                    "sex": "F" if idx % 2 == 0 else "M",
                    "comprehension_score": 0.8,
                }
            )
    word = pd.DataFrame(word_rows)
    sentence = (
        word[["speech_id", "paragraph_id", "sentence_id", "sentence_length_words"]]
        .drop_duplicates("sentence_id")
        .assign(
            sentence_text="mini",
            sentence_mean_boundary_opacity=1.5,
            sentence_vv_boundary_rate=0.25,
            lm_missing_rate=0.0,
            embedding_missing_rate=0.0,
        )
    )
    participant_rows = []
    for participant_id in participants:
        group, binary = groups[participant_id]
        pword = word[word["participant_id"].eq(participant_id)]
        participant_rows.append(
            {
                "participant_id": participant_id,
                "reader_group": group,
                "reader_group_binary": binary,
                "dyslexia_labeled": binary,
                "group_label": "dyslexia_labeled" if binary else "typical",
                "mean_ffd": pword["FFD"].mean(),
                "median_ffd": pword["FFD"].median(),
                "mean_gd": pword["GD"].mean(),
                "median_gd": pword["GD"].median(),
                "mean_trt": pword["TRT"].mean(),
                "median_trt": pword["TRT"].median(),
                "skipping_rate": pword["skip"].mean(),
                "refixation_rate": pword["refixation_count"].mean(),
                "mean_go_past_time": pword["go_past_time"].mean(),
                "trt_sd": pword["TRT"].std(),
                "trt_q90": pword["TRT"].quantile(0.9),
                "length_sensitivity": 1.0,
                "frequency_sensitivity": -1.0,
                "surprisal_sensitivity": 1.0,
                "entropy_sensitivity": 1.0,
                "age": 30 + binary,
                "sex": "F",
                "comprehension_score": 0.8,
            }
        )
    participant = pd.DataFrame(participant_rows)
    participant_labels = participant[
        ["participant_id", "reader_group", "reader_group_binary", "age", "sex", "comprehension_score"]
    ].copy()
    participant_labels["label_source"] = "project_metadata"
    participant_labels["diagnostic_provenance"] = "project_metadata"
    participant_labels["label_confidence"] = "medium"
    participant_labels["include_primary_analysis"] = True
    participant_labels["include_sensitivity_analysis"] = True
    quality = word[
        [
            "participant_id",
            "speech_id",
            "paragraph_id",
            "sentence_id",
            "word_id",
            "stimulus_word_key",
            "participant_word_key",
            "lm_missing",
            "lm_alignment_status",
            "lm_alignment_warning",
            "lm_alignment_error",
            "parser_status",
            "embedding_missing",
            "participant_label_missing",
            "segmentation_label_missing",
        ]
    ].copy()
    quality["gaze_missing"] = False
    quality["parser_missing"] = False
    quality["parser_confidence"] = "usable_for_surface_not_syntax"
    quality["participant_metadata_missing"] = False
    quality["include_primary_analysis"] = True
    quality["include_sensitivity_analysis"] = True
    quality["segmentation_confidence"] = "orthographic_proxy"
    quality["text_assignment_balance_status"] = "documented_not_controlled"
    quality["exclusion_reason"] = ""
    segmentation_word = word[
        [
            "speech_id",
            "paragraph_id",
            "sentence_id",
            "word_id",
            "word",
            "prev_boundary_type_orth",
            "prev_boundary_opacity_score",
        ]
    ].copy()
    segmentation_word["segmentation_label_version"] = "segmentation_orthographic_v1"
    segmentation_boundary = segmentation_word.copy()
    segmentation_boundary["boundary_id"] = segmentation_boundary["word_id"] + "_b"
    segmentation_boundary["orth_boundary_type"] = segmentation_boundary["prev_boundary_type_orth"]
    split_rows = []
    for fold, test_id in enumerate(participants):
        for participant_id in participants:
            split_rows.append(
                {
                    "split_name": "leave_one_participant_out",
                    "fold_id": fold,
                    "participant_id": participant_id,
                    "reader_group": groups[participant_id][0],
                    "split_role": "test" if participant_id == test_id else "train",
                    "include_in_fold": True,
                    "split_valid": True,
                    "skip_reason": "",
                    "split_seed": 17,
                    "split_version": "split_policy_v1",
                }
            )
    fold_map = {"P01": 0, "P04": 0, "P02": 1, "P05": 1, "P03": 2, "P06": 2}
    for fold in range(3):
        for participant_id in participants:
            split_rows.append(
                {
                    "split_name": "participant_grouped_kfold",
                    "fold_id": fold,
                    "participant_id": participant_id,
                    "reader_group": groups[participant_id][0],
                    "split_role": "test" if fold_map[participant_id] == fold else "train",
                    "include_in_fold": True,
                    "split_valid": True,
                    "skip_reason": "",
                    "split_seed": 17,
                    "split_version": "split_policy_v1",
                }
            )
    splits = pd.DataFrame(split_rows)
    for frame, path in [
        (word, prepared / "analysis_ready_word_level_v1_1.parquet"),
        (sentence, prepared / "analysis_ready_sentence_level_v1_1.parquet"),
        (participant, prepared / "analysis_ready_participant_level_v1_1.parquet"),
        (participant_labels, labels / "participant_labels_v1.parquet"),
        (quality, labels / "quality_labels_v1.parquet"),
        (splits, labels / "split_labels_v1.parquet"),
        (segmentation_word, labels / "segmentation_word_labels_v1.parquet"),
        (segmentation_boundary, labels / "segmentation_boundary_labels_v1.parquet"),
    ]:
        frame.to_parquet(path, index=False)
    return {"label_dir": label_dir, "prepared": prepared}


def test_research_exploration_generates_reports_and_stable_metrics(tmp_path: Path) -> None:
    paths = _write_mini_prepared(tmp_path)
    config = {
        "run": {"name": "research_exploration_v1", "output_root": str(tmp_path / "results")},
        "research_exploration": {
            "label_release_dir": str(paths["label_dir"]),
            "prepared_dataset_dir": str(paths["prepared"]),
            "repo_analysis_dir": str(tmp_path / "analysis" / "research_exploration"),
            "deterministic_seed": 17,
            "no_new_core_labels": True,
            "no_llm_generated_labels": True,
            "parser_status_expected": "surface_heuristic_fallback",
            "expected_row_counts": {
                "word_level": 72,
                "sentence_level": 2,
                "participant_level": 6,
                "participant_labels": 6,
                "segmentation_words": 72,
                "segmentation_boundaries": 72,
                "quality_labels": 72,
            },
            "modeling": {
                "max_word_model_rows": 50,
                "max_word_ladder_rows": 50,
                "participant_prediction_models": ["logistic_regression", "linear_svm"],
                "permutation_count": 2,
                "bootstrap_count": 2,
            },
            "leakage_policy": {
                "legal_split_names": [
                    "leave_one_participant_out",
                    "participant_grouped_kfold",
                    "sensitivity_exclude_uncertain_labels",
                ]
            },
        },
    }
    out = tmp_path / "research"
    manifest = run_research_exploration(config, out, repo_root=tmp_path)
    assert manifest["status"] == "complete"
    validation = validate_research_exploration(config, out, repo_root=tmp_path)
    assert validation["status"] == "passed", validation["errors"]
    preflight = validate_preflight(config, repo_root=tmp_path)
    assert preflight["status"] == "passed", preflight["errors"]

    metrics = pd.read_csv(
        out
        / "analysis"
        / "research_exploration"
        / "participant_prediction_ablation_metrics.csv"
    )
    assert set(PARTICIPANT_METRIC_COLUMNS).issubset(metrics.columns)
    assert (
        out
        / "analysis"
        / "research_exploration"
        / "phase3_research_exploration_decision_report.md"
    ).exists()
    ladder = pd.read_csv(
        out
        / "analysis"
        / "research_exploration"
        / "word_level_secondary_ladder_metrics.csv"
    )
    assert "stage" in ladder.columns
    assert not pd.read_parquet(
        out
        / "analysis"
        / "research_exploration"
        / "participant_sensitivity_profiles.parquet"
    )["participant_id"].duplicated().any()
