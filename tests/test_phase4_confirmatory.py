from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from copco_eye_bench.phase4_confirmatory import (
    PHASE4_METRIC_COLUMNS,
    PHASE4_RESIDUALIZATION_PREDICTORS,
    PRIMARY_EXCLUDED_FEATURES,
    assert_phase4_primary_feature_sets_safe,
    phase4_feature_groups,
    run_phase4_confirmatory,
    validate_phase4_confirmatory,
)


def _write_mini_phase4_inputs(root: Path) -> dict[str, Path]:
    label_dir = root / "label_release"
    prepared = label_dir / "prepared_dataset"
    labels = label_dir / "labels"
    phase3_analysis = root / "phase3" / "analysis" / "research_exploration"
    prepared.mkdir(parents=True)
    labels.mkdir(parents=True)
    phase3_analysis.mkdir(parents=True)
    participants = [f"P{idx:02d}" for idx in range(1, 7)]
    groups = {
        "P01": ("typical_control", 0),
        "P02": ("typical_control", 0),
        "P03": ("typical_control", 0),
        "P04": ("dyslexia_labeled", 1),
        "P05": ("dyslexia_labeled", 1),
        "P06": ("dyslexia_labeled", 1),
    }
    rows = []
    for participant_id in participants:
        group, binary = groups[participant_id]
        for idx in range(14):
            speech_id = f"s{idx % 2}"
            word_id = f"{speech_id}_sent_w{idx}"
            opacity = idx % 4
            surprisal = 0.8 + 0.2 * opacity
            entropy = 0.4 + 0.1 * opacity
            base = 145 + 12 * opacity + 7 * binary + 10 * binary * surprisal
            rows.append(
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
                    "go_past_time": float(base + 35),
                    "fixation_count": 1 + opacity + binary,
                    "skip": int(opacity == 0 and binary == 0),
                    "refixation_count": opacity // 2,
                    "word_length_chars": 3 + opacity,
                    "log_corpus_frequency": 2.2 - 0.1 * opacity,
                    "dfm_lm_word_surprisal": surprisal,
                    "dfm_lm_word_entropy": entropy,
                    "sentence_length_words": 8 + idx % 3,
                    "word_position_in_sentence_norm": idx / 14,
                    "prev_boundary_opacity_score": float(opacity),
                    "prev_boundary_type_orth": ["C#C", "C#V", "V#C", "V#V"][opacity],
                    "vocoid_run_cross_boundary": opacity,
                    "paragraph_cohesion": 0.8,
                    "local_semantic_drift": 0.2,
                    "embedding_missing": False,
                    "lm_missing": False,
                    "parser_missing": False,
                    "parser_status": "surface_heuristic_fallback",
                    "lm_alignment_status": "ok",
                    "lm_alignment_warning": None,
                    "lm_alignment_error": None,
                    "dfm_lm_alignment_status": "ok",
                    "dfm_lm_alignment_warning": None,
                    "dfm_lm_alignment_error": None,
                    "participant_label_missing": False,
                    "segmentation_label_missing": False,
                    "segmentation_label_version": "segmentation_orthographic_v1",
                    "age": 30 + binary,
                    "sex": "F" if idx % 2 == 0 else "M",
                    "comprehension_score": 0.8,
                }
            )
    word = pd.DataFrame(rows)
    participant_rows = []
    phase3_rows = []
    for participant_id in participants:
        group, binary = groups[participant_id]
        pword = word[word["participant_id"].eq(participant_id)]
        common = {
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
            "length_sensitivity": 0.5 + binary,
            "frequency_sensitivity": -0.4 - binary,
            "surprisal_sensitivity": 0.8 + binary,
            "entropy_sensitivity": 0.7 + binary,
            "age": 30 + binary,
            "sex": "F",
            "comprehension_score": 0.8,
        }
        participant_rows.append(
            {
                **common,
                "word_observation_count": len(pword),
                "n_speeches": pword["speech_id"].nunique(),
                "n_word_rows": len(pword),
                "n_words_read": len(pword),
                "mean_segmentation_opacity": pword["prev_boundary_opacity_score"].mean(),
                "mean_dfm_surprisal": pword["dfm_lm_word_surprisal"].mean(),
                "mean_dfm_entropy": pword["dfm_lm_word_entropy"].mean(),
            }
        )
        phase3_rows.append(
            {
                **common,
                "ffd_residual_mean": 0.1 * binary,
                "first_pass_residual_mean": 0.2 * binary,
                "go_past_residual_mean": 0.3 * binary,
                "trt_residual_mean": 0.4 * binary,
                "skipping_residual_mean": -0.1 * binary,
                "fixation_count_residual_mean": 0.2 * binary,
                "mean_word_length_exposure": pword["word_length_chars"].mean(),
                "mean_log_frequency_exposure": pword["log_corpus_frequency"].mean(),
                "mean_sentence_length_exposure": pword["sentence_length_words"].mean(),
                "mean_dfm_surprisal_exposure": pword["dfm_lm_word_surprisal"].mean(),
                "mean_dfm_entropy_exposure": pword["dfm_lm_word_entropy"].mean(),
                "mean_boundary_opacity_exposure": pword["prev_boundary_opacity_score"].mean(),
                "vv_boundary_exposure_rate": pword["prev_boundary_type_orth"].eq("V#V").mean(),
                "lm_missing_rate_phase3": 0.0,
                "embedding_missing_rate_phase3": 0.0,
                "high_opacity_trt_residual_cost": 0.2 + binary,
                "vv_trt_residual_cost": 0.1 + binary,
                "length_sensitivity_phase3": 0.5 + binary,
                "frequency_sensitivity_phase3": -0.5 - binary,
                "surprisal_sensitivity_phase3": 0.9 + binary,
                "entropy_sensitivity_phase3": 0.8 + binary,
                "boundary_opacity_sensitivity_phase3": 0.2 + binary,
            }
        )
    participant = pd.DataFrame(participant_rows)
    participant_labels = participant[
        ["participant_id", "reader_group", "reader_group_binary", "age", "sex", "comprehension_score"]
    ].copy()
    participant_labels["include_primary_analysis"] = True
    participant_labels["include_sensitivity_analysis"] = True
    split_rows = []
    for fold_id, test_id in enumerate(participants):
        for participant_id in participants:
            split_rows.append(
                {
                    "split_name": "leave_one_participant_out",
                    "fold_id": fold_id,
                    "participant_id": participant_id,
                    "reader_group": groups[participant_id][0],
                    "split_role": "test" if participant_id == test_id else "train",
                    "include_in_fold": True,
                    "split_valid": True,
                    "skip_reason": "",
                    "split_seed": 41,
                    "split_version": "split_policy_v1",
                }
            )
    fold_map = {"P01": 0, "P04": 0, "P02": 1, "P05": 1, "P03": 2, "P06": 2}
    for fold_id in range(3):
        for participant_id in participants:
            split_rows.append(
                {
                    "split_name": "participant_grouped_kfold",
                    "fold_id": fold_id,
                    "participant_id": participant_id,
                    "reader_group": groups[participant_id][0],
                    "split_role": "test" if fold_map[participant_id] == fold_id else "train",
                    "include_in_fold": True,
                    "split_valid": True,
                    "skip_reason": "",
                    "split_seed": 41,
                    "split_version": "split_policy_v1",
                }
            )
    splits = pd.DataFrame(split_rows)
    segmentation_boundary = word[
        ["speech_id", "paragraph_id", "sentence_id", "word_id", "prev_boundary_type_orth"]
    ].copy()
    segmentation_boundary["boundary_id"] = segmentation_boundary["word_id"] + "_b"
    for frame, path in [
        (word, prepared / "analysis_ready_word_level_v1_1.parquet"),
        (participant, prepared / "analysis_ready_participant_level_v1_1.parquet"),
        (participant_labels, labels / "participant_labels_v1.parquet"),
        (splits, labels / "split_labels_v1.parquet"),
        (segmentation_boundary, labels / "segmentation_boundary_labels_v1.parquet"),
        (pd.DataFrame(phase3_rows), phase3_analysis / "participant_sensitivity_profiles.parquet"),
    ]:
        frame.to_parquet(path, index=False)
    return {"label_dir": label_dir, "prepared": prepared, "phase3_analysis": phase3_analysis}


def _mini_config(paths: dict[str, Path], tmp_path: Path) -> dict:
    return {
        "run": {"name": "phase4_confirmatory_sensitivity_v1", "output_root": str(tmp_path / "results")},
        "phase4_confirmatory": {
            "label_release_dir": str(paths["label_dir"]),
            "prepared_dataset_dir": str(paths["prepared"]),
            "phase3_analysis_dir": str(paths["phase3_analysis"]),
            "repo_analysis_dir": str(tmp_path / "analysis" / "phase4_confirmatory"),
            "deterministic_seed": 41,
            "no_new_core_labels": True,
            "no_broad_exploratory_feature_expansion": True,
            "parser_status_expected": "surface_heuristic_fallback",
            "expected_row_counts": {
                "word_level": 84,
                "participant_level": 6,
                "participant_labels": 6,
                "participant_sensitivity_profiles_phase3": 6,
            },
            "modeling": {
                "participant_prediction_models": ["logistic_regression", "linear_svm"],
                "permutation_count": 2,
                "bootstrap_count": 2,
                "mixed_effects_max_rows": 84,
            },
            "leakage_policy": {
                "legal_split_names": [
                    "leave_one_participant_out",
                    "participant_grouped_kfold",
                    "text_balanced_sensitivity_lopo",
                ]
            },
        },
    }


def test_crossfit_residualization_predictors_do_not_use_labels() -> None:
    forbidden = {"reader_group", "reader_group_binary", "dyslexia_labeled", "participant_id"}
    assert not forbidden.intersection(PHASE4_RESIDUALIZATION_PREDICTORS)


def test_phase4_feature_groups_separate_exposure_and_sensitivity() -> None:
    profiles = pd.DataFrame(
        columns=[
            "mean_dfm_surprisal_exposure",
            "mean_dfm_entropy_exposure",
            "surprisal_sensitivity_phase3",
            "entropy_sensitivity_phase3",
            "crossfit_total_fixation_residual_dfm_surprisal_slope",
            "crossfit_total_fixation_residual_dfm_entropy_slope",
        ]
    )
    groups = phase4_feature_groups(profiles)
    assert "mean_dfm_surprisal_exposure" in groups["D1_dfm_exposure_only"]
    assert "mean_dfm_surprisal_exposure" not in groups["D2_dfm_sensitivity_only"]
    assert "surprisal_sensitivity_phase3" in groups["D2_dfm_sensitivity_only"]
    assert "crossfit_total_fixation_residual_dfm_surprisal_slope" in groups[
        "D3_dfm_residual_gaze_only"
    ]


def test_phase4_primary_feature_sets_exclude_counts_and_labels() -> None:
    groups = phase4_feature_groups(pd.DataFrame(columns=["mean_ffd", "surprisal_sensitivity_phase3"]))
    assert_phase4_primary_feature_sets_safe(groups)
    for columns in groups.values():
        assert not PRIMARY_EXCLUDED_FEATURES.intersection(columns)


def test_phase4_confirmatory_generates_reports_and_stable_schema(tmp_path: Path) -> None:
    paths = _write_mini_phase4_inputs(tmp_path)
    config = _mini_config(paths, tmp_path)
    out = tmp_path / "phase4"
    manifest = run_phase4_confirmatory(config, out, repo_root=tmp_path)
    assert manifest["status"] == "complete"
    validation = validate_phase4_confirmatory(config, out, repo_root=tmp_path)
    assert validation["status"] == "passed", validation["errors"]
    assert not any(
        diagnostic["train_contains_heldout"]
        for diagnostic in manifest["cross_fitted_residualization"]["diagnostics"]
    )
    assert not manifest["cross_fitted_residualization"]["reader_group_used"]
    analysis = out / "analysis" / "phase4_confirmatory"
    metrics = pd.read_csv(analysis / "confirmatory_prediction_metrics.csv")
    assert set(PHASE4_METRIC_COLUMNS).issubset(metrics.columns)
    predictions = pd.read_csv(analysis / "confirmatory_predictions.csv")
    lopo = predictions[
        predictions["split_name"].eq("leave_one_participant_out")
        & predictions["feature_group"].eq("D2_dfm_sensitivity_only")
        & predictions["model"].eq("logistic_regression")
    ]
    assert len(lopo) == 6
    assert not lopo["participant_id"].duplicated().any()
    assert not metrics["split_name"].astype(str).str.contains("random", case=False).any()
    for report_name in [
        "dfm_exposure_vs_sensitivity_report.md",
        "cross_fitted_residualization_report.md",
        "confirmatory_prediction_report.md",
        "robustness_report.md",
        "feature_stability_report.md",
        "mixed_effects_interaction_report.md",
        "segmentation_decision_report.md",
        "phase4_publication_decision_report.md",
    ]:
        assert (analysis / report_name).exists()
    manifest_payload = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["preflight"]["status"] == "passed"
