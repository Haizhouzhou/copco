from __future__ import annotations

from pathlib import Path

import pandas as pd

from copco_eye_bench.autoresearch import (
    FINAL_MODEL_GROUP,
    evaluate_decision_gates,
    run_autoresearch,
    run_refinement_loop,
    validate_autoresearch,
)


PARTICIPANTS = ["P01", "P02", "P03", "P04", "P05", "P06"]
GROUPS = {
    "P01": ("typical_control", 0),
    "P02": ("typical_control", 0),
    "P03": ("typical_control", 0),
    "P04": ("dyslexia_labeled", 1),
    "P05": ("dyslexia_labeled", 1),
    "P06": ("dyslexia_labeled", 1),
}
FINAL_FEATURES = [
    "crossfit_ffd_residual_dfm_surprisal_slope",
    "crossfit_ffd_residual_dfm_entropy_slope",
    "crossfit_total_fixation_residual_dfm_surprisal_slope",
]


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def _write_json(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.strip() + "\n", encoding="utf-8")


def _write_autoresearch_inputs(root: Path) -> dict[str, Path]:
    feature = root / "feature_release"
    label = root / "label_release"
    phase3 = root / "phase3"
    phase4 = root / "phase4"
    for directory in [feature, label, phase3, phase4]:
        directory.mkdir(parents=True, exist_ok=True)

    word_rows = []
    for participant_id in PARTICIPANTS:
        reader_group, binary = GROUPS[participant_id]
        for idx in range(10):
            speech_id = f"s{idx % 2}"
            word_id = f"{speech_id}_w{idx}"
            opacity = idx % 4
            word_rows.append(
                {
                    "participant_id": participant_id,
                    "speech_id": speech_id,
                    "paragraph_id": f"{speech_id}_p0",
                    "sentence_id": f"{speech_id}_sent{idx % 3}",
                    "word_id": word_id,
                    "participant_word_key": f"{participant_id}::{word_id}",
                    "stimulus_word_key": word_id,
                    "reader_group": reader_group,
                    "reader_group_binary": binary,
                    "dyslexia_labeled": binary,
                    "group_label": reader_group,
                    "FFD": 150.0 + opacity + binary,
                    "GD": 180.0 + opacity + binary,
                    "TRT": 250.0 + opacity + binary,
                    "go_past_time": 210.0 + opacity + binary,
                    "fixation_count": 1 + opacity,
                    "skip": int(opacity == 0),
                    "word_length_chars": 3 + opacity,
                    "log_corpus_frequency": 2.0 - 0.1 * opacity,
                    "dfm_lm_word_surprisal": 0.8 + 0.2 * opacity,
                    "dfm_lm_word_entropy": 0.4 + 0.1 * opacity,
                    "sentence_length_words": 8 + idx % 3,
                    "word_position_in_sentence_norm": idx / 10,
                    "prev_boundary_opacity_score": float(opacity),
                    "prev_boundary_type_orth": ["C#C", "C#V", "V#C", "V#V"][opacity],
                    "vocoid_run_cross_boundary": int(opacity == 3),
                    "lm_missing": False,
                    "lm_alignment_status": "warning",
                    "lm_alignment_warning": "non_special_token_unassigned",
                    "dfm_lm_alignment_status": "warning",
                    "dfm_lm_alignment_warning": "non_special_token_unassigned",
                    "parser_status": "surface_heuristic_fallback",
                    "segmentation_label_version": "segmentation_orthographic_v1",
                }
            )
    word = pd.DataFrame(word_rows)
    sentence = (
        word[["speech_id", "paragraph_id", "sentence_id", "sentence_length_words"]]
        .drop_duplicates("sentence_id")
        .assign(sentence_text="mini", sentence_mean_boundary_opacity=1.5)
    )
    participant_rows = []
    for participant_id in PARTICIPANTS:
        reader_group, binary = GROUPS[participant_id]
        pword = word[word["participant_id"].eq(participant_id)]
        participant_rows.append(
            {
                "participant_id": participant_id,
                "reader_group": reader_group,
                "reader_group_binary": binary,
                "dyslexia_labeled": binary,
                "group_label": reader_group,
                "n_speeches": pword["speech_id"].nunique(),
                "n_word_rows": len(pword),
                "n_words_read": len(pword),
                "word_observation_count": len(pword),
                "mean_dfm_surprisal": pword["dfm_lm_word_surprisal"].mean(),
                "mean_dfm_entropy": pword["dfm_lm_word_entropy"].mean(),
                "mean_segmentation_opacity": pword["prev_boundary_opacity_score"].mean(),
                "comprehension_score": 0.8,
            }
        )
    participant = pd.DataFrame(participant_rows)
    participant_labels = participant[
        ["participant_id", "reader_group", "reader_group_binary", "comprehension_score"]
    ].copy()
    participant_labels["include_primary_analysis"] = True
    quality = word[
        [
            "participant_id",
            "speech_id",
            "paragraph_id",
            "sentence_id",
            "word_id",
            "participant_word_key",
            "lm_missing",
            "lm_alignment_status",
            "lm_alignment_warning",
            "dfm_lm_alignment_status",
            "dfm_lm_alignment_warning",
            "parser_status",
        ]
    ].copy()
    split_rows = []
    for fold_id, heldout in enumerate(PARTICIPANTS):
        for participant_id in PARTICIPANTS:
            split_rows.append(
                {
                    "split_name": "leave_one_participant_out",
                    "fold_id": fold_id,
                    "participant_id": participant_id,
                    "split_role": "test" if participant_id == heldout else "train",
                    "include_in_fold": True,
                    "split_valid": True,
                }
            )
    for fold_id in range(3):
        for idx, participant_id in enumerate(PARTICIPANTS):
            split_rows.append(
                {
                    "split_name": "participant_grouped_kfold",
                    "fold_id": fold_id,
                    "participant_id": participant_id,
                    "split_role": "test" if idx % 3 == fold_id else "train",
                    "include_in_fold": True,
                    "split_valid": True,
                }
            )
    splits = pd.DataFrame(split_rows)
    segmentation_word = word[
        [
            "speech_id",
            "paragraph_id",
            "sentence_id",
            "word_id",
            "prev_boundary_type_orth",
            "prev_boundary_opacity_score",
        ]
    ].copy()

    _write_frame(word, label / "prepared_dataset" / "analysis_ready_word_level_v1_1.parquet")
    _write_frame(
        sentence,
        label / "prepared_dataset" / "analysis_ready_sentence_level_v1_1.parquet",
    )
    _write_frame(
        participant,
        label / "prepared_dataset" / "analysis_ready_participant_level_v1_1.parquet",
    )
    _write_frame(participant_labels, label / "labels" / "participant_labels_v1.parquet")
    _write_frame(quality, label / "labels" / "quality_labels_v1.parquet")
    _write_frame(splits, label / "labels" / "split_labels_v1.parquet")
    _write_frame(segmentation_word, label / "labels" / "segmentation_word_labels_v1.parquet")

    _write_json(feature / "manifest.json", '{"row_counts": {"word_level": 60}}')
    _write_json(
        label / "manifest.json",
        '{"row_counts": {"word_level": 60, "participant_level": 6}, '
        '"participant_counts": {"dyslexia_labeled": 3, "typical_control": 3}, '
        '"segmentation_label_version": "segmentation_orthographic_v1"}',
    )
    phase3_analysis = phase3 / "analysis" / "research_exploration"
    _write_json(phase3 / "manifest.json", '{"status": "complete"}')
    _write_frame(
        pd.DataFrame(
            [
                {
                    "feature_group": "D_dfm_exposure_and_sensitivity",
                    "model": "logistic_regression",
                    "split_name": "leave_one_participant_out",
                    "roc_auc": 0.90,
                    "pr_auc": 0.86,
                    "status": "complete",
                }
            ]
        ),
        phase3_analysis / "participant_prediction_ablation_metrics.csv",
    )

    phase4_analysis = phase4 / "analysis" / "phase4_confirmatory"
    metric_rows = []
    for group, auc, pr, n_features in [
        ("D1_dfm_exposure_only", 0.42, 0.37, 2),
        ("D2_dfm_sensitivity_only", 0.88, 0.84, 2),
        (FINAL_MODEL_GROUP, 0.89, 0.86, len(FINAL_FEATURES)),
        ("D4_dfm_exposure_plus_sensitivity", 0.87, 0.85, 5),
        ("G_all_allowed_non_exposure", 0.88, 0.84, 4),
        ("H_all_except_dfm", 0.60, 0.55, 3),
        ("I_all_except_segmentation", 0.88, 0.84, 4),
        ("J_all_except_raw_speed", 0.88, 0.84, 3),
        ("K_all_except_exposure_variables", 0.89, 0.86, 3),
    ]:
        metric_rows.append(
            {
                "analysis": "phase4_confirmatory_participant_prediction",
                "split_name": "leave_one_participant_out",
                "feature_group": group,
                "model": "logistic_regression",
                "n_features": n_features,
                "n_predictions": 6,
                "usable_folds": 6,
                "skipped_folds": 0,
                "roc_auc": auc,
                "pr_auc": pr,
                "balanced_accuracy": 0.83,
                "macro_f1": 0.83,
                "brier_score": 0.12,
                "calibration_intercept": -0.2,
                "calibration_slope": 0.9,
                "calibration_mean_predicted": 0.5,
                "calibration_observed_rate": 0.5,
                "status": "complete",
                "skip_reason": "",
                "fold_validity": "all_test_predictions_generated",
            }
        )
    _write_frame(pd.DataFrame(metric_rows), phase4_analysis / "confirmatory_prediction_metrics.csv")
    prediction_rows = []
    scores = [0.10, 0.20, 0.35, 0.65, 0.80, 0.90]
    for fold_id, participant_id in enumerate(PARTICIPANTS):
        _, binary = GROUPS[participant_id]
        prediction_rows.append(
            {
                "split_name": "leave_one_participant_out",
                "fold_id": fold_id,
                "model": "logistic_regression",
                "participant_id": participant_id,
                "y_true": binary,
                "y_score": scores[fold_id],
                "y_pred": int(scores[fold_id] >= 0.5),
                "fold_valid": True,
                "feature_group": FINAL_MODEL_GROUP,
                "n_features": len(FINAL_FEATURES),
            }
        )
    _write_frame(pd.DataFrame(prediction_rows), phase4_analysis / "confirmatory_predictions.csv")
    _write_frame(
        pd.DataFrame(
            [
                {
                    "metric": "roc_auc",
                    "feature_group": FINAL_MODEL_GROUP,
                    "model": "logistic_regression",
                    "split_name": "leave_one_participant_out",
                    "observed": 0.89,
                    "n_bootstrap": 1000,
                    "ci_low": 0.74,
                    "ci_high": 0.98,
                },
                {
                    "metric": "pr_auc",
                    "feature_group": FINAL_MODEL_GROUP,
                    "model": "logistic_regression",
                    "split_name": "leave_one_participant_out",
                    "observed": 0.86,
                    "n_bootstrap": 1000,
                    "ci_low": 0.72,
                    "ci_high": 0.97,
                },
            ]
        ),
        phase4_analysis / "bootstrap_results.csv",
    )
    _write_frame(
        pd.DataFrame(
            [
                {
                    "iteration": idx,
                    "feature_group": FINAL_MODEL_GROUP,
                    "model": "logistic_regression",
                    "split_name": "leave_one_participant_out",
                    "roc_auc": 0.2 + idx * 0.01,
                    "status": "complete",
                }
                for idx in range(6)
            ]
        ),
        phase4_analysis / "permutation_results.csv",
    )
    _write_frame(
        pd.DataFrame(
            [
                {
                    "removed_participant_id": participant_id,
                    "removed_reader_group": GROUPS[participant_id][0],
                    "leave_one_dyslexia_labeled": bool(GROUPS[participant_id][1]),
                    "feature_group": FINAL_MODEL_GROUP,
                    "model": "logistic_regression",
                    "split_name": "leave_one_participant_out",
                    "roc_auc": 0.88,
                    "pr_auc": 0.85,
                    "delta_roc_auc": 0.01 if participant_id == "P06" else 0.0,
                    "status": "complete",
                }
                for participant_id in PARTICIPANTS
            ]
        ),
        phase4_analysis / "influence_analysis.csv",
    )
    stability_rows = []
    for fold_id, participant_id in enumerate(PARTICIPANTS):
        for feature_name, coef in zip(FINAL_FEATURES, [0.9, -0.6, 0.4], strict=True):
            stability_rows.append(
                {
                    "feature_group": FINAL_MODEL_GROUP,
                    "fold_id": fold_id,
                    "heldout_participant_id": participant_id,
                    "feature": feature_name,
                    "standardized_logistic_coefficient": coef,
                    "coefficient_sign": "positive" if coef > 0 else "negative",
                }
            )
    _write_frame(pd.DataFrame(stability_rows), phase4_analysis / "feature_stability_by_fold.csv")
    _write_frame(
        pd.DataFrame(
            [
                {
                    "outcome": "log_ffd",
                    "outcome_kind": "linear",
                    "term": "reader_group_binary:dfm_lm_word_surprisal",
                    "phase4_interaction": "reader_group_x_dfm_surprisal",
                    "estimate": 0.08,
                    "std_error": 0.03,
                    "p_value": 0.01,
                    "ci_low": 0.02,
                    "ci_high": 0.14,
                    "n_obs": 60,
                    "model_type": "cluster_robust_ols",
                    "random_effects_attempted": False,
                    "fallback_reason": "mini test",
                    "convergence_warnings": "",
                }
            ]
        ),
        phase4_analysis / "mixed_effects_coefficients.csv",
    )
    _write_json(
        phase4 / "manifest.json",
        '{"status": "complete", "feature_groups": {'
        '"D1_dfm_exposure_only": ["mean_dfm_surprisal_exposure", "mean_dfm_entropy_exposure"],'
        '"D2_dfm_sensitivity_only": ["surprisal_sensitivity_phase3", "entropy_sensitivity_phase3"],'
        f'"{FINAL_MODEL_GROUP}": {FINAL_FEATURES!r},'
        '"D4_dfm_exposure_plus_sensitivity": ["mean_dfm_surprisal_exposure", '
        '"surprisal_sensitivity_phase3", "crossfit_ffd_residual_dfm_surprisal_slope"],'
        '"G_all_allowed_non_exposure": ["crossfit_ffd_residual_dfm_surprisal_slope"],'
        '"H_all_except_dfm": ["mean_ffd"],'
        '"I_all_except_segmentation": ["crossfit_ffd_residual_dfm_surprisal_slope"],'
        '"J_all_except_raw_speed": ["crossfit_ffd_residual_dfm_surprisal_slope"],'
        '"K_all_except_exposure_variables": ["crossfit_ffd_residual_dfm_surprisal_slope"]'
        "}}".replace("'", '"'),
    )
    return {"feature": feature, "label": label, "phase3": phase3, "phase4": phase4}


def _mini_config(paths: dict[str, Path], root: Path) -> dict:
    return {
        "run": {"name": "autoresearch_v1", "output_root": str(root / "results")},
        "autoresearch": {
            "frozen_inputs": {
                "feature_release_dir": str(paths["feature"]),
                "label_release_dir": str(paths["label"]),
                "phase3_dir": str(paths["phase3"]),
                "phase4_dir": str(paths["phase4"]),
            },
            "repo_analysis_dir": str(root / "analysis" / "autoresearch_v1"),
            "output_layout": {
                "tables": str(root / "analysis" / "autoresearch_v1" / "tables"),
                "figures": str(root / "analysis" / "autoresearch_v1" / "figures"),
                "manuscript": str(root / "analysis" / "autoresearch_v1" / "manuscript"),
                "reproducibility": str(root / "analysis" / "autoresearch_v1" / "reproducibility"),
            },
            "no_new_core_labels": True,
            "no_broad_exploratory_feature_expansion": True,
            "allowed_feature_groups": [
                "D1_dfm_exposure_only",
                "D2_dfm_sensitivity_only",
                "D3_dfm_residual_gaze_only",
                "D4_dfm_exposure_plus_sensitivity",
                "G_all_allowed_non_exposure",
                "H_all_except_dfm",
                "I_all_except_segmentation",
                "J_all_except_raw_speed",
                "K_all_except_exposure_variables",
            ],
            "prohibited_variables": [
                "n_words_read",
                "n_speeches",
                "n_word_rows",
                "total_word_rows",
                "word_observation_count",
                "participant_id",
                "reader_group",
                "reader_group_binary",
                "dyslexia_labeled",
                "group_label",
            ],
            "expected": {
                "word_level_rows": 60,
                "sentence_level_rows": len(pd.read_parquet(
                    paths["label"] / "prepared_dataset" / "analysis_ready_sentence_level_v1_1.parquet"
                )),
                "participant_level_rows": 6,
                "dyslexia_labeled": 3,
                "typical_control": 3,
                "parser_status": "surface_heuristic_fallback",
                "segmentation_label_version": "segmentation_orthographic_v1",
                "primary_roc_auc": 0.89,
                "primary_pr_auc": 0.86,
                "metric_tolerance": 0.001,
            },
            "decision_gates": {
                "roc_auc_bootstrap_lower_bound_min": 0.70,
                "permutation_p_value_max": 0.20,
            },
            "refinement_loop": {"max_auc_negligible_gain": 0.01},
        },
    }


def test_autoresearch_config_parsing_and_group_separation(tmp_path: Path) -> None:
    paths = _write_autoresearch_inputs(tmp_path)
    config = _mini_config(paths, tmp_path)
    groups = config["autoresearch"]["allowed_feature_groups"]
    assert "D1_dfm_exposure_only" in groups
    assert "D2_dfm_sensitivity_only" in groups
    assert FINAL_MODEL_GROUP in groups
    assert "n_words_read" in config["autoresearch"]["prohibited_variables"]


def test_final_decision_gates_pass_for_expected_locked_result(tmp_path: Path) -> None:
    paths = _write_autoresearch_inputs(tmp_path)
    config = _mini_config(paths, tmp_path)
    gate_report = evaluate_decision_gates(
        config,
        {"status": "passed", "errors": []},
        {"prediction_count": 6, "unique_prediction_participants": 6},
        {
            "bootstrap": [{"metric": "roc_auc", "ci_low": 0.74}],
            "permutation_p_value": 1 / 7,
            "dfm_exposure_vs_sensitivity": [
                {"feature_group": "D1_dfm_exposure_only", "roc_auc": 0.42},
                {"feature_group": FINAL_MODEL_GROUP, "roc_auc": 0.89},
            ],
            "raw_speed_dominates": False,
        },
        {"prohibited_variables_present": []},
    )
    assert gate_report["all_passed"]


def test_refinement_loop_prefers_primary_over_negligible_complex_gain(tmp_path: Path) -> None:
    paths = _write_autoresearch_inputs(tmp_path)
    config = _mini_config(paths, tmp_path)
    out = tmp_path / "out"
    data = __import__("copco_eye_bench.autoresearch", fromlist=["_load_frozen_data"])._load_frozen_data(
        config, tmp_path
    )
    dirs = __import__("copco_eye_bench.autoresearch", fromlist=["_result_dirs"])._result_dirs(
        config, out, tmp_path
    )
    __import__("copco_eye_bench.autoresearch", fromlist=["_ensure_dirs"])._ensure_dirs(dirs)
    refinement = run_refinement_loop(config, dirs, data)
    assert refinement["selected_candidate"]["candidate"] == "primary_D3"


def test_autoresearch_generates_reports_tables_and_reproducibility(tmp_path: Path) -> None:
    paths = _write_autoresearch_inputs(tmp_path)
    config = _mini_config(paths, tmp_path)
    out = tmp_path / "autoresearch"
    manifest = run_autoresearch(config, out, repo_root=tmp_path)
    assert manifest["status"] == "complete"

    validation = validate_autoresearch(config, out, repo_root=tmp_path)
    assert validation["status"] == "passed", validation["errors"]

    predictions = pd.read_csv(out / "final_model" / "final_model_predictions.csv")
    assert len(predictions) == 6
    assert not predictions["participant_id"].duplicated().any()
    assert not predictions["split_name"].str.contains("random", case=False).any()

    coefficients = pd.read_csv(out / "final_model" / "final_model_coefficients.csv")
    assert not set(coefficients["feature"]).intersection(config["autoresearch"]["prohibited_variables"])

    for path in [
        out / "validation" / "frozen_input_validation_report.md",
        out / "stress_tests" / "dfm_exposure_vs_sensitivity.csv",
        out / "tables" / "dataset_summary_table.csv",
        out / "tables" / "final_model_metrics_table.md",
        out / "manuscript" / "01_abstract_draft.md",
        out / "reproducibility" / "reproduce_autoresearch_only.sh",
        out / "decision" / "final_publication_decision_report.md",
    ]:
        assert path.exists(), path


def test_autoresearch_validation_fails_when_required_output_missing(tmp_path: Path) -> None:
    paths = _write_autoresearch_inputs(tmp_path)
    config = _mini_config(paths, tmp_path)
    out = tmp_path / "autoresearch"
    run_autoresearch(config, out, repo_root=tmp_path)
    (out / "decision" / "final_decision.json").unlink()
    validation = validate_autoresearch(config, out, repo_root=tmp_path)
    assert validation["status"] == "failed"
    assert any("final_decision.json" in error for error in validation["errors"])
