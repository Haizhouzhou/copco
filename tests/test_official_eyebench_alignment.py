from __future__ import annotations

from pathlib import Path

import pandas as pd

from copco_eye_bench.benchmark_bridge import PROHIBITED_FEATURES
from copco_eye_bench.official_eyebench_alignment import (
    ALIGN_RCS_COLUMNS,
    ALIGN_TYP_COLUMNS,
    _feature_columns,
    run_official_eyebench_alignment,
    validate_official_eyebench_alignment,
    validate_official_eyebench_alignment_config,
    validate_official_split_labels,
)
from tests.test_benchmark_bridge import _write_bridge_inputs


def _write_csv(frame: pd.DataFrame, path: Path, *, header: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, header=header)


def _write_fake_eyebench(root: Path, participants: list[str], speeches: list[str]) -> Path:
    eyebench = root / "eyebench"
    labels = eyebench / "data" / "CopCo" / "labels"
    folds = eyebench / "data" / "CopCo" / "folds_metadata"
    (eyebench / "src" / "data" / "preprocessing").mkdir(parents=True, exist_ok=True)
    (eyebench / "src" / "run" / "multi_run").mkdir(parents=True, exist_ok=True)
    (eyebench / "src" / "models").mkdir(parents=True, exist_ok=True)
    (eyebench / "src" / "configs").mkdir(parents=True, exist_ok=True)
    (eyebench / "README.md").write_text("CopCo_TYP CopCo_RCS", encoding="utf-8")
    (eyebench / "environment.yml").write_text("name: eyebench\n", encoding="utf-8")
    (eyebench / "pyproject.toml").write_text(
        "[project]\nname='eyebench'\nlicense={text='MIT License'}\n",
        encoding="utf-8",
    )
    participant_rows = []
    for idx, participant_id in enumerate(participants):
        participant_rows.append(
            {
                "subj": participant_id,
                "comprehension_accuracy": 0.8,
                "number_of_speeches": len(speeches),
                "number_of_questions": 4,
                "absolute_reading_time": 10,
                "relative_reading_time": 0.2,
                "words_per_minute": 250,
                "age": 30,
                "sex": "F",
                "native_language": "Danish",
                "vision": "normal",
                "score_reading_comprehension_test": float(idx + 1),
                "dyslexia": "yes" if idx >= len(participants) // 2 else "no",
                "pseudohomophone_score": 1,
            }
        )
    _write_csv(pd.DataFrame(participant_rows), labels / "participant_stats.csv")
    _write_csv(
        pd.DataFrame(
            {
                "Session_Name_": participants,
                "Trial_Index_": range(len(participants)),
                "Trial_Recycled_": False,
                "condition": "experiment",
                "text": "text",
                "question": "NO QUESTION",
                "expected_key": 1,
                "QUESTION_KEY_PRESSED": 1,
                "QUESTION_RT": 1,
                "QUESTION_ACCURACY": 1,
                "SENTENCE_RT": 1,
                "speechid": speeches[0],
                "paragraphid": 0,
                "counterbalance": 0,
                "LAST_COUNTERBALANCE": 0,
            }
        ),
        labels / "stimuli_and_comp_results.csv",
    )
    mapping_rows = []
    for speech_id in speeches:
        for word_id in range(12):
            mapping_rows.append(
                {
                    "index": 0,
                    "part": 1,
                    "speechId": speech_id,
                    "paragraphId": 0,
                    "wordId": word_id,
                    "word": f"word{word_id}",
                    "characters": "[]",
                    "char_IA_ids": "[]",
                    "sentenceId": 0,
                }
            )
    _write_csv(pd.DataFrame(mapping_rows), labels / "word2char_IA_mapping.csv")
    all_trials = [
        {
            "unique_paragraph_id": f"{speech_id}_0",
            "participant_id": participant_id,
            "unique_trial_id": f"{participant_id}_{speech_id}_0",
            "RCS_score": float(participants.index(participant_id) + 1),
            "dyslexia": int(participants.index(participant_id) >= len(participants) // 2),
            "speech_id": speech_id,
        }
        for participant_id in participants
        for speech_id in speeches
    ]
    all_trials_frame = pd.DataFrame(all_trials)
    for fold_id in range(4):
        test_participants = set(participants[fold_id::4])
        test_speech = speeches[fold_id % len(speeches)]
        train = all_trials_frame[
            ~all_trials_frame["participant_id"].isin(test_participants)
            & ~all_trials_frame["speech_id"].eq(test_speech)
        ].copy()
        unseen_reader = all_trials_frame[
            all_trials_frame["participant_id"].isin(test_participants)
            & ~all_trials_frame["speech_id"].eq(test_speech)
        ].copy()
        unseen_text = all_trials_frame[
            ~all_trials_frame["participant_id"].isin(test_participants)
            & all_trials_frame["speech_id"].eq(test_speech)
        ].copy()
        strict = all_trials_frame[
            all_trials_frame["participant_id"].isin(test_participants)
            & all_trials_frame["speech_id"].eq(test_speech)
        ].copy()
        frames = []
        for regime, frame in [
            ("train_train", train),
            ("test_unseen_subject_seen_item", unseen_reader),
            ("test_seen_subject_unseen_item", unseen_text),
            ("test_unseen_subject_unseen_item", strict),
        ]:
            copy = frame.copy()
            copy["regime"] = regime
            frames.append(copy)
        _write_csv(
            pd.concat(frames, ignore_index=True),
            folds / "trial_ids" / f"fold_{fold_id}_trial_ids_by_regime.csv",
        )
        _write_csv(pd.DataFrame(sorted(test_participants)), folds / "subjects" / f"fold_{fold_id}.csv", header=False)
        _write_csv(pd.DataFrame([test_speech]), folds / "items" / f"fold_{fold_id}.csv", header=False)
    result_dir = eyebench / "results" / "formatted_eyebench_benchmark_results"
    _write_csv(
        pd.DataFrame(
            [
                {
                    "Model": "AhnCNN",
                    "Unseen Reader_\\makecell{Balanced\\\\Accuracy}": "77.7 +/- 1.8",
                    "Unseen Text_\\makecell{Balanced\\\\Accuracy}": "77.5 +/- 2.7",
                    "Unseen Text \\& Reader_\\makecell{Balanced\\\\Accuracy}": "65.6 +/- 2.4",
                    "Average_\\makecell{Balanced\\\\Accuracy}": "75.0 +/- 0.8",
                    "Unseen Reader_AUROC": "85.3 +/- 1.6",
                    "Unseen Text_AUROC": "85.7 +/- 2.3",
                    "Unseen Text \\& Reader_AUROC": "74.9 +/- 2.8",
                    "Average_AUROC": "83.4 +/- 1.1",
                }
            ]
        ),
        result_dir / "CopCo_TYP_test.csv",
    )
    _write_csv(
        pd.DataFrame(
            [
                {
                    "Model": "Random Forest",
                    "Unseen Reader_RMSE": "2.5 +/- 0.1",
                    "Unseen Text_RMSE": "2.5 +/- 0.1",
                    "Unseen Text \\& Reader_RMSE": "2.5 +/- 0.1",
                    "Average_RMSE": "2.5 +/- 0.1",
                    "Unseen Reader_MAE": "2.0 +/- 0.1",
                    "Unseen Text_MAE": "2.0 +/- 0.1",
                    "Unseen Text \\& Reader_MAE": "2.0 +/- 0.1",
                    "Average_MAE": "2.0 +/- 0.1",
                    "Unseen Reader_R²": "0.07 +/- 0.0",
                    "Unseen Text_R²": "0.07 +/- 0.0",
                    "Unseen Text \\& Reader_R²": "0.08 +/- 0.0",
                    "Average_R²": "0.08 +/- 0.0",
                }
            ]
        ),
        result_dir / "CopCo_RCS_test.csv",
    )
    return eyebench


def _mini_alignment_config(paths: dict[str, Path], eyebench: Path, tmp_path: Path) -> dict:
    return {
        "run": {"name": "official_eyebench_alignment_v1", "output_root": str(tmp_path / "results")},
        "official_eyebench_alignment": {
            "version": "v1",
            "eyebench": {
                "path": str(eyebench),
                "repository_url": "https://github.com/EyeBench/eyebench.git",
                "environment_name": "missing_eyebench_test_env",
                "allow_data_download": False,
            },
            "frozen_inputs": {
                "prepared_dataset_dir": str(paths["prepared"]),
                "label_release_dir": str(paths["label_dir"]),
                "benchmark_bridge_dir": str(tmp_path / "benchmark_bridge"),
            },
            "repo_analysis_dir": str(tmp_path / "analysis" / "official_eyebench_alignment_v1"),
            "deterministic_seed": 131,
            "no_new_labels": True,
            "no_feature_engineering_search": True,
            "no_broad_model_search": True,
            "forbid_random_word_level_split": True,
            "tasks": ["CopCo_TYP", "CopCo_RCS"],
            "split_regimes": ["unseen_reader", "unseen_text", "unseen_reader_and_text"],
            "prohibited_features": sorted(
                PROHIBITED_FEATURES | {"unique_trial_id", "unique_paragraph_id", "RCS_score"}
            ),
            "residualization": {"min_words_for_slope": 8, "max_parallel_folds": 1, "reader_group_never_used": True},
            "rcs": {"target_column": "eyebench_rcs_score", "missing_target_value": -1, "scale": "EyeBench_RCS_score"},
            "decision_gates": {
                "CopCo_TYP": {"formatted_table": str(eyebench / "results" / "formatted_eyebench_benchmark_results" / "CopCo_TYP_test.csv")},
                "CopCo_RCS": {"formatted_table": str(eyebench / "results" / "formatted_eyebench_benchmark_results" / "CopCo_RCS_test.csv")},
            },
        },
    }


def _write_minimal_benchmarkbridge_reference(tmp_path: Path) -> None:
    bb = tmp_path / "benchmark_bridge"
    typ_rows = []
    rcs_rows = []
    for split_name in ["unseen_reader", "unseen_text", "unseen_reader_and_text"]:
        for level in ["participant_text_trial", "reader_aggregated"]:
            typ_rows.append(
                {
                    "task": "CopCo_TYP",
                    "split_name": split_name,
                    "feature_group": "D3_dfm_residual_gaze_only",
                    "model": "logistic_regression",
                    "evaluation_level": level,
                    "n_features": 2,
                    "n_predictions": 8,
                    "usable_folds": 4,
                    "skipped_folds": 0,
                    "roc_auc": 0.8,
                    "pr_auc": 0.7,
                    "balanced_accuracy": 0.75,
                    "macro_f1": 0.74,
                    "brier_score": 0.2,
                    "status": "complete",
                    "skip_reason": "",
                }
            )
            rcs_rows.append(
                {
                    "task": "CopCo_RCS",
                    "split_name": split_name,
                    "feature_group": "D3_dfm_residual_gaze_only",
                    "model": "ridge_regression",
                    "evaluation_level": level,
                    "target": "comprehension_score",
                    "target_scale": "raw_project_scale",
                    "n_features": 2,
                    "n_predictions": 8,
                    "usable_folds": 4,
                    "skipped_folds": 0,
                    "rmse": 0.1,
                    "mae": 0.08,
                    "r2": 0.01,
                    "status": "complete",
                    "skip_reason": "",
                }
            )
    _write_csv(pd.DataFrame(typ_rows), bb / "typ" / "typ_benchmark_metrics.csv")
    _write_csv(pd.DataFrame(rcs_rows), bb / "rcs" / "rcs_benchmark_metrics.csv")


def test_official_alignment_config_parses() -> None:
    import yaml

    config = yaml.safe_load(Path("configs/official_eyebench_alignment_v1.yaml").read_text())
    assert validate_official_eyebench_alignment_config(config)["status"] == "passed"


def test_official_alignment_end_to_end_and_validation(tmp_path: Path) -> None:
    paths = _write_bridge_inputs(tmp_path)
    participants = [f"P{idx:02d}" for idx in range(1, 9)]
    speeches = ["S1", "S2", "S3", "S4"]
    eyebench = _write_fake_eyebench(tmp_path, participants, speeches)
    _write_minimal_benchmarkbridge_reference(tmp_path)
    config = _mini_alignment_config(paths, eyebench, tmp_path)
    out = tmp_path / "results" / "official_eyebench_alignment_v1_test"

    manifest = run_official_eyebench_alignment(config, out, repo_root=tmp_path)
    report = validate_official_eyebench_alignment(config, out, repo_root=tmp_path)

    assert manifest["status"] == "complete"
    assert report["status"] == "passed", report
    assert (tmp_path / "docs" / "eyebench_vendor_manifest.md").exists()
    audit = pd.read_csv(tmp_path / "analysis" / "official_eyebench_alignment_v1" / "copco_alignment_audit.csv")
    assert {"field", "value"}.issubset(audit.columns)
    assert int(audit.loc[audit["field"].eq("n_common_participants"), "value"].iloc[0]) == 8
    splits = pd.read_parquet(out / "splits" / "official_eyebench_split_labels.parquet")
    split_errors, _ = validate_official_split_labels(splits)
    assert split_errors == []
    assert not splits["split_name"].str.contains("random", case=False, na=False).any()
    typ = pd.read_csv(out / "typ" / "typ_official_alignment_metrics.csv")
    rcs = pd.read_csv(out / "rcs" / "rcs_official_alignment_metrics.csv")
    assert set(ALIGN_TYP_COLUMNS).issubset(typ.columns)
    assert set(ALIGN_RCS_COLUMNS).issubset(rcs.columns)
    assert "eyebench_folds_full_feature_intersection" in set(typ["mode"])
    samples = pd.read_parquet(out / "data" / "official_alignment_trial_features.parquet")
    feature_groups = _feature_columns(samples)
    prohibited = set(config["official_eyebench_alignment"]["prohibited_features"])
    assert not set(feature_groups["D3_dfm_residual_gaze_only"]).intersection(prohibited)
    decision = (tmp_path / "analysis" / "official_eyebench_alignment_v1" / "official_eyebench_decision_report.md").read_text()
    assert "benchmark_relative_sota_only" in decision or "blocked_by_environment" in decision
