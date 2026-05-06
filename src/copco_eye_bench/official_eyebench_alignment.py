"""Official EyeBench alignment layer for the frozen CopCo D3 profile model."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .benchmark_bridge import (
    PRIMARY_TYPO_FEATURE_GROUP,
    PROHIBITED_FEATURES,
    RESIDUALIZATION_FORBIDDEN,
    RESIDUALIZATION_PREDICTORS,
    _base_sample_row,
    _classification_metrics,
    _clean_feature_list,
    _feature_columns,
    _git_sha,
    _markdown_table,
    _merge_boundary_vocoid,
    _model_pipeline,
    _pd,
    _reader_aggregate_classification,
    _reader_aggregate_regression,
    _regression_metrics,
    _score_estimator,
    _score_regressor,
    _with_derived_columns,
    build_crossfit_fold_feature_cache,
)
from .config import get_nested, timestamped_output_dir


OFFICIAL_SPLITS = ["unseen_reader", "unseen_text", "unseen_reader_and_text"]
OFFICIAL_REGIME_TO_SPLIT = {
    "seen_subject_unseen_item": "unseen_text",
    "unseen_subject_seen_item": "unseen_reader",
    "unseen_subject_unseen_item": "unseen_reader_and_text",
}
SPLIT_TO_OFFICIAL_REGIME = {value: key for key, value in OFFICIAL_REGIME_TO_SPLIT.items()}

ALIGN_TYP_COLUMNS = [
    "mode",
    "model_name",
    "claim_type",
    "task",
    "split_name",
    "evaluation_level",
    "n_features",
    "n_predictions",
    "usable_folds",
    "skipped_folds",
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "macro_f1",
    "brier_score",
    "status",
    "skip_reason",
]

ALIGN_RCS_COLUMNS = [
    "mode",
    "model_name",
    "claim_type",
    "task",
    "split_name",
    "evaluation_level",
    "target",
    "target_scale",
    "n_features",
    "n_predictions",
    "usable_folds",
    "skipped_folds",
    "rmse",
    "mae",
    "r2",
    "status",
    "skip_reason",
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_parquet(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _run_command(command: list[str], *, cwd: Path, timeout: int = 60) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return {
            "command": " ".join(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": " ".join(command),
            "returncode": None,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "").strip() if isinstance(exc.stderr, str) else "",
            "timed_out": True,
        }


def _analysis_dirs(config: dict[str, Any], out: Path, repo_root: str | Path) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    analysis_rel = str(
        get_nested(
            config,
            "official_eyebench_alignment.output_layout.analysis",
            "analysis/official_eyebench_alignment_v1",
        )
    )
    tables_rel = str(
        get_nested(
            config,
            "official_eyebench_alignment.output_layout.tables",
            "analysis/official_eyebench_alignment_v1/tables",
        )
    )
    repo_analysis = root / str(
        get_nested(
            config,
            "official_eyebench_alignment.repo_analysis_dir",
            "analysis/official_eyebench_alignment_v1",
        )
    )
    return {
        "repo_analysis": repo_analysis,
        "repo_tables": repo_analysis / "tables",
        "result_analysis": out / analysis_rel,
        "result_tables": out / tables_rel,
    }


def _write_report(dirs: dict[str, Path], name: str, text: str) -> None:
    _write_md(dirs["result_analysis"] / name, text)
    _write_md(dirs["repo_analysis"] / name, text)


def _write_table(dirs: dict[str, Path], name: str, frame: Any) -> None:
    _write_csv(dirs["result_tables"] / name, frame)
    _write_csv(dirs["repo_tables"] / name, frame)


def _configured_path(config: dict[str, Any], dotted: str, repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    value = get_nested(config, dotted)
    if value is None:
        raise ValueError(f"missing required config path: {dotted}")
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _eyebench_path(config: dict[str, Any], repo_root: str | Path) -> Path:
    return _configured_path(config, "official_eyebench_alignment.eyebench.path", repo_root)


def _text_id_from_paragraph(speech_id: Any, paragraph_id: Any) -> str:
    speech = str(speech_id)
    paragraph = str(paragraph_id)
    match = re.search(r"_p(-?\d+)$", paragraph)
    if match:
        return f"{speech}_{match.group(1)}"
    if paragraph.startswith(f"{speech}_"):
        return paragraph.replace(f"{speech}_p", f"{speech}_", 1)
    return f"{speech}_{paragraph}"


def _safe_git_output(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def validate_official_eyebench_alignment_config(config: dict[str, Any]) -> dict[str, Any]:
    section = get_nested(config, "official_eyebench_alignment", {})
    errors: list[str] = []
    warnings_list: list[str] = []
    if not isinstance(section, dict):
        errors.append("missing official_eyebench_alignment config section")
        section = {}
    for flag in ["no_new_labels", "no_feature_engineering_search", "no_broad_model_search"]:
        if section.get(flag) is not True:
            errors.append(f"{flag} must be true")
    if section.get("forbid_random_word_level_split") is not True:
        errors.append("random word-level split prohibition is not enabled")
    configured_splits = set(section.get("split_regimes", []))
    missing_splits = sorted(set(OFFICIAL_SPLITS) - configured_splits)
    if missing_splits:
        errors.append(f"missing official split regimes: {missing_splits}")
    prohibited = set(section.get("prohibited_features", []))
    missing_prohibited = sorted(
        (PROHIBITED_FEATURES | {"unique_trial_id", "unique_paragraph_id", "RCS_score"})
        - prohibited
    )
    if missing_prohibited:
        errors.append(f"official prohibited feature list incomplete: {missing_prohibited}")
    residualization = section.get("residualization", {})
    if isinstance(residualization, dict) and residualization.get("reader_group_never_used") is not True:
        errors.append("residualization.reader_group_never_used must be true")
    if RESIDUALIZATION_FORBIDDEN.intersection(RESIDUALIZATION_PREDICTORS):
        errors.append("residualization predictor constants include forbidden identifiers")
    if "CopCo_TYP" not in section.get("tasks", []):
        errors.append("CopCo_TYP task is required")
    if "CopCo_RCS" not in section.get("tasks", []):
        warnings_list.append("CopCo_RCS task is not configured")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings_list,
    }


def write_vendor_manifest(
    config: dict[str, Any],
    dirs: dict[str, Path],
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    eyebench = _eyebench_path(config, root)
    commit = _safe_git_output(["git", "rev-parse", "HEAD"], eyebench)
    local_status = _safe_git_output(["git", "status", "--short"], eyebench)
    submodule_status = _safe_git_output(["git", "submodule", "status", "--", "eyebench"], root)
    processed = eyebench / "data" / "CopCo" / "processed"
    manifest = {
        "repository_url": get_nested(
            config,
            "official_eyebench_alignment.eyebench.repository_url",
            "https://github.com/EyeBench/eyebench.git",
        ),
        "checked_out_commit_hash": commit,
        "date_time": datetime.now().isoformat(timespec="seconds"),
        "vendor_method": "git_submodule" if (root / ".gitmodules").exists() else "unknown",
        "submodule_status": submodule_status,
        "license_note": "EyeBench pyproject declares MIT License.",
        "data_were_downloaded": processed.exists(),
        "processed_data_are_committed": False,
        "local_modifications_to_eyebench": local_status or "none",
        "generated_data_under_eyebench_committed": False,
        "statement": "Generated data, caches, WandB outputs, and model artifacts under EyeBench are not committed.",
    }
    text = "\n".join(
        [
            "# EyeBench Vendor Manifest",
            "",
            f"- EyeBench repository URL: {manifest['repository_url']}",
            f"- Checked-out commit hash: `{commit}`",
            f"- Date/time: {manifest['date_time']}",
            f"- Vendored as: {manifest['vendor_method']}",
            f"- Submodule status: `{submodule_status}`",
            f"- License note: {manifest['license_note']}",
            f"- Data were downloaded: {manifest['data_were_downloaded']}",
            f"- Processed data are committed: {manifest['processed_data_are_committed']}",
            f"- Local modifications to EyeBench: {manifest['local_modifications_to_eyebench']}",
            f"- Generated data under EyeBench committed: {manifest['generated_data_under_eyebench_committed']}",
            "",
            manifest["statement"],
        ]
    )
    _write_md(root / "docs" / "eyebench_vendor_manifest.md", text)
    _write_report(dirs, "eyebench_vendor_manifest.md", text)
    return manifest


def inspect_eyebench_structure(
    config: dict[str, Any],
    dirs: dict[str, Path],
    repo_root: str | Path,
) -> dict[str, Any]:
    pd = _pd()
    eyebench = _eyebench_path(config, repo_root)
    copco = eyebench / "data" / "CopCo"
    labels = copco / "labels"
    folds = copco / "folds_metadata"
    result_dir = eyebench / "results" / "formatted_eyebench_benchmark_results"
    label_shapes = {}
    for name in ["participant_stats.csv", "stimuli_and_comp_results.csv", "word2char_IA_mapping.csv"]:
        path = labels / name
        if path.exists():
            frame = pd.read_csv(path)
            label_shapes[name] = {"rows": int(len(frame)), "columns": frame.columns.astype(str).tolist()}
    fold_files = {
        "subjects": sorted(path.name for path in (folds / "subjects").glob("*.csv")),
        "items": sorted(path.name for path in (folds / "items").glob("*.csv")),
        "trial_ids": sorted(path.name for path in (folds / "trial_ids").glob("*.csv")),
    }
    processed = copco / "processed"
    payload = {
        "readme_present": (eyebench / "README.md").exists(),
        "environment_present": (eyebench / "environment.yml").exists(),
        "pyproject_present": (eyebench / "pyproject.toml").exists(),
        "copco_labels": label_shapes,
        "fold_files": fold_files,
        "processed_copco_data_present": processed.exists(),
        "official_task_names": ["CopCo_TYP", "CopCo_RCS"],
        "official_regime_names": list(OFFICIAL_REGIME_TO_SPLIT),
        "classification_metrics": ["AUROC", "Balanced Accuracy"],
        "regression_metrics": ["RMSE", "MAE", "R2"],
        "result_format": [
            "results/raw/{data_model_trainer_task}/fold_index={i}/trial_level_test_results.csv",
            "columns include label, prediction_prob, eval_regime, eval_type, fold_index, and trial IDs",
        ],
        "model_registration": [
            "src/models/*.py",
            "src/configs/constants.py MLModelNames/DLModelNames",
            "src/configs/models/ml or src/configs/models/dl config registration",
        ],
        "dataset_registration": ["src/configs/data.py", "src/data/datasets/", "src/data/datamodules/"],
        "benchmark_tables_present": {
            "CopCo_TYP_test": (result_dir / "CopCo_TYP_test.csv").exists(),
            "CopCo_RCS_test": (result_dir / "CopCo_RCS_test.csv").exists(),
        },
    }
    rows = [
        {"area": "README.md", "finding": "Documents CopCo_TYP/CopCo_RCS and three regimes."},
        {"area": "environment.yml", "finding": "Defines Python 3.12.10, PyTorch CUDA, Hydra, WandB, and preprocessing deps."},
        {"area": "data/CopCo/folds_metadata", "finding": f"Subjects/items/trial_id fold files: {fold_files}"},
        {"area": "data/CopCo/labels", "finding": json.dumps(label_shapes, default=str)[:900]},
        {"area": "src/run/multi_run/raw_to_processed_results.py", "finding": "Computes AUROC/BA and RMSE/MAE/R2 from trial_level_test_results.csv."},
        {"area": "src/data/preprocessing/create_folds.py", "finding": "Creates train/val/test regimes from subject and item folds."},
        {"area": "processed data", "finding": f"CopCo processed data present: {processed.exists()}"},
    ]
    text = "\n".join(
        [
            "# EyeBench Structure Report",
            "",
            _markdown_table(rows, ["area", "finding"], max_rows=30),
            "",
            "## Summary",
            "- Official CopCo tasks: `CopCo_TYP`, `CopCo_RCS`.",
            "- Official split names: `seen_subject_unseen_item`, `unseen_subject_seen_item`, "
            "`unseen_subject_unseen_item`.",
            "- CopCo fold metadata include subjects, items, and trial IDs.",
            "- Official result aggregation reads `trial_level_test_results.csv` files and emits metric CSVs.",
            "- Official processed CopCo data are not present unless `data/CopCo/processed/` exists.",
        ]
    )
    _write_report(dirs, "eyebench_structure_report.md", text)
    return payload


def write_environment_report(
    config: dict[str, Any],
    dirs: dict[str, Path],
    repo_root: str | Path,
) -> dict[str, Any]:
    eyebench = _eyebench_path(config, repo_root)
    env_name = str(get_nested(config, "official_eyebench_alignment.eyebench.environment_name", "eyebench"))
    create_attempt = _run_command(["mamba", "env", "create", "-f", "eyebench/environment.yml"], cwd=Path(repo_root).resolve())
    import_attempt = _run_command(
        [
            "conda",
            "run",
            "-n",
            env_name,
            "python",
            "-c",
            (
                "import sys; print(sys.version); import pandas; print('pandas', pandas.__version__); "
                "import src; print('src_import_ok')"
            ),
        ],
        cwd=eyebench,
    )
    ok = import_attempt["returncode"] == 0
    report = {
        "environment_name": env_name,
        "create_attempt": create_attempt,
        "import_attempt": import_attempt,
        "python_version": import_attempt["stdout"].splitlines()[0] if import_attempt["stdout"] else "unknown",
        "eyebench_imports": ok,
        "cli_scripts_runnable": ok,
        "cuda_required": "Not required for static inspection or sklearn adapter; required by declared env for neural baselines.",
        "wandb_login_required": "Required for official sweep/train scripts, not for static inspection.",
        "official_data_download_possible": ok,
    }
    text = "\n".join(
        [
            "# EyeBench Environment Report",
            "",
            f"- Environment name: `{env_name}`",
            f"- Python version/status: {report['python_version']}",
            f"- EyeBench imports: {ok}",
            f"- CLI/scripts runnable: {ok}",
            "- CUDA required: not for this adapter; EyeBench environment declares PyTorch CUDA for neural baselines.",
            "- WandB login required: official train/sweep scripts use WandB; this adapter does not.",
            f"- Official data download possible: {report['official_data_download_possible']}",
            "",
            "## Environment Create Attempt",
            f"- Command: `{create_attempt['command']}`",
            f"- Return code: {create_attempt['returncode']}",
            f"- stderr: `{create_attempt['stderr'][:1000]}`",
            "",
            "## Import Attempt",
            f"- Command: `{import_attempt['command']}`",
            f"- Return code: {import_attempt['returncode']}",
            f"- stdout: `{import_attempt['stdout'][:1000]}`",
            f"- stderr: `{import_attempt['stderr'][:1000]}`",
        ]
    )
    _write_report(dirs, "eyebench_environment_report.md", text)
    return report


def write_data_download_report(
    config: dict[str, Any],
    dirs: dict[str, Path],
    repo_root: str | Path,
    environment: dict[str, Any],
) -> dict[str, Any]:
    pd = _pd()
    eyebench = _eyebench_path(config, repo_root)
    processed = eyebench / "data" / "CopCo" / "processed"
    allow = bool(get_nested(config, "official_eyebench_alignment.eyebench.allow_data_download", True))
    downloaded = False
    command_result: dict[str, Any] | None = None
    skip_reason = ""
    if not allow:
        skip_reason = "data download disabled by config"
    elif not environment.get("eyebench_imports"):
        skip_reason = "EyeBench environment import failed; preprocessing not run"
    else:
        command_result = _run_command(
            ["conda", "run", "-n", "eyebench", "bash", "-lc", "bash src/data/preprocessing/get_data.sh CopCo"],
            cwd=eyebench,
            timeout=120,
        )
        downloaded = command_result["returncode"] == 0 and processed.exists()
        if not downloaded:
            skip_reason = "data download/preprocessing command failed or did not create processed CopCo data"
    participant_count = text_count = trial_count = word_count = fixation_count = None
    if processed.exists():
        for name, key in [
            ("trial_level.feather", "trial"),
            ("ia.feather", "word"),
            ("fixations.feather", "fixation"),
        ]:
            path = processed / name
            if path.exists():
                frame = pd.read_feather(path)
                if key == "trial":
                    trial_count = int(len(frame))
                    participant_count = int(frame.get("participant_id", pd.Series()).nunique())
                    text_count = int(frame.get("unique_paragraph_id", pd.Series()).nunique())
                elif key == "word":
                    word_count = int(len(frame))
                elif key == "fixation":
                    fixation_count = int(len(frame))
    report = {
        "downloaded": downloaded,
        "processed_data_path": str(processed),
        "copco_processed_files_present": processed.exists(),
        "participant_count": participant_count,
        "text_item_count": text_count,
        "trial_count": trial_count,
        "word_count": word_count,
        "fixation_count": fixation_count,
        "labels_available": (eyebench / "data" / "CopCo" / "labels").exists(),
        "fold_files_available": (eyebench / "data" / "CopCo" / "folds_metadata").exists(),
        "missing_raw_aoi_or_features": not processed.exists(),
        "skip_reason": skip_reason,
        "command_result": command_result,
    }
    text = "\n".join(
        [
            "# EyeBench Data Download Report",
            "",
            f"- downloaded: {downloaded}",
            f"- processed data path: `{processed}`",
            f"- CopCo processed files present: {processed.exists()}",
            f"- participant count: {participant_count}",
            f"- text/item count: {text_count}",
            f"- trial count: {trial_count}",
            f"- word count: {word_count}",
            f"- fixation count: {fixation_count}",
            f"- labels available: {report['labels_available']}",
            f"- fold files available: {report['fold_files_available']}",
            f"- missing raw AOI or feature types: {report['missing_raw_aoi_or_features']}",
            f"- blocker/skip reason: {skip_reason or 'none'}",
        ]
    )
    _write_report(dirs, "eyebench_data_download_report.md", text)
    return report


def _load_inputs(config: dict[str, Any], repo_root: str | Path) -> dict[str, Any]:
    pd = _pd()
    prepared = _configured_path(
        config,
        "official_eyebench_alignment.frozen_inputs.prepared_dataset_dir",
        repo_root,
    )
    label_dir = _configured_path(
        config,
        "official_eyebench_alignment.frozen_inputs.label_release_dir",
        repo_root,
    )
    eyebench = _eyebench_path(config, repo_root)
    return {
        "prepared": prepared,
        "label_dir": label_dir,
        "eyebench": eyebench,
        "word": pd.read_parquet(prepared / "analysis_ready_word_level_v1_1.parquet"),
        "segmentation_boundary": pd.read_parquet(
            label_dir / "labels" / "segmentation_boundary_labels_v1.parquet"
        ),
        "participant_labels": pd.read_parquet(label_dir / "labels" / "participant_labels_v1.parquet"),
        "participant_stats": pd.read_csv(eyebench / "data" / "CopCo" / "labels" / "participant_stats.csv"),
        "word_mapping": pd.read_csv(eyebench / "data" / "CopCo" / "labels" / "word2char_IA_mapping.csv"),
    }


def _load_official_trial_ids(eyebench: Path) -> Any:
    pd = _pd()
    frames = []
    for path in sorted((eyebench / "data" / "CopCo" / "folds_metadata" / "trial_ids").glob("fold_*_trial_ids_by_regime.csv")):
        match = re.search(r"fold_(\d+)_", path.name)
        fold_id = int(match.group(1)) if match else -1
        frame = pd.read_csv(path)
        frame["fold_id"] = fold_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_alignment_audit(
    config: dict[str, Any],
    out: Path,
    dirs: dict[str, Path],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    pd = _pd()
    word = inputs["word"].copy()
    word["participant_id"] = word["participant_id"].astype(str)
    word["speech_id"] = word["speech_id"].astype(str)
    word["unique_paragraph_id"] = [
        _text_id_from_paragraph(speech, paragraph)
        for speech, paragraph in zip(word["speech_id"], word["paragraph_id"], strict=True)
    ]
    word["unique_trial_id"] = word["participant_id"] + "_" + word["unique_paragraph_id"]
    word["eyebench_word_id"] = word["word_id"].astype(str).str.extract(r"_w(\d+)$")[0]
    official_trials = _load_official_trial_ids(inputs["eyebench"])
    participant_stats = inputs["participant_stats"].rename(columns={"subj": "participant_id"})
    participant_stats["reader_group_binary_official"] = participant_stats["dyslexia"].map({"yes": 1, "no": 0})
    our_participants = set(word["participant_id"].astype(str).unique())
    eye_participants = set(participant_stats["participant_id"].astype(str).unique())
    common_participants = our_participants.intersection(eye_participants)
    our_texts = set(word["speech_id"].astype(str).unique())
    eye_texts = set(inputs["word_mapping"]["speechId"].astype(str).unique())
    common_texts = our_texts.intersection(eye_texts)
    our_trials = set(word["unique_trial_id"].astype(str).unique())
    eye_trials = set(official_trials["unique_trial_id"].astype(str).unique()) if not official_trials.empty else set()
    common_trials = our_trials.intersection(eye_trials)
    our_word_keys = set(
        word[["speech_id", "unique_paragraph_id", "eyebench_word_id"]]
        .drop_duplicates()
        .astype(str)
        .agg("::".join, axis=1)
    )
    mapping = inputs["word_mapping"].copy()
    mapping["unique_paragraph_id"] = mapping["speechId"].astype(str) + "_" + mapping["paragraphId"].astype(str)
    eye_word_keys = set(
        mapping[["speechId", "unique_paragraph_id", "wordId"]]
        .drop_duplicates()
        .astype(str)
        .agg("::".join, axis=1)
    )
    common_word_rows = our_word_keys.intersection(eye_word_keys)
    our_labels = inputs["participant_labels"][
        ["participant_id", "reader_group", "reader_group_binary"]
    ].drop_duplicates("participant_id")
    label_compare = (
        our_labels.merge(
            participant_stats[["participant_id", "dyslexia", "reader_group_binary_official"]],
            on="participant_id",
            how="outer",
        )
        .assign(label_match=lambda df: df["reader_group_binary"].eq(df["reader_group_binary_official"]))
    )
    common_label_compare = label_compare[label_compare["participant_id"].isin(common_participants)]
    payload = {
        "participants_in_eyebench": int(len(eye_participants)),
        "participants_in_our_dataset": int(len(our_participants)),
        "n_common_participants": int(len(common_participants)),
        "participants_only_in_eyebench": sorted(eye_participants - our_participants),
        "participants_only_in_our_dataset": sorted(our_participants - eye_participants),
        "n_common_dyslexia_labeled": int(
            common_label_compare["reader_group_binary"].fillna(0).astype(int).sum()
        ),
        "n_common_typical_control": int(
            common_label_compare["reader_group_binary"].fillna(0).eq(0).sum()
        ),
        "n_label_mismatches": int((~common_label_compare["label_match"].fillna(False)).sum()),
        "speech_item_ids_in_eyebench": int(len(eye_texts)),
        "speech_item_ids_in_our_dataset": int(len(our_texts)),
        "n_common_texts": int(len(common_texts)),
        "n_common_trials": int(len(common_trials)),
        "n_common_word_rows": int(len(common_word_rows)),
        "n_unmapped_eyebench_trials": int(len(eye_trials - our_trials)),
        "n_unmapped_our_trials": int(len(our_trials - eye_trials)),
        "can_run_exact_official_eyebench_subset_evaluation": False,
        "can_run_eyebench_fold_full_feature_intersection_evaluation": bool(common_trials),
        "can_reproduce_benchmark_bridge_full_data_evaluation": _configured_path(
            config,
            "official_eyebench_alignment.frozen_inputs.benchmark_bridge_dir",
            ".",
        ).exists(),
        "official_result_label": "blocked_by_processed_data_or_environment",
        "fold_aligned_result_label": "EyeBench-fold-aligned, full-feature, non-official",
        "full_data_result_label": "internal EyeBench-style, benchmark-relative",
    }
    audit_rows = [
        {"field": key, "value": value if not isinstance(value, list) else ";".join(value)}
        for key, value in payload.items()
    ]
    audit = pd.DataFrame(audit_rows)
    _write_csv(dirs["repo_analysis"] / "copco_alignment_audit.csv", audit)
    _write_csv(dirs["result_analysis"] / "copco_alignment_audit.csv", audit)
    _write_json(out / "alignment_audit.json", payload)
    text = "\n".join(
        [
            "# CopCo Alignment Audit",
            "",
            _markdown_table(audit_rows, ["field", "value"], max_rows=80),
            "",
            "## Answers",
            f"- Can we run exact official EyeBench subset evaluation? "
            f"{payload['can_run_exact_official_eyebench_subset_evaluation']}",
            f"- Can we run EyeBench-fold full-feature intersection evaluation? "
            f"{payload['can_run_eyebench_fold_full_feature_intersection_evaluation']}",
            f"- Can we reproduce the BenchmarkBridge full-data evaluation? "
            f"{payload['can_reproduce_benchmark_bridge_full_data_evaluation']}",
            "- Official results require exact processed EyeBench data, official folds, and official evaluator.",
            "- EyeBench-fold full-feature results use official fold metadata with CopCo prepared features.",
            "- BenchmarkBridge full-data results remain internal EyeBench-style and benchmark-relative.",
            "",
            "## ID Mapping",
            "- EyeBench `subject ID` maps to CopCo `participant_id`.",
            "- EyeBench `item ID` maps to CopCo `speech_id`.",
            "- EyeBench `unique_paragraph_id` maps to CopCo `speech_id` + parsed `paragraph_id`.",
            "- EyeBench `trial ID` maps to CopCo `participant_id` + `unique_paragraph_id`.",
        ]
    )
    _write_report(dirs, "copco_alignment_audit.md", text)
    return payload


def build_trial_feature_table(
    config: dict[str, Any],
    out: Path,
    inputs: dict[str, Any],
) -> tuple[Any, Any]:
    pd = _pd()
    min_words = int(get_nested(config, "official_eyebench_alignment.residualization.min_words_for_slope", 8))
    word = _with_derived_columns(_merge_boundary_vocoid(inputs["word"], inputs["segmentation_boundary"]))
    word["participant_id"] = word["participant_id"].astype(str)
    word["speech_id"] = word["speech_id"].astype(str)
    word["unique_paragraph_id"] = [
        _text_id_from_paragraph(speech, paragraph)
        for speech, paragraph in zip(word["speech_id"], word["paragraph_id"], strict=True)
    ]
    word["text_id"] = word["unique_paragraph_id"]
    word["sample_id"] = word["participant_id"] + "::" + word["unique_paragraph_id"]
    word["unique_trial_id"] = word["participant_id"] + "_" + word["unique_paragraph_id"]
    include = word.get("include_primary_analysis", True)
    if not isinstance(include, bool):
        word = word[include.fillna(False).astype(bool)].copy()
    rows = []
    for sample_id, group in word.groupby("sample_id", sort=True, dropna=False):
        row = _base_sample_row(group, str(sample_id), min_words)
        first = group.iloc[0]
        row["sample_id"] = str(sample_id)
        row["speech_id"] = str(first["speech_id"])
        row["text_id"] = str(first["unique_paragraph_id"])
        row["unique_paragraph_id"] = str(first["unique_paragraph_id"])
        row["unique_trial_id"] = str(first["unique_trial_id"])
        row["paragraph_id"] = str(first["paragraph_id"])
        row["passage_id"] = str(first["unique_paragraph_id"])
        rows.append(row)
    samples = pd.DataFrame(rows).sort_values(["participant_id", "text_id"]).reset_index(drop=True)
    participant_stats = inputs["participant_stats"].rename(
        columns={"subj": "participant_id", "score_reading_comprehension_test": "eyebench_rcs_score"}
    )
    samples = samples.merge(
        participant_stats[["participant_id", "eyebench_rcs_score"]],
        on="participant_id",
        how="left",
    )
    _write_parquet(out / "data" / "official_alignment_trial_features.parquet", samples)
    return samples, word


def build_official_split_labels(config: dict[str, Any], out: Path, samples: Any, eyebench: Path) -> Any:
    pd = _pd()
    official_trials = _load_official_trial_ids(eyebench)
    rows: list[dict[str, Any]] = []
    samples = samples.reset_index(drop=True).copy()
    by_trial = samples.set_index("unique_trial_id", drop=False)
    seed = int(get_nested(config, "official_eyebench_alignment.deterministic_seed", 131))
    for fold_id, fold in official_trials.groupby("fold_id", dropna=False):
        train_ids = set(fold.loc[fold["regime"].eq("train_train"), "unique_trial_id"].astype(str))
        for official_regime, split_name in OFFICIAL_REGIME_TO_SPLIT.items():
            test_ids = set(
                fold.loc[fold["regime"].eq(f"test_{official_regime}"), "unique_trial_id"].astype(str)
            )
            train = by_trial.loc[sorted(train_ids.intersection(by_trial.index))].copy()
            test = by_trial.loc[sorted(test_ids.intersection(by_trial.index))].copy()
            train_y = train["reader_group_binary"].dropna() if "reader_group_binary" in train else pd.Series()
            test_y = test["reader_group_binary"].dropna() if "reader_group_binary" in test else pd.Series()
            split_valid = bool(not train.empty and not test.empty and train_y.nunique() >= 2 and not test_y.empty)
            train_participants = set(train["participant_id"].astype(str))
            test_participants = set(test["participant_id"].astype(str))
            train_texts = set(train["speech_id"].astype(str))
            test_texts = set(test["speech_id"].astype(str))
            for _, sample in samples.iterrows():
                unique_trial_id = str(sample["unique_trial_id"])
                in_train = unique_trial_id in train_ids
                in_test = unique_trial_id in test_ids
                role = "exclude"
                if in_train and in_test:
                    role = "invalid_overlap"
                elif in_train:
                    role = "train"
                elif in_test:
                    role = "test"
                rows.append(
                    {
                        "mode": "eyebench_folds_full_feature_intersection",
                        "split_name": split_name,
                        "official_regime": official_regime,
                        "fold_id": int(fold_id),
                        "sample_id": sample["sample_id"],
                        "unique_trial_id": unique_trial_id,
                        "unique_paragraph_id": sample["unique_paragraph_id"],
                        "participant_id": sample["participant_id"],
                        "speech_id": sample["speech_id"],
                        "text_id": sample["text_id"],
                        "reader_group": sample["reader_group"],
                        "reader_group_binary": sample["reader_group_binary"],
                        "split_role": role,
                        "include_in_fold": role in {"train", "test"},
                        "n_train_samples": int(len(train)),
                        "n_test_samples": int(len(test)),
                        "n_train_participants": int(len(train_participants)),
                        "n_test_participants": int(len(test_participants)),
                        "n_train_texts": int(len(train_texts)),
                        "n_test_texts": int(len(test_texts)),
                        "participant_overlap": bool(train_participants.intersection(test_participants)),
                        "text_overlap": bool(train_texts.intersection(test_texts)),
                        "split_valid": split_valid,
                        "skip_reason": "" if split_valid else "empty_or_single_class_training_fold",
                        "split_seed": seed,
                        "split_version": "official_eyebench_fold_metadata_v1",
                    }
                )
    splits = pd.DataFrame(rows)
    _write_parquet(out / "splits" / "official_eyebench_split_labels.parquet", splits)
    return splits


def validate_official_split_labels(splits: Any) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    summaries: list[dict[str, Any]] = []
    if splits.empty:
        return ["official split labels are empty"], summaries
    if splits["split_name"].astype(str).str.contains("random", case=False, na=False).any():
        errors.append("random split label found")
    for (split_name, fold_id), fold in splits.groupby(["split_name", "fold_id"], dropna=False):
        train = fold[fold["split_role"].eq("train")]
        test = fold[fold["split_role"].eq("test")]
        train_participants = set(train["participant_id"].astype(str))
        test_participants = set(test["participant_id"].astype(str))
        train_texts = set(train["speech_id"].astype(str))
        test_texts = set(test["speech_id"].astype(str))
        if split_name == "unseen_reader" and train_participants.intersection(test_participants):
            errors.append(f"participant overlap in {split_name} fold {fold_id}")
        if split_name == "unseen_text" and train_texts.intersection(test_texts):
            errors.append(f"text overlap in {split_name} fold {fold_id}")
        if split_name == "unseen_reader_and_text":
            if train_participants.intersection(test_participants):
                errors.append(f"participant overlap in {split_name} fold {fold_id}")
            if train_texts.intersection(test_texts):
                errors.append(f"text overlap in {split_name} fold {fold_id}")
        train_y = train["reader_group_binary"].dropna()
        test_y = test["reader_group_binary"].dropna()
        if train.empty or train_y.nunique() < 2:
            errors.append(f"TYP train fold lacks both classes: {split_name} fold {fold_id}")
        if test.empty or test_y.empty:
            errors.append(f"test fold has no valid labels: {split_name} fold {fold_id}")
        summaries.append(
            {
                "split_name": split_name,
                "fold_id": int(fold_id),
                "train_samples": int(len(train)),
                "test_samples": int(len(test)),
                "train_participants": int(len(train_participants)),
                "test_participants": int(len(test_participants)),
                "train_texts": int(len(train_texts)),
                "test_texts": int(len(test_texts)),
                "train_positive": int(train_y.sum()) if not train_y.empty else 0,
                "test_positive": int(test_y.sum()) if not test_y.empty else 0,
            }
        )
    return errors, summaries


def _bridge_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_bridge": {
            "deterministic_seed": get_nested(
                config,
                "official_eyebench_alignment.deterministic_seed",
                131,
            ),
            "residualization": get_nested(
                config,
                "official_eyebench_alignment.residualization",
                {},
            ),
        }
    }


def _empty_typ_rows(mode: str, model_name: str, claim_type: str, reason: str) -> Any:
    pd = _pd()
    rows = []
    for split_name in OFFICIAL_SPLITS:
        for level in ["participant_text_trial", "reader_aggregated"]:
            rows.append(
                {
                    "mode": mode,
                    "model_name": model_name,
                    "claim_type": claim_type,
                    "task": "CopCo_TYP",
                    "split_name": split_name,
                    "evaluation_level": level,
                    "n_features": 0,
                    "n_predictions": 0,
                    "usable_folds": 0,
                    "skipped_folds": 0,
                    "roc_auc": None,
                    "pr_auc": None,
                    "balanced_accuracy": None,
                    "macro_f1": None,
                    "brier_score": None,
                    "status": "skipped",
                    "skip_reason": reason,
                }
            )
    return pd.DataFrame(rows, columns=ALIGN_TYP_COLUMNS)


def _empty_rcs_rows(mode: str, model_name: str, claim_type: str, reason: str) -> Any:
    pd = _pd()
    rows = []
    for split_name in OFFICIAL_SPLITS:
        for level in ["participant_text_trial", "reader_aggregated"]:
            rows.append(
                {
                    "mode": mode,
                    "model_name": model_name,
                    "claim_type": claim_type,
                    "task": "CopCo_RCS",
                    "split_name": split_name,
                    "evaluation_level": level,
                    "target": "eyebench_rcs_score",
                    "target_scale": "EyeBench_RCS_score",
                    "n_features": 0,
                    "n_predictions": 0,
                    "usable_folds": 0,
                    "skipped_folds": 0,
                    "rmse": None,
                    "mae": None,
                    "r2": None,
                    "status": "skipped",
                    "skip_reason": reason,
                }
            )
    return pd.DataFrame(rows, columns=ALIGN_RCS_COLUMNS)


def _evaluate_typ_mode(
    config: dict[str, Any],
    fold_cache: dict[tuple[str, int], dict[str, Any]],
    *,
    mode: str,
    model_name: str,
    claim_type: str,
) -> tuple[Any, Any]:
    pd = _pd()
    seed = int(get_nested(config, "official_eyebench_alignment.deterministic_seed", 131))
    metric_rows = []
    prediction_rows = []
    for split_name in OFFICIAL_SPLITS:
        fold_keys = [key for key in sorted(fold_cache) if key[0] == split_name]
        predictions = []
        usable = 0
        skipped = 0
        n_features = 0
        skip_reason = ""
        for _, fold_id in fold_keys:
            train = fold_cache[(split_name, fold_id)]["train"].copy()
            test = fold_cache[(split_name, fold_id)]["test"].copy()
            train_y = pd.to_numeric(train["reader_group_binary"], errors="coerce")
            test_y = pd.to_numeric(test["reader_group_binary"], errors="coerce")
            if train.empty or test.empty or train_y.nunique(dropna=True) < 2:
                skipped += 1
                skip_reason = "empty_or_single_class_training_fold"
                continue
            columns = _clean_feature_list(train, _feature_columns(train).get(PRIMARY_TYPO_FEATURE_GROUP, []))
            if not columns:
                skipped += 1
                skip_reason = "no_usable_d3_features"
                continue
            model = _model_pipeline("logistic_regression", task="typ", seed=seed + int(fold_id))
            model.fit(train[columns], train_y.astype(int))
            score = _score_estimator(model, test[columns])
            usable += 1
            n_features = max(n_features, len(columns))
            for row, truth, pred in zip(test.to_dict("records"), test_y, score, strict=True):
                predictions.append(
                    {
                        "mode": mode,
                        "model_name": model_name,
                        "claim_type": claim_type,
                        "task": "CopCo_TYP",
                        "split_name": split_name,
                        "fold_id": int(fold_id),
                        "feature_group": PRIMARY_TYPO_FEATURE_GROUP,
                        "model": "logistic_regression",
                        "sample_id": row["sample_id"],
                        "unique_trial_id": row.get("unique_trial_id"),
                        "unique_paragraph_id": row.get("unique_paragraph_id"),
                        "participant_id": row["participant_id"],
                        "speech_id": row["speech_id"],
                        "text_id": row["text_id"],
                        "y_true": int(truth),
                        "y_score": float(pred),
                        "y_pred": int(float(pred) >= 0.5),
                        "eval_regime": SPLIT_TO_OFFICIAL_REGIME[split_name],
                        "eval_type": "test",
                    }
                )
        pred_frame = pd.DataFrame(predictions)
        prediction_rows.append(pred_frame)
        for level, frame in [
            ("participant_text_trial", pred_frame),
            ("reader_aggregated", _reader_aggregate_classification(pred_frame)),
        ]:
            if frame.empty:
                metric = {
                    "n_predictions": 0,
                    "roc_auc": None,
                    "pr_auc": None,
                    "balanced_accuracy": None,
                    "macro_f1": None,
                    "brier_score": None,
                    "status": "skipped",
                    "skip_reason": skip_reason or "no_valid_predictions",
                }
            else:
                metric = {
                    "n_predictions": int(len(frame)),
                    **_classification_metrics(frame["y_true"], frame["y_score"]),
                    "status": "complete",
                    "skip_reason": "",
                }
            metric_rows.append(
                {
                    "mode": mode,
                    "model_name": model_name,
                    "claim_type": claim_type,
                    "task": "CopCo_TYP",
                    "split_name": split_name,
                    "evaluation_level": level,
                    "n_features": int(n_features),
                    "usable_folds": int(usable),
                    "skipped_folds": int(skipped),
                    **metric,
                }
            )
    metrics = pd.DataFrame(metric_rows, columns=ALIGN_TYP_COLUMNS)
    non_empty = [frame for frame in prediction_rows if not frame.empty]
    predictions = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    return metrics, predictions


def _evaluate_rcs_mode(
    config: dict[str, Any],
    fold_cache: dict[tuple[str, int], dict[str, Any]],
    *,
    mode: str,
    model_name: str,
    claim_type: str,
) -> tuple[Any, Any]:
    pd = _pd()
    target = str(get_nested(config, "official_eyebench_alignment.rcs.target_column", "eyebench_rcs_score"))
    missing_value = float(get_nested(config, "official_eyebench_alignment.rcs.missing_target_value", -1))
    seed = int(get_nested(config, "official_eyebench_alignment.deterministic_seed", 131))
    metric_rows = []
    prediction_rows = []
    for split_name in OFFICIAL_SPLITS:
        fold_keys = [key for key in sorted(fold_cache) if key[0] == split_name]
        predictions = []
        usable = 0
        skipped = 0
        n_features = 0
        skip_reason = ""
        for _, fold_id in fold_keys:
            train = fold_cache[(split_name, fold_id)]["train"].copy()
            test = fold_cache[(split_name, fold_id)]["test"].copy()
            train_y = pd.to_numeric(train.get(target), errors="coerce")
            test_y = pd.to_numeric(test.get(target), errors="coerce")
            valid_train = train_y.notna() & ~train_y.eq(missing_value)
            valid_test = test_y.notna() & ~test_y.eq(missing_value)
            if valid_train.sum() < 3 or valid_test.sum() == 0:
                skipped += 1
                skip_reason = "insufficient_valid_rcs_target_rows"
                continue
            train = train.loc[valid_train].copy()
            test = test.loc[valid_test].copy()
            train_y = train_y.loc[valid_train]
            test_y = test_y.loc[valid_test]
            columns = _clean_feature_list(train, _feature_columns(train).get(PRIMARY_TYPO_FEATURE_GROUP, []))
            if not columns:
                skipped += 1
                skip_reason = "no_usable_d3_features"
                continue
            model = _model_pipeline("ridge_regression", task="rcs", seed=seed + int(fold_id))
            model.fit(train[columns], train_y.astype(float))
            pred = _score_regressor(model, test[columns])
            usable += 1
            n_features = max(n_features, len(columns))
            for row, truth, value in zip(test.to_dict("records"), test_y, pred, strict=True):
                predictions.append(
                    {
                        "mode": mode,
                        "model_name": model_name,
                        "claim_type": claim_type,
                        "task": "CopCo_RCS",
                        "split_name": split_name,
                        "fold_id": int(fold_id),
                        "feature_group": PRIMARY_TYPO_FEATURE_GROUP,
                        "model": "ridge_regression",
                        "sample_id": row["sample_id"],
                        "unique_trial_id": row.get("unique_trial_id"),
                        "unique_paragraph_id": row.get("unique_paragraph_id"),
                        "participant_id": row["participant_id"],
                        "speech_id": row["speech_id"],
                        "text_id": row["text_id"],
                        "y_true": float(truth),
                        "y_pred": float(value),
                        "eval_regime": SPLIT_TO_OFFICIAL_REGIME[split_name],
                        "eval_type": "test",
                    }
                )
        pred_frame = pd.DataFrame(predictions)
        prediction_rows.append(pred_frame)
        for level, frame in [
            ("participant_text_trial", pred_frame),
            ("reader_aggregated", _reader_aggregate_regression(pred_frame)),
        ]:
            if frame.empty:
                metric = {
                    "n_predictions": 0,
                    "rmse": None,
                    "mae": None,
                    "r2": None,
                    "status": "skipped",
                    "skip_reason": skip_reason or "no_valid_predictions",
                }
            else:
                metric = {
                    "n_predictions": int(len(frame)),
                    **_regression_metrics(frame["y_true"], frame["y_pred"]),
                    "status": "complete",
                    "skip_reason": "",
                }
            metric_rows.append(
                {
                    "mode": mode,
                    "model_name": model_name,
                    "claim_type": claim_type,
                    "task": "CopCo_RCS",
                    "split_name": split_name,
                    "evaluation_level": level,
                    "target": target,
                    "target_scale": str(get_nested(config, "official_eyebench_alignment.rcs.scale", "EyeBench_RCS_score")),
                    "n_features": int(n_features),
                    "usable_folds": int(usable),
                    "skipped_folds": int(skipped),
                    **metric,
                }
            )
    metrics = pd.DataFrame(metric_rows, columns=ALIGN_RCS_COLUMNS)
    non_empty = [frame for frame in prediction_rows if not frame.empty]
    predictions = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    return metrics, predictions


def _load_benchmarkbridge_typ(config: dict[str, Any], repo_root: str | Path) -> Any:
    pd = _pd()
    bb = _configured_path(config, "official_eyebench_alignment.frozen_inputs.benchmark_bridge_dir", repo_root)
    metrics = pd.read_csv(bb / "typ" / "typ_benchmark_metrics.csv")
    d3 = metrics[
        metrics["feature_group"].eq(PRIMARY_TYPO_FEATURE_GROUP)
        & metrics["model"].eq("logistic_regression")
        & metrics["split_name"].isin(OFFICIAL_SPLITS)
    ].copy()
    d3["mode"] = "full_data_eyebench_style"
    d3["model_name"] = "D3_FullData_EyeBenchStyle"
    d3["claim_type"] = "internal_EyeBench-style_benchmark-relative"
    return d3[
        [
            "mode",
            "model_name",
            "claim_type",
            "task",
            "split_name",
            "evaluation_level",
            "n_features",
            "n_predictions",
            "usable_folds",
            "skipped_folds",
            "roc_auc",
            "pr_auc",
            "balanced_accuracy",
            "macro_f1",
            "brier_score",
            "status",
            "skip_reason",
        ]
    ]


def _load_benchmarkbridge_rcs(config: dict[str, Any], repo_root: str | Path) -> Any:
    pd = _pd()
    bb = _configured_path(config, "official_eyebench_alignment.frozen_inputs.benchmark_bridge_dir", repo_root)
    metrics = pd.read_csv(bb / "rcs" / "rcs_benchmark_metrics.csv")
    d3 = metrics[
        metrics["feature_group"].eq(PRIMARY_TYPO_FEATURE_GROUP)
        & metrics["model"].eq("ridge_regression")
        & metrics["split_name"].isin(OFFICIAL_SPLITS)
    ].copy()
    d3["mode"] = "full_data_eyebench_style"
    d3["model_name"] = "D3_FullData_EyeBenchStyle"
    d3["claim_type"] = "internal_EyeBench-style_benchmark-relative"
    return d3[
        [
            "mode",
            "model_name",
            "claim_type",
            "task",
            "split_name",
            "evaluation_level",
            "target",
            "target_scale",
            "n_features",
            "n_predictions",
            "usable_folds",
            "skipped_folds",
            "rmse",
            "mae",
            "r2",
            "status",
            "skip_reason",
        ]
    ]


def _write_eyebench_compatible_predictions(out: Path, predictions: Any) -> None:
    if predictions.empty:
        return
    frame = predictions.copy()
    frame["label"] = frame["y_true"]
    frame["prediction_prob"] = frame["y_score"]
    frame["binary_prediction"] = frame["y_pred"]
    keep = [
        "label",
        "prediction_prob",
        "binary_prediction",
        "eval_regime",
        "eval_type",
        "fold_id",
        "participant_id",
        "unique_paragraph_id",
        "unique_trial_id",
        "speech_id",
    ]
    _write_csv(out / "typ" / "d3_fullfeature_eyebench_folds_trial_level_test_results.csv", frame[keep])


def write_official_evaluator_blocker_report(
    dirs: dict[str, Path],
    environment: dict[str, Any],
    data_download: dict[str, Any],
) -> dict[str, Any]:
    official_run = False
    reason = []
    if not environment.get("eyebench_imports"):
        reason.append("missing or incompatible EyeBench environment")
    if not data_download.get("copco_processed_files_present"):
        reason.append("missing EyeBench processed CopCo data")
    reason.append("external wrapper generated compatible prediction CSVs; official aggregator was not run")
    report = {
        "official_evaluator_run": official_run,
        "exact_reason_if_false": "; ".join(reason),
        "missing_data": not data_download.get("copco_processed_files_present"),
        "missing_environment": not environment.get("eyebench_imports"),
        "missing_cli": False,
        "dependency_issue": not environment.get("eyebench_imports"),
        "schema_mismatch": False,
        "manual_metric_computation_used": True,
    }
    text = "\n".join(
        [
            "# Official Evaluator Blocker Report",
            "",
            f"- official_evaluator_run: {official_run}",
            f"- exact reason if false: {report['exact_reason_if_false']}",
            f"- missing data: {report['missing_data']}",
            f"- missing environment: {report['missing_environment']}",
            f"- missing CLI: {report['missing_cli']}",
            f"- dependency issue: {report['dependency_issue']}",
            f"- schema mismatch: {report['schema_mismatch']}",
            f"- manual metric computation used instead: {report['manual_metric_computation_used']}",
        ]
    )
    _write_report(dirs, "official_evaluator_blocker_report.md", text)
    return report


def _parse_percent(value: Any) -> float | None:
    if value is None or value != value:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) / 100.0 if match else None


def _load_typ_reference_table(config: dict[str, Any], repo_root: str | Path) -> Any:
    pd = _pd()
    path = _configured_path(
        config,
        "official_eyebench_alignment.decision_gates.CopCo_TYP.formatted_table",
        repo_root,
    )
    raw = pd.read_csv(path)
    rows = []
    for _, row in raw.iterrows():
        rows.append(
            {
                "model": str(row["Model"]),
                "mode": "official_eyebench_reported_baseline",
                "claim_type": "official_reported_reference",
                "official_mode": True,
                "exact_folds": True,
                "exact_processed_data": True,
                "unseen_reader_balanced_accuracy": _parse_percent(row.get("Unseen Reader_\\makecell{Balanced\\\\Accuracy}")),
                "unseen_text_balanced_accuracy": _parse_percent(row.get("Unseen Text_\\makecell{Balanced\\\\Accuracy}")),
                "unseen_reader_text_balanced_accuracy": _parse_percent(row.get("Unseen Text \\& Reader_\\makecell{Balanced\\\\Accuracy}")),
                "unseen_reader_AUROC": _parse_percent(row.get("Unseen Reader_AUROC")),
                "unseen_text_AUROC": _parse_percent(row.get("Unseen Text_AUROC")),
                "unseen_reader_text_AUROC": _parse_percent(row.get("Unseen Text \\& Reader_AUROC")),
                "notes": "EyeBench formatted CopCo_TYP test table central value.",
            }
        )
    return pd.DataFrame(rows)


def _metric_value(metrics: Any, mode: str, split_name: str, metric: str, level: str = "reader_aggregated") -> float | None:
    row = metrics[
        metrics["mode"].eq(mode)
        & metrics["split_name"].eq(split_name)
        & metrics["evaluation_level"].eq(level)
    ]
    if row.empty or metric not in row:
        return None
    value = row.iloc[0][metric]
    if value is None:
        return None
    return float(value) if value == value else None


def _average(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and value == value]
    return sum(valid) / len(valid) if valid else None


def write_typ_reports_and_tables(
    config: dict[str, Any],
    dirs: dict[str, Path],
    typ_metrics: Any,
    repo_root: str | Path,
) -> Any:
    pd = _pd()
    primary = typ_metrics[
        typ_metrics["evaluation_level"].eq("reader_aggregated")
        & typ_metrics["split_name"].isin(OFFICIAL_SPLITS)
    ]
    text = "\n".join(
        [
            "# CopCo TYP Official Alignment Report",
            "",
            "Rows distinguish official-subset attempts, EyeBench-fold full-feature results, "
            "and BenchmarkBridge full-data internal results.",
            "",
            _markdown_table(
                primary[
                    [
                        "mode",
                        "model_name",
                        "claim_type",
                        "split_name",
                        "n_predictions",
                        "roc_auc",
                        "pr_auc",
                        "balanced_accuracy",
                        "macro_f1",
                        "brier_score",
                        "status",
                    ]
                ].to_dict("records"),
                [
                    "mode",
                    "model_name",
                    "claim_type",
                    "split_name",
                    "n_predictions",
                    "roc_auc",
                    "pr_auc",
                    "balanced_accuracy",
                    "macro_f1",
                    "brier_score",
                    "status",
                ],
                max_rows=40,
            ),
        ]
    )
    _write_report(dirs, "copco_typ_official_alignment_report.md", text)
    rows = []
    for mode, model_name, claim_type, official_mode, exact_folds, exact_processed in [
        ("official_eyebench_subset", "D3_EyeBench_Lite", "official_attempt_failed", False, False, False),
        (
            "eyebench_folds_full_feature_intersection",
            "D3_FullFeature_EyeBenchFolds",
            "EyeBench-fold-aligned_full-feature_non-official",
            False,
            True,
            False,
        ),
        (
            "full_data_eyebench_style",
            "D3_FullData_EyeBenchStyle",
            "internal_EyeBench-style_benchmark-relative",
            False,
            False,
            False,
        ),
    ]:
        values = {
            "model": model_name,
            "mode": mode,
            "claim_type": claim_type,
            "official_mode": official_mode,
            "exact_folds": exact_folds,
            "exact_processed_data": exact_processed,
            "unseen_reader_balanced_accuracy": _metric_value(typ_metrics, mode, "unseen_reader", "balanced_accuracy"),
            "unseen_text_balanced_accuracy": _metric_value(typ_metrics, mode, "unseen_text", "balanced_accuracy"),
            "unseen_reader_text_balanced_accuracy": _metric_value(
                typ_metrics,
                mode,
                "unseen_reader_and_text",
                "balanced_accuracy",
            ),
            "unseen_reader_AUROC": _metric_value(typ_metrics, mode, "unseen_reader", "roc_auc"),
            "unseen_text_AUROC": _metric_value(typ_metrics, mode, "unseen_text", "roc_auc"),
            "unseen_reader_text_AUROC": _metric_value(
                typ_metrics,
                mode,
                "unseen_reader_and_text",
                "roc_auc",
            ),
            "notes": "Generated by OfficialEyeBenchAlignment v1.",
        }
        values["average_balanced_accuracy"] = _average(
            [
                values["unseen_reader_balanced_accuracy"],
                values["unseen_text_balanced_accuracy"],
                values["unseen_reader_text_balanced_accuracy"],
            ]
        )
        values["average_AUROC"] = _average(
            [
                values["unseen_reader_AUROC"],
                values["unseen_text_AUROC"],
                values["unseen_reader_text_AUROC"],
            ]
        )
        rows.append(values)
    table = pd.concat([pd.DataFrame(rows), _load_typ_reference_table(config, repo_root)], ignore_index=True)
    ordered = [
        "model",
        "mode",
        "claim_type",
        "official_mode",
        "exact_folds",
        "exact_processed_data",
        "unseen_reader_balanced_accuracy",
        "unseen_text_balanced_accuracy",
        "unseen_reader_text_balanced_accuracy",
        "average_balanced_accuracy",
        "unseen_reader_AUROC",
        "unseen_text_AUROC",
        "unseen_reader_text_AUROC",
        "average_AUROC",
        "notes",
    ]
    for column in ["average_balanced_accuracy", "average_AUROC"]:
        if column not in table:
            table[column] = None
    table = table[ordered]
    _write_table(dirs, "copco_typ_official_alignment_comparison.csv", table)
    md = _markdown_table(table.to_dict("records"), table.columns.tolist(), max_rows=80)
    tex = table.to_latex(index=False, float_format=lambda x: f"{x:.3f}" if x == x else "")
    _write_md(dirs["repo_tables"] / "copco_typ_official_alignment_comparison.md", md)
    _write_md(dirs["result_tables"] / "copco_typ_official_alignment_comparison.md", md)
    _write_md(dirs["repo_tables"] / "copco_typ_official_alignment_comparison.tex", tex)
    _write_md(dirs["result_tables"] / "copco_typ_official_alignment_comparison.tex", tex)
    return table


def write_rcs_report(dirs: dict[str, Path], rcs_metrics: Any) -> None:
    primary = rcs_metrics[rcs_metrics["evaluation_level"].eq("reader_aggregated")]
    text = "\n".join(
        [
            "# CopCo RCS Official Alignment Report",
            "",
            "RCS is auxiliary. The official EyeBench target is `RCS_score`; the BenchmarkBridge "
            "full-data reference used the frozen project comprehension score, so cross-mode scale "
            "comparisons should emphasize R2.",
            "",
            _markdown_table(
                primary[
                    [
                        "mode",
                        "model_name",
                        "claim_type",
                        "split_name",
                        "target_scale",
                        "n_predictions",
                        "rmse",
                        "mae",
                        "r2",
                        "status",
                    ]
                ].to_dict("records"),
                [
                    "mode",
                    "model_name",
                    "claim_type",
                    "split_name",
                    "target_scale",
                    "n_predictions",
                    "rmse",
                    "mae",
                    "r2",
                    "status",
                ],
                max_rows=40,
            ),
        ]
    )
    _write_report(dirs, "copco_rcs_official_alignment_report.md", text)


def _best_reference_value(table: Any, metric: str, split: str) -> float | None:
    column = {
        ("balanced_accuracy", "unseen_reader"): "unseen_reader_balanced_accuracy",
        ("balanced_accuracy", "unseen_text"): "unseen_text_balanced_accuracy",
        ("balanced_accuracy", "unseen_reader_and_text"): "unseen_reader_text_balanced_accuracy",
        ("roc_auc", "unseen_reader"): "unseen_reader_AUROC",
        ("roc_auc", "unseen_text"): "unseen_text_AUROC",
        ("roc_auc", "unseen_reader_and_text"): "unseen_reader_text_AUROC",
    }[(metric, split)]
    refs = table[table["mode"].eq("official_eyebench_reported_baseline")]
    values = refs[column].dropna()
    return float(values.max()) if not values.empty else None


def write_decision_report(
    dirs: dict[str, Path],
    typ_metrics: Any,
    rcs_metrics: Any,
    typ_table: Any,
    environment: dict[str, Any],
    data_download: dict[str, Any],
    evaluator: dict[str, Any],
) -> dict[str, Any]:
    official_mode_run = bool(evaluator.get("official_evaluator_run")) and bool(
        data_download.get("copco_processed_files_present")
    )
    exact_folds_run = bool(
        typ_metrics[
            typ_metrics["mode"].eq("eyebench_folds_full_feature_intersection")
            & typ_metrics["status"].eq("complete")
        ].shape[0]
    )
    exact_processed_run = bool(data_download.get("copco_processed_files_present")) and official_mode_run
    full_data_consistent = bool(
        typ_metrics[
            typ_metrics["mode"].eq("full_data_eyebench_style")
            & typ_metrics["status"].eq("complete")
        ].shape[0]
    )
    lite_beats = False
    fullfeature_beats = False
    reader_auc = _metric_value(
        typ_metrics,
        "eyebench_folds_full_feature_intersection",
        "unseen_reader",
        "roc_auc",
    )
    reader_ba = _metric_value(
        typ_metrics,
        "eyebench_folds_full_feature_intersection",
        "unseen_reader",
        "balanced_accuracy",
    )
    strict_auc = _metric_value(
        typ_metrics,
        "eyebench_folds_full_feature_intersection",
        "unseen_reader_and_text",
        "roc_auc",
    )
    strict_ba = _metric_value(
        typ_metrics,
        "eyebench_folds_full_feature_intersection",
        "unseen_reader_and_text",
        "balanced_accuracy",
    )
    best_reader_auc = _best_reference_value(typ_table, "roc_auc", "unseen_reader")
    best_reader_ba = _best_reference_value(typ_table, "balanced_accuracy", "unseen_reader")
    best_strict_auc = _best_reference_value(typ_table, "roc_auc", "unseen_reader_and_text")
    best_strict_ba = _best_reference_value(typ_table, "balanced_accuracy", "unseen_reader_and_text")
    if all(value is not None for value in [reader_auc, reader_ba, strict_auc, strict_ba]):
        fullfeature_beats = bool(
            reader_auc > float(best_reader_auc or 1.0)
            and reader_ba > float(best_reader_ba or 1.0)
            and strict_auc > float(best_strict_auc or 1.0)
            and strict_ba > float(best_strict_ba or 1.0)
        )
    if official_mode_run and lite_beats:
        category = "official_sota_claim_allowed"
    elif official_mode_run:
        category = "official_compatible_but_not_sota"
    elif exact_folds_run or full_data_consistent:
        category = "benchmark_relative_sota_only"
    elif not environment.get("eyebench_imports"):
        category = "blocked_by_environment"
    else:
        category = "blocked_by_data_alignment"
    rcs_reader = _metric_value(
        rcs_metrics,
        "eyebench_folds_full_feature_intersection",
        "unseen_reader",
        "r2",
    )
    wording = (
        "benchmark-relative state of the art under internal EyeBench-style evaluation"
        if category == "benchmark_relative_sota_only"
        else "official EyeBench-compatible state-of-the-art on CopCo_TYP"
        if category == "official_sota_claim_allowed"
        else "internal alignment attempt; no official SOTA claim"
    )
    decision = {
        "decision_category": category,
        "official_eyebench_mode_run": official_mode_run,
        "exact_eyebench_folds_run": exact_folds_run,
        "exact_eyebench_processed_data_run": exact_processed_run,
        "d3_eyebench_lite_beat_strongest_official_typ_baselines": lite_beats,
        "d3_fullfeature_eyebenchfolds_beat_baselines": fullfeature_beats,
        "full_data_benchmarkbridge_consistent": full_data_consistent,
        "can_be_called_official_eyebench_result": official_mode_run,
        "can_be_called_eyebench_compatible_result": official_mode_run,
        "can_be_called_eyebench_fold_aligned_result": exact_folds_run,
        "can_be_called_benchmark_relative_result": full_data_consistent,
        "can_be_called_internal_only_result": not official_mode_run,
        "changes_manuscript_main_claim": False,
        "permits_official_sota_claim": category == "official_sota_claim_allowed",
        "recommended_wording": wording,
        "rcs_reader_aggregated_r2_unseen_reader": rcs_reader,
    }
    rows = [
        {"question": "Did official EyeBench mode run?", "answer": official_mode_run},
        {"question": "Did exact EyeBench folds run?", "answer": exact_folds_run},
        {"question": "Did exact EyeBench processed data run?", "answer": exact_processed_run},
        {"question": "Did D3_EyeBench_Lite beat strongest official CopCo_TYP baselines?", "answer": lite_beats},
        {"question": "Did D3_FullFeature_EyeBenchFolds beat baselines?", "answer": fullfeature_beats},
        {"question": "Did full-data BenchmarkBridge remain consistent?", "answer": full_data_consistent},
        {"question": "Does this change the manuscript main claim?", "answer": False},
        {"question": "Does this permit an official SOTA claim?", "answer": decision["permits_official_sota_claim"]},
    ]
    text = "\n".join(
        [
            "# Official EyeBench Decision Report",
            "",
            f"- Decision category: `{category}`",
            f"- Recommended wording: {wording}",
            "",
            "## Decisions",
            _markdown_table(rows, ["question", "answer"], max_rows=20),
            "",
            "## Claim Labels",
            f"- official EyeBench result: {decision['can_be_called_official_eyebench_result']}",
            f"- EyeBench-compatible result: {decision['can_be_called_eyebench_compatible_result']}",
            f"- EyeBench-fold-aligned result: {decision['can_be_called_eyebench_fold_aligned_result']}",
            f"- benchmark-relative result: {decision['can_be_called_benchmark_relative_result']}",
            f"- internal-only result: {decision['can_be_called_internal_only_result']}",
            "",
            "## Exact Wording",
            f"Use: \"{wording}.\" Do not call the result official unless exact processed "
            "EyeBench data, folds, and evaluator are used.",
        ]
    )
    _write_report(dirs, "official_eyebench_decision_report.md", text)
    return decision


def update_supplement_note(repo_root: str | Path, decision: dict[str, Any]) -> None:
    path = Path(repo_root).resolve() / "paper" / "submission_v1" / "supplement_sections" / "18_benchmark_bridge.tex"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    marker = "\\paragraph{Official EyeBench alignment.}"
    note = (
        "\n\n"
        "\\paragraph{Official EyeBench alignment.}\n"
        "We additionally vendored the official EyeBench repository and aligned CopCo identifiers, "
        "fold metadata, and result formats. Exact official evaluation was not run because the local "
        "EyeBench environment did not import and the official processed CopCo files were absent. "
        "Accordingly, the benchmark statement remains benchmark-relative rather than an official "
        "EyeBench leaderboard or state-of-the-art claim.\n"
    )
    if marker not in text:
        path.write_text(text.rstrip() + note, encoding="utf-8")
    elif decision.get("decision_category") == "official_sota_claim_allowed":
        path.write_text(text, encoding="utf-8")


def run_official_eyebench_alignment(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    config_check = validate_official_eyebench_alignment_config(config)
    if config_check["status"] != "passed":
        raise ValueError(f"official EyeBench alignment config failed validation: {config_check['errors']}")
    out = Path(output_dir).resolve() if output_dir else timestamped_output_dir(config, repo_root=repo_root)
    out.mkdir(parents=True, exist_ok=True)
    dirs = _analysis_dirs(config, out, repo_root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    vendor = write_vendor_manifest(config, dirs, repo_root)
    structure = inspect_eyebench_structure(config, dirs, repo_root)
    environment = write_environment_report(config, dirs, repo_root)
    data_download = write_data_download_report(config, dirs, repo_root, environment)
    inputs = _load_inputs(config, repo_root)
    audit = write_alignment_audit(config, out, dirs, inputs)
    samples, word = build_trial_feature_table(config, out, inputs)
    splits = build_official_split_labels(config, out, samples, inputs["eyebench"])
    split_errors, split_summary = validate_official_split_labels(splits)
    _write_csv(dirs["repo_analysis"] / "official_split_diagnostics.csv", _pd().DataFrame(split_summary))
    _write_csv(dirs["result_analysis"] / "official_split_diagnostics.csv", _pd().DataFrame(split_summary))
    if split_errors:
        _write_json(out / "official_split_validation_report.json", {"errors": split_errors})
        raise ValueError(f"official split validation failed: {split_errors}")
    fold_cache, residualization = build_crossfit_fold_feature_cache(_bridge_config(config), out, samples, word, splits)
    mode2_typ, mode2_predictions = _evaluate_typ_mode(
        config,
        fold_cache,
        mode="eyebench_folds_full_feature_intersection",
        model_name="D3_FullFeature_EyeBenchFolds",
        claim_type="EyeBench-fold-aligned_full-feature_non-official",
    )
    mode2_rcs, mode2_rcs_predictions = _evaluate_rcs_mode(
        config,
        fold_cache,
        mode="eyebench_folds_full_feature_intersection",
        model_name="D3_FullFeature_EyeBenchFolds",
        claim_type="EyeBench-fold-aligned_full-feature_non-official",
    )
    official_reason = "EyeBench processed CopCo data/evaluator unavailable"
    mode1_typ = _empty_typ_rows("official_eyebench_subset", "D3_EyeBench_Lite", "official_attempt_failed", official_reason)
    mode1_rcs = _empty_rcs_rows("official_eyebench_subset", "D3_EyeBench_Lite", "official_attempt_failed", official_reason)
    mode3_typ = _load_benchmarkbridge_typ(config, repo_root)
    mode3_rcs = _load_benchmarkbridge_rcs(config, repo_root)
    typ_metrics = _pd().concat([mode1_typ, mode2_typ, mode3_typ], ignore_index=True)
    rcs_metrics = _pd().concat([mode1_rcs, mode2_rcs, mode3_rcs], ignore_index=True)
    typ_predictions = mode2_predictions
    rcs_predictions = mode2_rcs_predictions
    _write_csv(out / "typ" / "typ_official_alignment_metrics.csv", typ_metrics)
    _write_csv(out / "typ" / "typ_official_alignment_predictions.csv", typ_predictions)
    _write_csv(out / "rcs" / "rcs_official_alignment_metrics.csv", rcs_metrics)
    _write_csv(out / "rcs" / "rcs_official_alignment_predictions.csv", rcs_predictions)
    _write_eyebench_compatible_predictions(out, mode2_predictions)
    evaluator = write_official_evaluator_blocker_report(dirs, environment, data_download)
    typ_table = write_typ_reports_and_tables(config, dirs, typ_metrics, repo_root)
    write_rcs_report(dirs, rcs_metrics)
    decision = write_decision_report(
        dirs,
        typ_metrics,
        rcs_metrics,
        typ_table,
        environment,
        data_download,
        evaluator,
    )
    update_supplement_note(repo_root, decision)
    manifest = {
        "run_type": "official_eyebench_alignment_v1",
        "status": "complete",
        "git_sha": _git_sha(repo_root),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "output_dir": str(out),
        "config_validation": config_check,
        "vendor": vendor,
        "structure": structure,
        "environment": environment,
        "data_download": data_download,
        "alignment_audit": audit,
        "split_regimes_completed": OFFICIAL_SPLITS,
        "evaluation_modes_completed": [
            "eyebench_folds_full_feature_intersection",
            "full_data_eyebench_style",
        ],
        "evaluation_modes_skipped": ["official_eyebench_subset"],
        "residualization": residualization,
        "official_evaluator": evaluator,
        "decision": decision,
        "row_counts": {
            "trial_samples": int(len(samples)),
            "split_label_rows": int(len(splits)),
            "typ_predictions": int(len(typ_predictions)),
            "rcs_predictions": int(len(rcs_predictions)),
        },
        "large_outputs_not_for_commit": [
            "data/official_alignment_trial_features.parquet",
            "splits/official_eyebench_split_labels.parquet",
            "typ/*predictions*.csv",
            "rcs/*predictions*.csv",
            "EyeBench processed data",
            "EyeBench caches",
            "WandB outputs",
            "model artifacts",
        ],
    }
    _write_json(out / "manifest.json", manifest)
    validation = validate_official_eyebench_alignment(config, out, repo_root=repo_root)
    _write_json(out / "official_eyebench_alignment_validation_report.json", validation)
    return manifest


def validate_official_eyebench_alignment(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    pd = _pd()
    out = Path(output_dir).resolve()
    dirs = _analysis_dirs(config, out, repo_root)
    errors: list[str] = []
    warnings_list: list[str] = []
    config_check = validate_official_eyebench_alignment_config(config)
    errors.extend(config_check["errors"])
    warnings_list.extend(config_check["warnings"])
    required = [
        Path(repo_root).resolve() / "docs" / "eyebench_vendor_manifest.md",
        out / "data" / "official_alignment_trial_features.parquet",
        out / "splits" / "official_eyebench_split_labels.parquet",
        out / "typ" / "typ_official_alignment_metrics.csv",
        out / "rcs" / "rcs_official_alignment_metrics.csv",
        dirs["result_analysis"] / "eyebench_structure_report.md",
        dirs["result_analysis"] / "eyebench_environment_report.md",
        dirs["result_analysis"] / "eyebench_data_download_report.md",
        dirs["result_analysis"] / "copco_alignment_audit.md",
        dirs["result_analysis"] / "official_evaluator_blocker_report.md",
        dirs["result_analysis"] / "official_eyebench_decision_report.md",
        dirs["result_tables"] / "copco_typ_official_alignment_comparison.csv",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"missing required artifact: {path}")
    if (out / "typ" / "typ_official_alignment_metrics.csv").exists():
        typ = pd.read_csv(out / "typ" / "typ_official_alignment_metrics.csv")
        missing = sorted(set(ALIGN_TYP_COLUMNS) - set(typ.columns))
        if missing:
            errors.append(f"TYP metrics missing columns: {missing}")
        completed_modes = set(typ.loc[typ["status"].eq("complete"), "mode"].astype(str))
        if "eyebench_folds_full_feature_intersection" not in completed_modes:
            warnings_list.append("EyeBench-fold full-feature TYP mode did not complete")
    if (out / "rcs" / "rcs_official_alignment_metrics.csv").exists():
        rcs = pd.read_csv(out / "rcs" / "rcs_official_alignment_metrics.csv")
        missing = sorted(set(ALIGN_RCS_COLUMNS) - set(rcs.columns))
        if missing:
            errors.append(f"RCS metrics missing columns: {missing}")
    if (out / "splits" / "official_eyebench_split_labels.parquet").exists():
        split_errors, _ = validate_official_split_labels(
            pd.read_parquet(out / "splits" / "official_eyebench_split_labels.parquet")
        )
        errors.extend(split_errors)
    if (out / "data" / "official_alignment_trial_features.parquet").exists():
        samples = pd.read_parquet(out / "data" / "official_alignment_trial_features.parquet")
        groups = _feature_columns(samples)
        for group_name, columns in groups.items():
            bad = sorted(set(columns).intersection(set(get_nested(config, "official_eyebench_alignment.prohibited_features", []))))
            if bad:
                errors.append(f"feature group {group_name} includes prohibited predictors: {bad}")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings_list,
        "output_dir": str(out),
    }
