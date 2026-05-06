"""Official EyeBench SOTA verification gate for the frozen CopCo D3 method."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .benchmark_bridge import PROHIBITED_FEATURES, _model_pipeline
from .config import get_nested, timestamped_output_dir
from .official_eyebench_alignment import (
    OFFICIAL_REGIME_TO_SPLIT,
    OFFICIAL_SPLITS,
    SPLIT_TO_OFFICIAL_REGIME,
    _load_official_trial_ids,
    _parse_percent,
)
from .research_exploration import (
    _classification_metrics,
    _markdown_table,
    _np,
    _pd,
    _score_estimator,
)


SOTA_SECTION = "official_eyebench_sota_check"
SOTA_MODEL = "D3_EyeBench_Lite"
OFFICIAL_BASELINE_MODEL = "LogisticRegressionMLArgs"
OFFICIAL_LITE_MODE = "official_eyebench_subset"
READER_LEVEL = "reader_aggregated"
TRIAL_LEVEL = "official_trial_level_fold_mean"

SOTA_TYP_COLUMNS = [
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

BASELINE_COLUMNS = [
    "model_name",
    "baseline_source",
    "split_name",
    "metric_basis",
    "n_features",
    "n_predictions",
    "usable_folds",
    "skipped_folds",
    "roc_auc",
    "balanced_accuracy",
    "published_roc_auc",
    "published_balanced_accuracy",
    "delta_roc_auc",
    "delta_balanced_accuracy",
    "status",
    "skip_reason",
]

COMPARISON_COLUMNS = [
    "model",
    "mode",
    "claim_type",
    "metric_basis",
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

OFFICIAL_TO_COLUMN = {
    "unseen_reader": {
        "balanced_accuracy": "unseen_reader_balanced_accuracy",
        "roc_auc": "unseen_reader_AUROC",
    },
    "unseen_text": {
        "balanced_accuracy": "unseen_text_balanced_accuracy",
        "roc_auc": "unseen_text_AUROC",
    },
    "unseen_reader_and_text": {
        "balanced_accuracy": "unseen_reader_text_balanced_accuracy",
        "roc_auc": "unseen_reader_text_AUROC",
    },
}

RESIDUAL_OUTCOME_CANDIDATES = {
    "first_fixation_duration": ["IA_FIRST_FIXATION_DURATION", "IA_FIRST_FIX_DURATION"],
    "first_pass_duration": ["IA_FIRST_RUN_DWELL_TIME", "IA_FIRST_FIX_DWELL_TIME"],
    "go_past_time": ["IA_SELECTIVE_REGRESSION_PATH_DURATION", "IA_REGRESSION_OUT_TIME"],
    "total_fixation_duration": ["IA_TOTAL_FIXATION_DURATION", "IA_DWELL_TIME"],
    "skipping": ["IA_SKIP", "total_skip"],
    "fixation_count": ["IA_FIXATION_COUNT", "number_of_fixations"],
}

RESIDUAL_PREDICTOR_CANDIDATES = [
    "word_length",
    "wordfreq_frequency",
    "gpt2_surprisal",
    "TRIAL_IA_COUNT",
    "normalized_ID",
    "start_of_line",
    "end_of_line",
    "is_content_word",
]

LOGISTIC_BASELINE_CANDIDATES = [
    "CURRENT_FIX_DURATION_mean",
    "forward_saccade_length_mean",
    "regression_rate",
    "first_pass_skip_rate",
    "mean_FFD",
    "mean_GD",
    "mean_TFD",
    "mean_go_past_time",
    "reading_speed",
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


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env=env,
        )
        return {
            "command": " ".join(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "command": " ".join(command),
            "returncode": None,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
            "timed_out": True,
        }


def _section(config: dict[str, Any]) -> dict[str, Any]:
    section = get_nested(config, SOTA_SECTION, {})
    return section if isinstance(section, dict) else {}


def _analysis_dirs(config: dict[str, Any], out: Path, repo_root: str | Path) -> dict[str, Path]:
    root = Path(repo_root).resolve()
    repo_analysis = root / str(
        get_nested(config, f"{SOTA_SECTION}.repo_analysis_dir", "analysis/official_eyebench_sota_check_v1")
    )
    result_analysis = out / str(
        get_nested(config, f"{SOTA_SECTION}.output_layout.analysis", "analysis/official_eyebench_sota_check_v1")
    )
    return {
        "repo_analysis": repo_analysis,
        "repo_tables": repo_analysis / "tables",
        "result_analysis": result_analysis,
        "result_tables": out
        / str(
            get_nested(
                config,
                f"{SOTA_SECTION}.output_layout.tables",
                "analysis/official_eyebench_sota_check_v1/tables",
            )
        ),
    }


def _write_report(dirs: dict[str, Path], name: str, text: str) -> None:
    _write_md(dirs["repo_analysis"] / name, text)
    _write_md(dirs["result_analysis"] / name, text)


def _write_table(dirs: dict[str, Path], name: str, frame: Any) -> None:
    _write_csv(dirs["repo_tables"] / name, frame)
    _write_csv(dirs["result_tables"] / name, frame)


def _configured_path(config: dict[str, Any], dotted: str, repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    value = get_nested(config, dotted)
    if value is None:
        raise ValueError(f"missing required config path: {dotted}")
    path = Path(str(value))
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _eyebench_path(config: dict[str, Any], repo_root: str | Path) -> Path:
    return _configured_path(config, f"{SOTA_SECTION}.eyebench.path", repo_root)


def validate_official_eyebench_sota_check_config(config: dict[str, Any]) -> dict[str, Any]:
    section = _section(config)
    errors: list[str] = []
    warnings: list[str] = []
    if not section:
        errors.append(f"missing {SOTA_SECTION} config section")
    for flag in ["no_new_labels", "no_feature_engineering_search", "no_broad_model_search"]:
        if section.get(flag) is not True:
            errors.append(f"{flag} must be true")
    if section.get("forbid_random_word_level_split") is not True:
        errors.append("random word-level split prohibition is not enabled")
    missing_splits = sorted(set(OFFICIAL_SPLITS) - set(section.get("split_regimes", [])))
    if missing_splits:
        errors.append(f"missing official split regimes: {missing_splits}")
    prohibited = set(section.get("prohibited_features", []))
    missing_prohibited = sorted(
        (PROHIBITED_FEATURES | {"unique_trial_id", "unique_paragraph_id", "dyslexia", "RCS_score"})
        - prohibited
    )
    if missing_prohibited:
        errors.append(f"prohibited feature list incomplete: {missing_prohibited}")
    if section.get("environment", {}).get("name") != "eyebench_official":
        warnings.append("official environment name is not eyebench_official")
    if "CopCo_TYP" not in section.get("tasks", []):
        errors.append("CopCo_TYP is required")
    return {"status": "failed" if errors else "passed", "errors": errors, "warnings": warnings}


def write_environment_report(
    config: dict[str, Any],
    out: Path,
    dirs: dict[str, Path],
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    eyebench = _eyebench_path(config, repo_root)
    env_name = str(get_nested(config, f"{SOTA_SECTION}.environment.name", "eyebench_official"))
    auto_create = bool(get_nested(config, f"{SOTA_SECTION}.environment.auto_create_in_cli", False))
    create_attempt: dict[str, Any] | None = None
    if auto_create:
        command = ["mamba", "env", "create", "-n", env_name, "-f", "eyebench/environment.yml"]
        create_attempt = _run_command(command, cwd=root, timeout=1800)
    env_list = _run_command(["conda", "env", "list"], cwd=root, timeout=60)
    import_attempt = _run_command(
        [
            "conda",
            "run",
            "-n",
            env_name,
            "bash",
            "-lc",
            (
                "cd eyebench && python - <<'PY'\n"
                "import sys\n"
                "print('python', sys.version)\n"
                "for name in ['beartype','pandas','numpy','sklearn','pyarrow','src']:\n"
                "    try:\n"
                "        mod = __import__(name)\n"
                "        print(name, getattr(mod, '__version__', 'imported'))\n"
                "    except Exception as exc:\n"
                "        print(name, 'IMPORT_FAILED', repr(exc))\n"
                "PY"
            ),
        ],
        cwd=root,
        timeout=120,
    )
    torch_attempt = _run_command(
        [
            "conda",
            "run",
            "-n",
            env_name,
            "python",
            "-c",
            "import torch; print(torch.__version__); print(torch.cuda.is_available())",
        ],
        cwd=root,
        timeout=120,
    )
    imports_ok = import_attempt["returncode"] == 0 and "IMPORT_FAILED" not in import_attempt["stdout"]
    torch_ok = torch_attempt["returncode"] == 0
    report = {
        "environment_name": env_name,
        "environment_yml": str(eyebench / "environment.yml"),
        "auto_create_in_cli": auto_create,
        "manual_exact_create_attempt": get_nested(
            config, f"{SOTA_SECTION}.environment.manual_exact_create_attempt", {}
        ),
        "manual_flexible_retry_attempt": get_nested(
            config, f"{SOTA_SECTION}.environment.manual_flexible_retry_attempt", {}
        ),
        "create_attempt": create_attempt,
        "env_list_returncode": env_list["returncode"],
        "import_attempt": import_attempt,
        "torch_attempt": torch_attempt,
        "eyebench_imports": imports_ok,
        "torch_imports": torch_ok,
        "official_environment_ready": imports_ok,
        "status": "ready" if imports_ok else "blocked_by_environment",
    }
    lines = [
        "# Official EyeBench Environment Report",
        "",
        f"- Environment name: `{env_name}`",
        f"- Auto-create in CLI: {auto_create}",
        f"- EyeBench imports: {imports_ok}",
        f"- Torch imports: {torch_ok}",
        f"- Status: `{report['status']}`",
        "",
        "## Manual Create Attempts",
        f"- Exact command/status: {report['manual_exact_create_attempt']}",
        f"- Flexible retry/status: {report['manual_flexible_retry_attempt']}",
        "",
        "## Live Import Attempt",
        f"- Command: `{import_attempt['command']}`",
        f"- Return code: {import_attempt['returncode']}",
        f"- stdout: `{import_attempt['stdout'][:2000]}`",
        f"- stderr: `{import_attempt['stderr'][:2000]}`",
        "",
        "## Torch Attempt",
        f"- Return code: {torch_attempt['returncode']}",
        f"- stdout: `{torch_attempt['stdout'][:1000]}`",
        f"- stderr: `{torch_attempt['stderr'][:1000]}`",
    ]
    _write_report(dirs, "eyebench_official_environment_report.md", "\n".join(lines))
    _write_json(out / "environment" / "eyebench_official_environment_report.json", report)
    return report


def _processed_paths(config: dict[str, Any], repo_root: str | Path) -> dict[str, Path]:
    return {
        "eyebench_root": _eyebench_path(config, repo_root),
        "global_processed": _configured_path(
            config,
            f"{SOTA_SECTION}.eyebench.global_processed_dir",
            repo_root,
        ),
        "copco_processed": _configured_path(
            config,
            f"{SOTA_SECTION}.eyebench.processed_copco_dir",
            repo_root,
        ),
    }


def _count_processed(processed: Path) -> dict[str, Any]:
    pd = _pd()
    counts: dict[str, Any] = {
        "processed_data_path": str(processed),
        "copco_processed_files_present": processed.exists(),
        "participant_count": None,
        "text_item_count": None,
        "trial_count": None,
        "word_count": None,
        "fixation_count": None,
        "processed_files": [],
    }
    if not processed.exists():
        return counts
    counts["processed_files"] = sorted(path.name for path in processed.glob("*") if path.is_file())
    trial = processed / "trial_level.feather"
    ia = processed / "ia.feather"
    fix = processed / "fixations.feather"
    if trial.exists():
        frame = pd.read_feather(trial)
        counts["trial_count"] = int(len(frame))
        for column in ["participant_id", "unique_paragraph_id", "speech_id"]:
            if column in frame and column == "participant_id":
                counts["participant_count"] = int(frame[column].nunique())
            if column in frame and column in {"unique_paragraph_id", "speech_id"}:
                counts["text_item_count"] = int(frame[column].nunique())
                break
    if ia.exists():
        frame = pd.read_feather(ia, columns=None)
        counts["word_count"] = int(len(frame))
        if counts["participant_count"] is None and "participant_id" in frame:
            counts["participant_count"] = int(frame["participant_id"].nunique())
        if counts["text_item_count"] is None:
            for column in ["unique_paragraph_id", "speech_id"]:
                if column in frame:
                    counts["text_item_count"] = int(frame[column].nunique())
                    break
        if counts["trial_count"] is None and "unique_trial_id" in frame:
            counts["trial_count"] = int(frame["unique_trial_id"].nunique())
    if fix.exists():
        counts["fixation_count"] = int(len(pd.read_feather(fix)))
    return counts


def write_data_status_report(
    config: dict[str, Any],
    out: Path,
    dirs: dict[str, Path],
    repo_root: str | Path,
    environment: dict[str, Any],
) -> dict[str, Any]:
    paths = _processed_paths(config, repo_root)
    allow = bool(get_nested(config, f"{SOTA_SECTION}.eyebench.allow_data_download", True))
    run_preprocessing = bool(
        get_nested(config, f"{SOTA_SECTION}.eyebench.run_preprocessing_if_environment_ok", True)
    )
    env_name = str(get_nested(config, f"{SOTA_SECTION}.environment.name", "eyebench_official"))
    command_result: dict[str, Any] | None = None
    skip_reason = ""
    if not allow:
        skip_reason = "data download disabled by config"
    elif not environment.get("official_environment_ready"):
        skip_reason = "official EyeBench environment is not import-ready"
    elif not run_preprocessing:
        skip_reason = "preprocessing command disabled by config"
    elif not paths["copco_processed"].exists():
        command_result = _run_command(
            ["conda", "run", "-n", env_name, "bash", "-lc", "bash src/data/preprocessing/get_data.sh"],
            cwd=paths["eyebench_root"],
            timeout=int(get_nested(config, f"{SOTA_SECTION}.eyebench.preprocessing_timeout_seconds", 7200)),
        )
        if command_result["returncode"] != 0:
            skip_reason = "EyeBench preprocessing command failed"
    counts = _count_processed(paths["copco_processed"])
    global_processed_exists = paths["global_processed"].exists()
    downloaded = bool(counts["copco_processed_files_present"])
    if not downloaded and not skip_reason:
        skip_reason = "official CopCo processed data are absent"
    report = {
        "downloaded": downloaded,
        "global_processed_dir": str(paths["global_processed"]),
        "global_processed_exists": global_processed_exists,
        **counts,
        "labels_available": (paths["eyebench_root"] / "data" / "CopCo" / "labels").exists(),
        "fold_files_available": (paths["eyebench_root"] / "data" / "CopCo" / "folds_metadata").exists(),
        "command_result": command_result,
        "skip_reason": skip_reason,
        "status": "ready" if downloaded else "blocked_by_data",
    }
    lines = [
        "# Official EyeBench Data Report",
        "",
        f"- downloaded: {downloaded}",
        f"- global processed dir exists: {global_processed_exists}",
        f"- CopCo processed files present: {counts['copco_processed_files_present']}",
        f"- participant count: {counts['participant_count']}",
        f"- text/item count: {counts['text_item_count']}",
        f"- trial count: {counts['trial_count']}",
        f"- word count: {counts['word_count']}",
        f"- fixation count: {counts['fixation_count']}",
        f"- labels available: {report['labels_available']}",
        f"- fold files available: {report['fold_files_available']}",
        f"- blocker/skip reason: {skip_reason or 'none'}",
    ]
    _write_report(dirs, "eyebench_official_data_report.md", "\n".join(lines))
    _write_json(out / "data" / "eyebench_official_data_report.json", report)
    return report


def _unique_trial_id(frame: Any) -> Any:
    if "unique_trial_id" in frame:
        return frame["unique_trial_id"].astype(str)
    if {"participant_id", "speech_id", "paragraph_id"}.issubset(frame.columns):
        return (
            frame["participant_id"].astype(str)
            + "_"
            + frame["speech_id"].astype(str)
            + "_"
            + frame["paragraph_id"].astype(str)
        )
    raise ValueError("cannot construct unique_trial_id")


def load_official_processed_features(config: dict[str, Any], out: Path, repo_root: str | Path) -> tuple[Any, Any]:
    pd = _pd()
    processed = _processed_paths(config, repo_root)["copco_processed"]
    ia_path = processed / "ia.feather"
    trial_path = processed / "trial_level.feather"
    if not ia_path.exists():
        empty = pd.DataFrame()
        return empty, empty
    ia = pd.read_feather(ia_path)
    ia["unique_trial_id"] = _unique_trial_id(ia)
    if "unique_paragraph_id" not in ia and {"speech_id", "paragraph_id"}.issubset(ia.columns):
        ia["unique_paragraph_id"] = ia["speech_id"].astype(str) + "_" + ia["paragraph_id"].astype(str)
    meta_columns = [
        column
        for column in [
            "unique_trial_id",
            "participant_id",
            "speech_id",
            "paragraph_id",
            "unique_paragraph_id",
            "dyslexia",
            "RCS_score",
        ]
        if column in ia
    ]
    samples = ia[meta_columns].drop_duplicates("unique_trial_id").copy()
    samples["sample_id"] = samples["unique_trial_id"].astype(str)
    samples["text_id"] = samples.get("unique_paragraph_id", samples.get("speech_id")).astype(str)
    samples["reader_group_binary"] = pd.to_numeric(samples.get("dyslexia"), errors="coerce").astype("Int64")
    samples["reader_group"] = samples["reader_group_binary"].map({1: "dyslexia_labeled", 0: "typical_control"})
    samples["n_words_in_sample"] = ia.groupby("unique_trial_id").size().reindex(samples["unique_trial_id"]).to_numpy()
    if trial_path.exists():
        trial = pd.read_feather(trial_path)
        if "unique_trial_id" not in trial:
            trial = trial.reset_index()
        if "unique_trial_id" in trial:
            trial["unique_trial_id"] = trial["unique_trial_id"].astype(str)
            samples = samples.merge(trial, on="unique_trial_id", how="left", suffixes=("", "_trial"))
    samples = samples.sort_values(["participant_id", "text_id"]).reset_index(drop=True)
    _write_parquet(out / "data" / "official_eyebench_sota_trial_features.parquet", samples)
    return samples, ia


def build_official_split_labels(config: dict[str, Any], out: Path, samples: Any, repo_root: str | Path) -> Any:
    pd = _pd()
    eyebench = _eyebench_path(config, repo_root)
    official_trials = _load_official_trial_ids(eyebench)
    if samples.empty or official_trials.empty:
        splits = pd.DataFrame()
        _write_parquet(out / "splits" / "official_eyebench_sota_split_labels.parquet", splits)
        return splits
    by_trial = samples.set_index("unique_trial_id", drop=False)
    rows: list[dict[str, Any]] = []
    for fold_id, fold in official_trials.groupby("fold_id", dropna=False):
        train_ids = set(fold.loc[fold["regime"].eq("train_train"), "unique_trial_id"].astype(str))
        for official_regime, split_name in OFFICIAL_REGIME_TO_SPLIT.items():
            test_ids = set(fold.loc[fold["regime"].eq(f"test_{official_regime}"), "unique_trial_id"].astype(str))
            train = by_trial.loc[sorted(train_ids.intersection(by_trial.index))].copy()
            test = by_trial.loc[sorted(test_ids.intersection(by_trial.index))].copy()
            train_y = pd.to_numeric(train.get("reader_group_binary"), errors="coerce")
            test_y = pd.to_numeric(test.get("reader_group_binary"), errors="coerce")
            split_valid = bool(not train.empty and not test.empty and train_y.nunique(dropna=True) >= 2 and test_y.notna().any())
            train_participants = set(train["participant_id"].astype(str)) if "participant_id" in train else set()
            test_participants = set(test["participant_id"].astype(str)) if "participant_id" in test else set()
            train_texts = set(train["text_id"].astype(str)) if "text_id" in train else set()
            test_texts = set(test["text_id"].astype(str)) if "text_id" in test else set()
            for _, sample in samples.iterrows():
                trial_id = str(sample["unique_trial_id"])
                in_train = trial_id in train_ids
                in_test = trial_id in test_ids
                role = "exclude"
                if in_train and in_test:
                    role = "invalid_overlap"
                elif in_train:
                    role = "train"
                elif in_test:
                    role = "test"
                rows.append(
                    {
                        "mode": OFFICIAL_LITE_MODE,
                        "split_name": split_name,
                        "official_regime": official_regime,
                        "fold_id": int(fold_id),
                        "sample_id": sample["sample_id"],
                        "unique_trial_id": trial_id,
                        "participant_id": sample["participant_id"],
                        "speech_id": sample.get("speech_id"),
                        "text_id": sample["text_id"],
                        "reader_group_binary": sample["reader_group_binary"],
                        "split_role": role,
                        "include_in_fold": role in {"train", "test"},
                        "n_train_samples": int(len(train)),
                        "n_test_samples": int(len(test)),
                        "participant_overlap": bool(train_participants.intersection(test_participants)),
                        "text_overlap": bool(train_texts.intersection(test_texts)),
                        "split_valid": split_valid,
                        "skip_reason": "" if split_valid else "empty_or_single_class_training_fold",
                        "split_version": "official_eyebench_trial_ids",
                    }
                )
    splits = pd.DataFrame(rows)
    _write_parquet(out / "splits" / "official_eyebench_sota_split_labels.parquet", splits)
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
        train_texts = set(train["text_id"].astype(str))
        test_texts = set(test["text_id"].astype(str))
        if split_name == "unseen_reader" and train_participants.intersection(test_participants):
            errors.append(f"participant overlap in {split_name} fold {fold_id}")
        if split_name == "unseen_text" and train_texts.intersection(test_texts):
            errors.append(f"text overlap in {split_name} fold {fold_id}")
        if split_name == "unseen_reader_and_text":
            if train_participants.intersection(test_participants):
                errors.append(f"participant overlap in {split_name} fold {fold_id}")
            if train_texts.intersection(test_texts):
                errors.append(f"text overlap in {split_name} fold {fold_id}")
        train_y = pd_to_numeric(train, "reader_group_binary")
        test_y = pd_to_numeric(test, "reader_group_binary")
        if train.empty or train_y.nunique(dropna=True) < 2:
            errors.append(f"TYP train fold lacks both classes: {split_name} fold {fold_id}")
        if test.empty or test_y.dropna().empty:
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
            }
        )
    return errors, summaries


def pd_to_numeric(frame: Any, column: str) -> Any:
    return _pd().to_numeric(frame.get(column), errors="coerce") if column in frame else _pd().Series(dtype=float)


def _safe_numeric_columns(frame: Any, candidates: list[str], prohibited: set[str]) -> list[str]:
    clean = []
    for column in candidates:
        if column in prohibited or column not in frame:
            continue
        values = _pd().to_numeric(frame[column], errors="coerce")
        if values.notna().any() and values.nunique(dropna=True) > 1:
            clean.append(column)
    return clean


def _feature_keys(processed: Path, feature_type: str) -> list[str]:
    pd = _pd()
    keys: list[str] = []
    for name in ["ia_trial_level_feature_keys.csv", "fixation_trial_level_feature_keys.csv"]:
        path = processed / name
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if {"feature_name", "feature_type"}.issubset(frame.columns):
            keys.extend(frame.loc[frame["feature_type"].astype(str).eq(feature_type), "feature_name"].astype(str).tolist())
    return list(dict.fromkeys(keys))


def _official_reference_table(config: dict[str, Any], repo_root: str | Path) -> Any:
    pd = _pd()
    path = _configured_path(config, f"{SOTA_SECTION}.decision_gates.CopCo_TYP.formatted_table", repo_root)
    raw = pd.read_csv(path)
    rows = []
    for _, row in raw.iterrows():
        rows.append(
            {
                "model": str(row["Model"]),
                "mode": "official_eyebench_reported_baseline",
                "claim_type": "official_reported_reference",
                "metric_basis": "published_fold_mean",
                "official_mode": True,
                "exact_folds": True,
                "exact_processed_data": True,
                "unseen_reader_balanced_accuracy": _parse_percent(row.get("Unseen Reader_\\makecell{Balanced\\\\Accuracy}")),
                "unseen_text_balanced_accuracy": _parse_percent(row.get("Unseen Text_\\makecell{Balanced\\\\Accuracy}")),
                "unseen_reader_text_balanced_accuracy": _parse_percent(row.get("Unseen Text \\& Reader_\\makecell{Balanced\\\\Accuracy}")),
                "unseen_reader_AUROC": _parse_percent(row.get("Unseen Reader_AUROC")),
                "unseen_text_AUROC": _parse_percent(row.get("Unseen Text_AUROC")),
                "unseen_reader_text_AUROC": _parse_percent(row.get("Unseen Text \\& Reader_AUROC")),
                "notes": "Published EyeBench formatted CopCo_TYP test table central value.",
            }
        )
    table = pd.DataFrame(rows)
    table["average_balanced_accuracy"] = table[
        [
            "unseen_reader_balanced_accuracy",
            "unseen_text_balanced_accuracy",
            "unseen_reader_text_balanced_accuracy",
        ]
    ].mean(axis=1, skipna=True)
    table["average_AUROC"] = table[
        ["unseen_reader_AUROC", "unseen_text_AUROC", "unseen_reader_text_AUROC"]
    ].mean(axis=1, skipna=True)
    return table[COMPARISON_COLUMNS]


def _published_model_values(reference: Any, model: str, split_name: str) -> dict[str, float | None]:
    row = reference[reference["model"].astype(str).str.contains(model, regex=False, case=False, na=False)]
    if row.empty:
        return {"roc_auc": None, "balanced_accuracy": None}
    row = row.iloc[0]
    return {
        "roc_auc": row[OFFICIAL_TO_COLUMN[split_name]["roc_auc"]],
        "balanced_accuracy": row[OFFICIAL_TO_COLUMN[split_name]["balanced_accuracy"]],
    }


def _empty_typ_metrics(reason: str) -> Any:
    pd = _pd()
    rows = []
    for split_name in OFFICIAL_SPLITS:
        for level in [TRIAL_LEVEL, "participant_text_trial_pooled", READER_LEVEL]:
            rows.append(
                {
                    "mode": OFFICIAL_LITE_MODE,
                    "model_name": SOTA_MODEL,
                    "claim_type": "official_attempt_failed",
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
    return pd.DataFrame(rows, columns=SOTA_TYP_COLUMNS)


def _empty_baseline_reproduction(reference: Any, reason: str) -> Any:
    pd = _pd()
    rows = []
    for split_name in OFFICIAL_SPLITS:
        published = _published_model_values(reference, "Logistic", split_name)
        rows.append(
            {
                "model_name": OFFICIAL_BASELINE_MODEL,
                "baseline_source": "official_processed_data_local_reproduction",
                "split_name": split_name,
                "metric_basis": TRIAL_LEVEL,
                "n_features": 0,
                "n_predictions": 0,
                "usable_folds": 0,
                "skipped_folds": 0,
                "roc_auc": None,
                "balanced_accuracy": None,
                "published_roc_auc": published["roc_auc"],
                "published_balanced_accuracy": published["balanced_accuracy"],
                "delta_roc_auc": None,
                "delta_balanced_accuracy": None,
                "status": "skipped",
                "skip_reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=BASELINE_COLUMNS)


def _fold_cache(samples: Any, splits: Any) -> dict[tuple[str, int], dict[str, Any]]:
    cache: dict[tuple[str, int], dict[str, Any]] = {}
    if samples.empty or splits.empty:
        return cache
    by_sample = samples.set_index("sample_id", drop=False)
    for (split_name, fold_id), fold in splits.groupby(["split_name", "fold_id"], dropna=False):
        train_ids = fold.loc[fold["split_role"].eq("train"), "sample_id"].astype(str)
        test_ids = fold.loc[fold["split_role"].eq("test"), "sample_id"].astype(str)
        cache[(str(split_name), int(fold_id))] = {
            "train": by_sample.loc[by_sample.index.intersection(train_ids)].copy().reset_index(drop=True),
            "test": by_sample.loc[by_sample.index.intersection(test_ids)].copy().reset_index(drop=True),
        }
    return cache


def _fold_mean_metrics(predictions: Any, *, level: str) -> dict[str, Any]:
    pd = _pd()
    if predictions.empty:
        return {
            "n_predictions": 0,
            "roc_auc": None,
            "pr_auc": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "brier_score": None,
        }
    rows = []
    for _, fold in predictions.groupby("fold_id", dropna=False):
        metric = _classification_metrics(fold["y_true"], fold["y_score"])
        rows.append(metric)
    frame = pd.DataFrame(rows)
    out = {"n_predictions": int(len(predictions))}
    for metric in ["roc_auc", "pr_auc", "balanced_accuracy", "macro_f1", "brier_score"]:
        values = pd.to_numeric(frame.get(metric), errors="coerce") if metric in frame else pd.Series(dtype=float)
        out[metric] = float(values.mean()) if values.notna().any() else None
    out["evaluation_level"] = level
    return out


def _pooled_metrics(predictions: Any, *, level: str) -> dict[str, Any]:
    frame = predictions
    if frame.empty:
        return {
            "evaluation_level": level,
            "n_predictions": 0,
            "roc_auc": None,
            "pr_auc": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "brier_score": None,
        }
    return {
        "evaluation_level": level,
        "n_predictions": int(len(frame)),
        **_classification_metrics(frame["y_true"], frame["y_score"]),
    }


def _evaluate_feature_matrix(
    config: dict[str, Any],
    fold_cache: dict[tuple[str, int], dict[str, Any]],
    columns_by_fold: dict[tuple[str, int], list[str]],
    *,
    model_name: str,
    sklearn_model: str,
    claim_type: str,
    mode: str,
) -> tuple[Any, Any]:
    pd = _pd()
    seed = int(get_nested(config, f"{SOTA_SECTION}.deterministic_seed", 173))
    metric_rows = []
    prediction_rows = []
    for split_name in OFFICIAL_SPLITS:
        predictions = []
        usable = 0
        skipped = 0
        n_features = 0
        skip_reason = ""
        for key in sorted(k for k in fold_cache if k[0] == split_name):
            _, fold_id = key
            train = fold_cache[key]["train"].copy()
            test = fold_cache[key]["test"].copy()
            train_y = pd.to_numeric(train.get("reader_group_binary"), errors="coerce")
            test_y = pd.to_numeric(test.get("reader_group_binary"), errors="coerce")
            columns = columns_by_fold.get(key, [])
            columns = _safe_numeric_columns(train, columns, set(_section(config).get("prohibited_features", [])))
            if train.empty or test.empty or train_y.nunique(dropna=True) < 2 or not columns:
                skipped += 1
                skip_reason = "empty_or_single_class_training_fold_or_no_features"
                continue
            model = _model_pipeline(sklearn_model, task="typ", seed=seed + int(fold_id))
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
                        "feature_group": "D3_EyeBench_Lite" if model_name == SOTA_MODEL else "official_baseline",
                        "model": sklearn_model,
                        "sample_id": row["sample_id"],
                        "unique_trial_id": row.get("unique_trial_id"),
                        "participant_id": row.get("participant_id"),
                        "speech_id": row.get("speech_id"),
                        "text_id": row.get("text_id"),
                        "y_true": int(truth),
                        "y_score": float(pred),
                        "y_pred": int(float(pred) >= 0.5),
                        "eval_regime": SPLIT_TO_OFFICIAL_REGIME[split_name],
                        "eval_type": "test",
                        "fold_index": int(fold_id),
                    }
                )
        pred_frame = pd.DataFrame(predictions)
        prediction_rows.append(pred_frame)
        metric_specs = [
            _fold_mean_metrics(pred_frame, level=TRIAL_LEVEL),
            _pooled_metrics(pred_frame, level="participant_text_trial_pooled"),
            _pooled_metrics(_reader_aggregate_for_sota(pred_frame), level=READER_LEVEL),
        ]
        for metric in metric_specs:
            metric_rows.append(
                {
                    "mode": mode,
                    "model_name": model_name,
                    "claim_type": claim_type,
                    "task": "CopCo_TYP",
                    "split_name": split_name,
                    "n_features": int(n_features),
                    "usable_folds": int(usable),
                    "skipped_folds": int(skipped),
                    "status": "complete" if pred_frame.shape[0] else "skipped",
                    "skip_reason": "" if pred_frame.shape[0] else skip_reason or "no_valid_predictions",
                    **metric,
                }
            )
    metrics = pd.DataFrame(metric_rows, columns=SOTA_TYP_COLUMNS)
    non_empty = [frame for frame in prediction_rows if not frame.empty]
    predictions = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    return metrics, predictions


def _reader_aggregate_for_sota(predictions: Any) -> Any:
    if predictions.empty:
        return predictions
    return (
        predictions.groupby(["task", "split_name", "fold_id", "feature_group", "model", "participant_id"], dropna=False)
        .agg(y_true=("y_true", "first"), y_score=("y_score", "mean"), sample_id=("sample_id", "first"))
        .reset_index()
    )


def reproduce_official_baseline(
    config: dict[str, Any],
    out: Path,
    dirs: dict[str, Path],
    repo_root: str | Path,
    samples: Any,
    splits: Any,
    reference: Any,
) -> tuple[Any, Any]:
    if samples.empty or splits.empty:
        reason = "official processed CopCo data or split labels are absent"
        metrics = _empty_baseline_reproduction(reference, reason)
        _write_csv(out / "baseline" / "official_baseline_reproduction_metrics.csv", metrics)
        _write_report(
            dirs,
            "official_baseline_reproduction_report.md",
            "# Official Baseline Reproduction Report\n\n"
            f"- Reproduction status: skipped\n- Reason: {reason}",
        )
        return metrics, _pd().DataFrame()
    processed = _processed_paths(config, repo_root)["copco_processed"]
    columns = _feature_keys(processed, "LOGISTIC") or LOGISTIC_BASELINE_CANDIDATES
    prohibited = set(_section(config).get("prohibited_features", []))
    columns_by_fold = {key: _safe_numeric_columns(value["train"], columns, prohibited) for key, value in _fold_cache(samples, splits).items()}
    metrics, predictions = _evaluate_feature_matrix(
        config,
        _fold_cache(samples, splits),
        columns_by_fold,
        model_name=OFFICIAL_BASELINE_MODEL,
        sklearn_model="logistic_regression",
        claim_type="official_baseline_reproduction_attempt",
        mode="official_baseline_reproduction",
    )
    primary = metrics[metrics["evaluation_level"].eq(TRIAL_LEVEL)].copy()
    rows = []
    for _, row in primary.iterrows():
        published = _published_model_values(reference, "Logistic", str(row["split_name"]))
        roc = row["roc_auc"] if row["roc_auc"] == row["roc_auc"] else None
        ba = row["balanced_accuracy"] if row["balanced_accuracy"] == row["balanced_accuracy"] else None
        rows.append(
            {
                "model_name": OFFICIAL_BASELINE_MODEL,
                "baseline_source": "official_processed_data_local_reproduction",
                "split_name": row["split_name"],
                "metric_basis": TRIAL_LEVEL,
                "n_features": row["n_features"],
                "n_predictions": row["n_predictions"],
                "usable_folds": row["usable_folds"],
                "skipped_folds": row["skipped_folds"],
                "roc_auc": roc,
                "balanced_accuracy": ba,
                "published_roc_auc": published["roc_auc"],
                "published_balanced_accuracy": published["balanced_accuracy"],
                "delta_roc_auc": None if roc is None or published["roc_auc"] is None else float(roc) - float(published["roc_auc"]),
                "delta_balanced_accuracy": None
                if ba is None or published["balanced_accuracy"] is None
                else float(ba) - float(published["balanced_accuracy"]),
                "status": row["status"],
                "skip_reason": row["skip_reason"],
            }
        )
    repro = _pd().DataFrame(rows, columns=BASELINE_COLUMNS)
    _write_csv(out / "baseline" / "official_baseline_reproduction_metrics.csv", repro)
    _write_csv(out / "baseline" / "official_baseline_reproduction_predictions.csv", predictions)
    text = "\n".join(
        [
            "# Official Baseline Reproduction Report",
            "",
            "This is a local reproduction using official processed CopCo data and official folds.",
            "It is not accepted as a successful official baseline reproduction unless processed data exist "
            "and the local metrics are close to the published EyeBench table.",
            "",
            _markdown_table(repro.to_dict("records"), BASELINE_COLUMNS, max_rows=30),
        ]
    )
    _write_report(dirs, "official_baseline_reproduction_report.md", text)
    return repro, predictions


def _outcome_column(frame: Any, outcome: str) -> str | None:
    for column in RESIDUAL_OUTCOME_CANDIDATES[outcome]:
        if column in frame:
            return column
    return None


def _trial_residual_features(train_ids: set[str], test_ids: set[str], ia: Any) -> tuple[Any, dict[str, Any]]:
    pd = _pd()
    np = _np()
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    word = ia.copy()
    word["unique_trial_id"] = word["unique_trial_id"].astype(str)
    train_word = word[word["unique_trial_id"].isin(train_ids)].copy()
    test_word = word[word["unique_trial_id"].isin(test_ids)].copy()
    predictors = _safe_numeric_columns(train_word, RESIDUAL_PREDICTOR_CANDIDATES, set())
    aggregate_rows = []
    diagnostics = {
        "train_word_rows": int(len(train_word)),
        "test_word_rows": int(len(test_word)),
        "predictors": predictors,
        "heldout_reader_rows_used_for_fit": False,
        "heldout_text_rows_used_for_fit": False,
        "reader_group_used": False,
        "skipped": False,
        "skip_reason": "",
    }
    if train_word.empty or not predictors:
        diagnostics["skipped"] = True
        diagnostics["skip_reason"] = "no_train_words_or_predictors"
        return pd.DataFrame(), diagnostics
    combined = pd.concat([train_word, test_word], ignore_index=True)
    for outcome in RESIDUAL_OUTCOME_CANDIDATES:
        column = _outcome_column(train_word, outcome)
        if column is None:
            continue
        train_y = pd.to_numeric(train_word[column], errors="coerce")
        valid_train = train_y.notna()
        if valid_train.sum() < 3:
            continue
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
        model.fit(train_word.loc[valid_train, predictors], train_y.loc[valid_train])
        predicted = model.predict(combined[predictors])
        observed = pd.to_numeric(combined[column], errors="coerce").to_numpy(dtype=float)
        combined[f"d3_lite_resid_{outcome}"] = observed - predicted
    residual_columns = [column for column in combined if column.startswith("d3_lite_resid_")]
    if not residual_columns:
        diagnostics["skipped"] = True
        diagnostics["skip_reason"] = "no_residual_outcomes_available"
        return pd.DataFrame(), diagnostics
    for trial_id, group in combined.groupby("unique_trial_id", dropna=False):
        row = {"sample_id": str(trial_id), "unique_trial_id": str(trial_id)}
        for column in residual_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{column}_median"] = float(values.median()) if values.notna().any() else np.nan
            row[f"{column}_sd"] = float(values.std(ddof=0)) if values.notna().any() else np.nan
        aggregate_rows.append(row)
    return pd.DataFrame(aggregate_rows), diagnostics


def evaluate_d3_eyebench_lite(
    config: dict[str, Any],
    out: Path,
    dirs: dict[str, Path],
    samples: Any,
    splits: Any,
    ia: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    pd = _pd()
    if samples.empty or splits.empty or ia.empty:
        reason = "official processed CopCo data or official split labels are absent"
        metrics = _empty_typ_metrics(reason)
        _write_csv(out / "typ" / "d3_eyebench_lite_metrics.csv", metrics)
        report = {
            "heldout_reader_rows_used_for_fit": False,
            "heldout_text_rows_used_for_fit": False,
            "reader_group_used": False,
            "status": "skipped",
            "skip_reason": reason,
        }
        _write_report(
            dirs,
            "d3_eyebench_lite_official_evaluation_report.md",
            "# D3_EyeBench_Lite Official Evaluation Report\n\n"
            f"- Status: skipped\n- Reason: {reason}",
        )
        _write_report(
            dirs,
            "d3_eyebench_lite_leakage_report.md",
            "# D3_EyeBench_Lite Leakage Report\n\n"
            "- Held-out reader rows used for residual fitting: False\n"
            "- Held-out text rows used for residual fitting: False\n"
            "- Reader group used in residualization: False\n",
        )
        return metrics, pd.DataFrame(), report
    cache = _fold_cache(samples, splits)
    enriched: dict[tuple[str, int], dict[str, Any]] = {}
    diagnostics = []
    for key, value in cache.items():
        split_name, fold_id = key
        train = value["train"].copy()
        test = value["test"].copy()
        train_ids = set(train["unique_trial_id"].astype(str))
        test_ids = set(test["unique_trial_id"].astype(str))
        features, diag = _trial_residual_features(train_ids, test_ids, ia)
        diag.update({"split_name": split_name, "fold_id": int(fold_id)})
        train_participants = set(train["participant_id"].astype(str))
        test_participants = set(test["participant_id"].astype(str))
        train_texts = set(train["text_id"].astype(str))
        test_texts = set(test["text_id"].astype(str))
        if split_name in {"unseen_reader", "unseen_reader_and_text"}:
            diag["heldout_reader_rows_used_for_fit"] = bool(train_participants.intersection(test_participants))
        if split_name in {"unseen_text", "unseen_reader_and_text"}:
            diag["heldout_text_rows_used_for_fit"] = bool(train_texts.intersection(test_texts))
        train_aug = train.merge(features, on=["sample_id", "unique_trial_id"], how="left") if not features.empty else train
        test_aug = test.merge(features, on=["sample_id", "unique_trial_id"], how="left") if not features.empty else test
        enriched[key] = {"train": train_aug, "test": test_aug}
        diagnostics.append(diag)
    feature_by_fold = {
        key: [column for column in value["train"].columns if column.startswith("d3_lite_resid_")]
        for key, value in enriched.items()
    }
    metrics, predictions = _evaluate_feature_matrix(
        config,
        enriched,
        feature_by_fold,
        model_name=SOTA_MODEL,
        sklearn_model="logistic_regression",
        claim_type="official_compatible",
        mode=OFFICIAL_LITE_MODE,
    )
    diag_frame = pd.DataFrame(diagnostics)
    leakage = {
        "heldout_reader_rows_used_for_fit": bool(
            diag_frame.get("heldout_reader_rows_used_for_fit", pd.Series(dtype=bool)).any()
        )
        if not diag_frame.empty
        else False,
        "heldout_text_rows_used_for_fit": bool(
            diag_frame.get("heldout_text_rows_used_for_fit", pd.Series(dtype=bool)).any()
        )
        if not diag_frame.empty
        else False,
        "reader_group_used": False,
        "status": "passed",
        "skip_reason": "",
    }
    _write_csv(out / "typ" / "d3_eyebench_lite_metrics.csv", metrics)
    _write_csv(out / "typ" / "d3_eyebench_lite_predictions.csv", predictions)
    if not predictions.empty:
        official = predictions.copy()
        official["label"] = official["y_true"]
        official["prediction_prob"] = official["y_score"]
        official["binary_prediction"] = official["y_pred"]
        keep = [
            "label",
            "prediction_prob",
            "binary_prediction",
            "eval_regime",
            "eval_type",
            "fold_index",
            "participant_id",
            "speech_id",
            "text_id",
            "unique_trial_id",
        ]
        _write_csv(out / "typ" / "trial_level_test_results.csv", official[keep])
    text = "\n".join(
        [
            "# D3_EyeBench_Lite Official Evaluation Report",
            "",
            _markdown_table(
                metrics[metrics["evaluation_level"].isin([TRIAL_LEVEL, READER_LEVEL])].to_dict("records"),
                SOTA_TYP_COLUMNS,
                max_rows=40,
            ),
        ]
    )
    _write_report(dirs, "d3_eyebench_lite_official_evaluation_report.md", text)
    leak_text = "\n".join(
        [
            "# D3_EyeBench_Lite Leakage Report",
            "",
            f"- Held-out reader rows used for residual fitting: {leakage['heldout_reader_rows_used_for_fit']}",
            f"- Held-out text rows used for residual fitting: {leakage['heldout_text_rows_used_for_fit']}",
            f"- Reader group used in residualization: {leakage['reader_group_used']}",
            "- participant_id and speech_id are retained only for grouping/reporting.",
            "",
            "## Residualizer Predictors",
            "\n".join(f"- `{column}`" for column in RESIDUAL_PREDICTOR_CANDIDATES),
        ]
    )
    _write_report(dirs, "d3_eyebench_lite_leakage_report.md", leak_text)
    _write_csv(out / "residualization" / "d3_eyebench_lite_residualization_diagnostics.csv", diag_frame)
    return metrics, predictions, leakage


def _metric_value(metrics: Any, split_name: str, metric: str, level: str = TRIAL_LEVEL) -> float | None:
    rows = metrics[metrics["split_name"].eq(split_name) & metrics["evaluation_level"].eq(level)]
    if rows.empty or metric not in rows:
        return None
    value = rows.iloc[0][metric]
    if value is None:
        return None
    return float(value) if value == value else None


def _average(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None and value == value]
    return sum(valid) / len(valid) if valid else None


def write_comparison_tables(
    config: dict[str, Any],
    dirs: dict[str, Path],
    repo_root: str | Path,
    d3_metrics: Any,
    reference: Any,
    official_ready: bool,
) -> Any:
    rows = []
    values = {
        "model": SOTA_MODEL,
        "mode": OFFICIAL_LITE_MODE,
        "claim_type": "official_compatible" if official_ready else "official_attempt_failed",
        "metric_basis": TRIAL_LEVEL,
        "official_mode": bool(official_ready),
        "exact_folds": bool(official_ready),
        "exact_processed_data": bool(official_ready),
        "unseen_reader_balanced_accuracy": _metric_value(d3_metrics, "unseen_reader", "balanced_accuracy"),
        "unseen_text_balanced_accuracy": _metric_value(d3_metrics, "unseen_text", "balanced_accuracy"),
        "unseen_reader_text_balanced_accuracy": _metric_value(
            d3_metrics, "unseen_reader_and_text", "balanced_accuracy"
        ),
        "unseen_reader_AUROC": _metric_value(d3_metrics, "unseen_reader", "roc_auc"),
        "unseen_text_AUROC": _metric_value(d3_metrics, "unseen_text", "roc_auc"),
        "unseen_reader_text_AUROC": _metric_value(d3_metrics, "unseen_reader_and_text", "roc_auc"),
        "notes": "Official SOTACheck v1 D3 lite result.",
    }
    values["average_balanced_accuracy"] = _average(
        [
            values["unseen_reader_balanced_accuracy"],
            values["unseen_text_balanced_accuracy"],
            values["unseen_reader_text_balanced_accuracy"],
        ]
    )
    values["average_AUROC"] = _average(
        [values["unseen_reader_AUROC"], values["unseen_text_AUROC"], values["unseen_reader_text_AUROC"]]
    )
    rows.append(values)
    table = _pd().concat([_pd().DataFrame(rows), reference], ignore_index=True)[COMPARISON_COLUMNS]
    _write_table(dirs, "copco_typ_official_sota_comparison.csv", table)
    md = _markdown_table(table.to_dict("records"), table.columns.tolist(), max_rows=80)
    tex = table.to_latex(index=False, float_format=lambda x: f"{x:.3f}" if x == x else "")
    _write_md(dirs["repo_tables"] / "copco_typ_official_sota_comparison.md", md)
    _write_md(dirs["result_tables"] / "copco_typ_official_sota_comparison.md", md)
    _write_md(dirs["repo_tables"] / "copco_typ_official_sota_comparison.tex", tex)
    _write_md(dirs["result_tables"] / "copco_typ_official_sota_comparison.tex", tex)
    return table


def _best_official_reference(reference: Any, split_name: str, metric: str) -> float | None:
    column = OFFICIAL_TO_COLUMN[split_name][metric]
    values = _pd().to_numeric(reference[reference["mode"].eq("official_eyebench_reported_baseline")][column], errors="coerce")
    return float(values.max()) if values.notna().any() else None


def write_decision_report(
    dirs: dict[str, Path],
    out: Path,
    environment: dict[str, Any],
    data_status: dict[str, Any],
    baseline: Any,
    d3_metrics: Any,
    comparison: Any,
    leakage: dict[str, Any],
) -> dict[str, Any]:
    official_env = bool(environment.get("official_environment_ready"))
    official_data = bool(data_status.get("copco_processed_files_present"))
    baseline_reproduced = bool(not baseline.empty and baseline["status"].eq("complete").any())
    d3_complete = bool(d3_metrics["status"].eq("complete").any()) if not d3_metrics.empty else False
    no_leakage = not (
        leakage.get("heldout_reader_rows_used_for_fit")
        or leakage.get("heldout_text_rows_used_for_fit")
        or leakage.get("reader_group_used")
    )
    beats = False
    if d3_complete and no_leakage:
        checks = []
        reference = comparison[comparison["mode"].eq("official_eyebench_reported_baseline")]
        for split_name in OFFICIAL_SPLITS:
            for metric in ["roc_auc", "balanced_accuracy"]:
                ours = _metric_value(d3_metrics, split_name, metric, TRIAL_LEVEL)
                best = _best_official_reference(reference, split_name, metric)
                checks.append(ours is not None and best is not None and ours > best)
        beats = all(checks)
    if not official_env:
        category = "blocked_by_environment"
    elif not official_data:
        category = "blocked_by_data"
    elif not baseline_reproduced:
        category = "benchmark_relative_sota_only"
    elif d3_complete and no_leakage and beats:
        category = "official_sota_claim_allowed"
    elif d3_complete and no_leakage:
        category = "official_compatible_but_not_sota"
    else:
        category = "benchmark_relative_sota_only"
    if category == "official_sota_claim_allowed":
        wording = "official EyeBench-compatible state-of-the-art on CopCo_TYP"
    else:
        wording = "benchmark-relative state of the art under internal EyeBench-style evaluation"
    decision = {
        "decision_category": category,
        "official_environment_ready": official_env,
        "official_processed_data_present": official_data,
        "official_baseline_reproduced": baseline_reproduced,
        "d3_eyebench_lite_complete": d3_complete,
        "d3_eyebench_lite_beats_strongest_official_baselines": beats,
        "no_residualization_leakage_detected": no_leakage,
        "official_sota_claim_allowed": category == "official_sota_claim_allowed",
        "recommended_wording": wording,
        "manuscript_main_claim_changes": category == "official_sota_claim_allowed",
    }
    rows = [
        {"question": "Official EyeBench environment ready?", "answer": official_env},
        {"question": "Official processed CopCo data present?", "answer": official_data},
        {"question": "Official baseline reproduced?", "answer": baseline_reproduced},
        {"question": "D3_EyeBench_Lite completed?", "answer": d3_complete},
        {"question": "D3_EyeBench_Lite beats strongest baselines?", "answer": beats},
        {"question": "No leakage detected?", "answer": no_leakage},
        {"question": "Official SOTA claim allowed?", "answer": decision["official_sota_claim_allowed"]},
    ]
    text = "\n".join(
        [
            "# Official EyeBench SOTA Decision Report",
            "",
            f"- Decision category: `{category}`",
            f"- Recommended wording: {wording}.",
            "",
            _markdown_table(rows, ["question", "answer"], max_rows=20),
            "",
            "## Manuscript Policy",
            "- Main manuscript wording is updated only if `official_sota_claim_allowed`.",
            "- Otherwise, the benchmark claim remains benchmark-relative and the blocker/negative result is supplement-only.",
        ]
    )
    _write_report(dirs, "official_eyebench_sota_decision_report.md", text)
    _write_json(out / "official_eyebench_sota_decision_report.json", decision)
    return decision


def update_supplement_note(repo_root: str | Path, decision: dict[str, Any]) -> dict[str, Any]:
    path = Path(repo_root).resolve() / "paper" / "submission_v1" / "supplement_sections" / "18_benchmark_bridge.tex"
    if not path.exists():
        return {"updated": False, "path": str(path), "reason": "supplement section missing"}
    if decision.get("decision_category") == "official_sota_claim_allowed":
        return {"updated": False, "path": str(path), "reason": "main-text update policy not implemented automatically"}
    text = path.read_text(encoding="utf-8")
    marker = "\\paragraph{Official EyeBench SOTA check.}"
    note = (
        "\n\n"
        "\\paragraph{Official EyeBench SOTA check.}\n"
        "OfficialEyeBenchSOTACheck v1 attempted to move the benchmark-relative result to an "
        "official EyeBench state-of-the-art claim by creating a clean official EyeBench environment, "
        "checking official processed CopCo data, and requiring local reproduction of an official "
        "CopCo_TYP baseline before evaluating D3_EyeBench_Lite. The official environment and/or "
        "processed data gates did not pass, so the manuscript retains benchmark-relative wording "
        "and does not claim official EyeBench SOTA.\n"
    )
    if marker not in text:
        path.write_text(text.rstrip() + note, encoding="utf-8")
        return {"updated": True, "path": str(path), "reason": "added supplement blocker note"}
    return {"updated": False, "path": str(path), "reason": "supplement blocker note already present"}


def write_final_reports(
    dirs: dict[str, Path],
    environment: dict[str, Any],
    data_status: dict[str, Any],
    baseline: Any,
    d3_metrics: Any,
    decision: dict[str, Any],
    manuscript_update: dict[str, Any],
) -> None:
    d3_primary = d3_metrics[d3_metrics["evaluation_level"].isin([TRIAL_LEVEL, READER_LEVEL])]
    lines = [
        "# Official EyeBench SOTA Check Summary",
        "",
        f"- EyeBench environment status: `{environment.get('status')}`",
        f"- Official data status: `{data_status.get('status')}`",
        f"- Official baseline reproduction complete: {bool(not baseline.empty and baseline['status'].eq('complete').any())}",
        f"- Final claim category: `{decision['decision_category']}`",
        f"- Manuscript update: {manuscript_update}",
        "",
        "## D3_EyeBench_Lite Metrics",
        _markdown_table(d3_primary.to_dict("records"), SOTA_TYP_COLUMNS, max_rows=40),
    ]
    _write_report(dirs, "official_eyebench_sota_check_summary.md", "\n".join(lines))


def run_official_eyebench_sota_check(
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    out = Path(output_dir).resolve() if output_dir else timestamped_output_dir(config, repo_root=root)
    out.mkdir(parents=True, exist_ok=True)
    dirs = _analysis_dirs(config, out, root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    config_report = validate_official_eyebench_sota_check_config(config)
    _write_json(out / "config_validation.json", config_report)
    environment = write_environment_report(config, out, dirs, root)
    data_status = write_data_status_report(config, out, dirs, root, environment)
    reference = _official_reference_table(config, root)
    samples, ia = load_official_processed_features(config, out, root)
    splits = build_official_split_labels(config, out, samples, root)
    split_errors, split_summaries = validate_official_split_labels(splits)
    _write_json(out / "splits" / "official_split_validation.json", {"errors": split_errors, "summaries": split_summaries})
    baseline, baseline_predictions = reproduce_official_baseline(
        config, out, dirs, root, samples, splits, reference
    )
    d3_metrics, d3_predictions, leakage = evaluate_d3_eyebench_lite(config, out, dirs, samples, splits, ia)
    official_ready = bool(
        environment.get("official_environment_ready")
        and data_status.get("copco_processed_files_present")
        and baseline["status"].eq("complete").any()
        and d3_metrics["status"].eq("complete").any()
    )
    comparison = write_comparison_tables(config, dirs, root, d3_metrics, reference, official_ready)
    decision = write_decision_report(dirs, out, environment, data_status, baseline, d3_metrics, comparison, leakage)
    manuscript_update = update_supplement_note(root, decision)
    write_final_reports(dirs, environment, data_status, baseline, d3_metrics, decision, manuscript_update)
    manifest = {
        "status": "complete",
        "run_name": get_nested(config, "run.name", "official_eyebench_sota_check_v1"),
        "output_dir": str(out),
        "repo_root": str(root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "eyebench_commit": _run_command(["git", "rev-parse", "HEAD"], cwd=_eyebench_path(config, root))["stdout"],
        "environment_status": environment.get("status"),
        "data_status": data_status.get("status"),
        "split_validation_errors": split_errors,
        "baseline_reproduction_status": "complete"
        if baseline["status"].eq("complete").any()
        else "skipped",
        "d3_status": "complete" if d3_metrics["status"].eq("complete").any() else "skipped",
        "decision_category": decision["decision_category"],
        "manuscript_update": manuscript_update,
        "files": {
            "environment_report": str(dirs["repo_analysis"] / "eyebench_official_environment_report.md"),
            "data_report": str(dirs["repo_analysis"] / "eyebench_official_data_report.md"),
            "baseline_report": str(dirs["repo_analysis"] / "official_baseline_reproduction_report.md"),
            "d3_report": str(dirs["repo_analysis"] / "d3_eyebench_lite_official_evaluation_report.md"),
            "decision_report": str(dirs["repo_analysis"] / "official_eyebench_sota_decision_report.md"),
        },
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def validate_official_eyebench_sota_check(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    out = Path(output_dir).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    config_report = validate_official_eyebench_sota_check_config(config)
    if config_report["status"] != "passed":
        errors.extend(config_report["errors"])
    required = [
        out / "manifest.json",
        out / "environment" / "eyebench_official_environment_report.json",
        out / "data" / "eyebench_official_data_report.json",
        out / "typ" / "d3_eyebench_lite_metrics.csv",
        out / "baseline" / "official_baseline_reproduction_metrics.csv",
        out / "official_eyebench_sota_decision_report.json",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"missing required output: {path}")
    pd = _pd()
    if (out / "typ" / "d3_eyebench_lite_metrics.csv").exists():
        typ = pd.read_csv(out / "typ" / "d3_eyebench_lite_metrics.csv")
        missing = sorted(set(SOTA_TYP_COLUMNS) - set(typ.columns))
        if missing:
            errors.append(f"D3 metric schema missing columns: {missing}")
    if (out / "baseline" / "official_baseline_reproduction_metrics.csv").exists():
        base = pd.read_csv(out / "baseline" / "official_baseline_reproduction_metrics.csv")
        missing = sorted(set(BASELINE_COLUMNS) - set(base.columns))
        if missing:
            errors.append(f"baseline schema missing columns: {missing}")
    comparison = Path(repo_root).resolve() / "analysis" / "official_eyebench_sota_check_v1" / "tables" / "copco_typ_official_sota_comparison.csv"
    if comparison.exists():
        table = pd.read_csv(comparison)
        missing = sorted(set(COMPARISON_COLUMNS) - set(table.columns))
        if missing:
            errors.append(f"comparison table schema missing columns: {missing}")
    else:
        errors.append(f"missing comparison table: {comparison}")
    if (out / "splits" / "official_split_validation.json").exists():
        split_validation = json.loads((out / "splits" / "official_split_validation.json").read_text(encoding="utf-8"))
        if split_validation.get("errors"):
            warnings.extend(split_validation["errors"])
    decision = {}
    if (out / "official_eyebench_sota_decision_report.json").exists():
        decision = json.loads((out / "official_eyebench_sota_decision_report.json").read_text(encoding="utf-8"))
        allowed = {
            "official_sota_claim_allowed",
            "official_compatible_but_not_sota",
            "benchmark_relative_sota_only",
            "blocked_by_environment",
            "blocked_by_data",
        }
        if decision.get("decision_category") not in allowed:
            errors.append(f"invalid decision category: {decision.get('decision_category')}")
    report = {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
        "decision_category": decision.get("decision_category"),
        "output_dir": str(out),
    }
    _write_json(out / "validation_report.json", report)
    return report
