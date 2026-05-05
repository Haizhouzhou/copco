"""Feature-table construction for the CopCo dyslexia-labeled reader program."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .config import get_nested
from .ids import (
    EXCLUDED_PARTICIPANTS,
    PRACTICE_SPEECH_ID,
    add_stable_ids,
    clean_scalar,
    normalize_participant_id,
)
from .resources import normalize_word, public_resource_statuses, syllable_count
from .splits import leave_one_participant_out, leave_one_speech_out, participant_grouped_folds


GAZE_METRIC_MAP = {
    "FFD": "word_first_fix_dur",
    "GD": "word_first_pass_dur",
    "TRT": "word_total_fix_dur",
    "fixation_count": "number_of_fixations",
}

IA_CROSSCHECK_MAP = {
    "FFD": "IA_FIRST_FIXATION_DURATION",
    "GD": "IA_FIRST_RUN_DWELL_TIME",
    "TRT": "IA_DWELL_TIME",
    "fixation_count": "IA_FIXATION_COUNT",
    "skip": "IA_SKIP",
    "regression_in": "IA_REGRESSION_IN",
    "regression_out": "IA_REGRESSION_OUT",
}


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "pandas is required for feature construction. Install runtime dependencies "
            "inside the copco environment."
        ) from exc
    return pd


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _legacy_root(config: dict[str, Any], repo_root: Path) -> Path:
    root = Path(str(get_nested(config, "dataset.legacy_root", "copco-processing")))
    return root if root.is_absolute() else repo_root / root


def _read_extracted_features(
    config: dict[str, Any], repo_root: Path, sample_participants: int | None = None
) -> Any:
    pd = _require_pandas()
    root = _legacy_root(config, repo_root)
    pattern = str(get_nested(config, "dataset.extracted_features_glob", "ExtractedFeatures/P*.csv"))
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no extracted feature CSV files found under {root / pattern}")
    files, sample_report = _select_extracted_feature_files(config, repo_root, files, sample_participants)

    frames = []
    for path in files:
        frame = pd.read_csv(path, dtype={"part": "string"})
        frame["participant_id"] = normalize_participant_id(path.stem)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out.attrs["sample_report"] = sample_report
    return out


def _read_participants(config: dict[str, Any], repo_root: Path) -> Any:
    pd = _require_pandas()
    root = _legacy_root(config, repo_root)
    relative = Path(str(get_nested(config, "dataset.participant_stats_path", "participant_stats.csv")))
    path = relative if relative.is_absolute() else root / relative
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"subj": "string"})


def _parse_label(value: Any) -> int | None:
    text = clean_scalar(value).lower()
    if text in {"1", "true", "yes", "y", "dyslexia", "dyslexic", "dyslexia_labeled"}:
        return 1
    if text in {"0", "false", "no", "n", "control", "typical", "non-dyslexic"}:
        return 0
    return None


def _select_extracted_feature_files(
    config: dict[str, Any],
    repo_root: Path,
    files: list[Path],
    sample_participants: int | None,
) -> tuple[list[Path], dict[str, Any]]:
    if not sample_participants:
        return files, {"strategy": "all_available", "selected_participants": len(files)}

    require_two_classes = bool(get_nested(config, "sample.require_two_classes", False))
    min_per_class = int(get_nested(config, "sample.min_participants_per_class", 1))
    seed = int(get_nested(config, "sample.random_seed", get_nested(config, "cv.random_seeds", [17])[0]))
    report: dict[str, Any] = {
        "strategy": "first_n",
        "requested_participants": int(sample_participants),
        "require_two_classes": require_two_classes,
        "min_participants_per_class": min_per_class,
        "random_seed": seed,
    }
    if not require_two_classes:
        selected = files[:sample_participants]
        report["selected_participants"] = [normalize_participant_id(path.stem) for path in selected]
        return selected, report

    if sample_participants < min_per_class * 2:
        raise ValueError(
            "class-aware smoke sampling needs sample.participants >= "
            "2 * sample.min_participants_per_class"
        )

    raw_participants = _read_participants(config, repo_root)
    if raw_participants.empty:
        raise ValueError("class-aware smoke sampling requires participant metadata with labels")
    label_column = None
    if "dyslexia" in raw_participants.columns:
        label_column = "dyslexia"
    elif "dyslexia_labeled" in raw_participants.columns:
        label_column = "dyslexia_labeled"
    if label_column is None:
        raise ValueError("class-aware smoke sampling requires a dyslexia or dyslexia_labeled column")

    source_col = "subj" if "subj" in raw_participants.columns else "participant_id"
    labels = raw_participants[[source_col, label_column]].copy()
    labels["participant_id"] = labels[source_col].map(normalize_participant_id)
    labels["dyslexia_labeled"] = labels[label_column].map(_parse_label)
    labels = labels.dropna(subset=["participant_id", "dyslexia_labeled"]).copy()
    labels["dyslexia_labeled"] = labels["dyslexia_labeled"].astype(int)

    file_by_participant = {normalize_participant_id(path.stem): path for path in files}
    available = labels[labels["participant_id"].isin(file_by_participant)].drop_duplicates(
        "participant_id"
    )
    excluded = {
        normalize_participant_id(value)
        for value in get_nested(config, "dataset.excluded_participants", ["P14"])
    }
    available = available[~available["participant_id"].isin(excluded)].copy()
    class_counts = available["dyslexia_labeled"].value_counts().to_dict()
    report["available_label_counts"] = {
        "typical": int(class_counts.get(0, 0)),
        "dyslexia_labeled": int(class_counts.get(1, 0)),
    }

    if len(class_counts) < 2:
        selected = files[:sample_participants]
        report["strategy"] = "first_n_single_class_available"
        report["selected_participants"] = [normalize_participant_id(path.stem) for path in selected]
        report["warning"] = "only one class present in available participant metadata"
        return selected, report

    selected_ids: list[str] = []
    for label in (1, 0):
        class_ids = sorted(
            available[available["dyslexia_labeled"] == label]["participant_id"].astype(str).tolist()
        )
        if len(class_ids) < min_per_class:
            raise ValueError(f"class-aware smoke sampling lacks {min_per_class} participants for class {label}")
        selected_ids.extend(class_ids[:min_per_class])

    all_available_ids = sorted(available["participant_id"].astype(str).tolist())
    for participant_id in all_available_ids:
        if len(selected_ids) >= sample_participants:
            break
        if participant_id not in selected_ids:
            selected_ids.append(participant_id)

    selected_set = set(selected_ids)
    selected = [path for path in files if normalize_participant_id(path.stem) in selected_set]
    selected = selected[:sample_participants]
    report["strategy"] = "class_aware"
    report["selected_participants"] = [normalize_participant_id(path.stem) for path in selected]
    selected_labels = available[available["participant_id"].isin(report["selected_participants"])]
    selected_counts = selected_labels["dyslexia_labeled"].value_counts().to_dict()
    report["selected_label_counts"] = {
        "typical": int(selected_counts.get(0, 0)),
        "dyslexia_labeled": int(selected_counts.get(1, 0)),
    }
    return selected, report


def _prepare_participants(raw: Any, observations: Any) -> Any:
    pd = _require_pandas()
    if raw.empty:
        participants = pd.DataFrame({"participant_id": sorted(observations["participant_id"].unique())})
        participants["dyslexia_labeled"] = pd.NA
        participants["group_label"] = "missing_label"
        participants["label_provenance"] = "missing_participant_stats"
        participants["label_provenance_strength"] = "missing"
        return participants

    participants = raw.copy()
    source_col = "subj" if "subj" in participants.columns else "participant_id"
    participants["participant_id"] = participants[source_col].map(normalize_participant_id)
    if "dyslexia" in participants.columns:
        participants["dyslexia_labeled"] = participants["dyslexia"].map(_parse_label)
    elif "dyslexia_labeled" not in participants.columns:
        participants["dyslexia_labeled"] = pd.NA

    participants["group_label"] = participants["dyslexia_labeled"].map(
        {0: "typical", 1: "dyslexia-labeled reader"}
    )
    participants["group_label"] = participants["group_label"].fillna("missing_label")
    if "label_provenance" not in participants.columns:
        participants["label_provenance"] = "operational_copco_metadata_no_instrument"
    if "label_provenance_strength" not in participants.columns:
        participants["label_provenance_strength"] = "metadata_no_instrument"

    excluded = EXCLUDED_PARTICIPANTS
    participants = participants[~participants["participant_id"].isin(excluded)].copy()
    participant_order = sorted(observations["participant_id"].dropna().unique())
    participants = participants[participants["participant_id"].isin(participant_order)].copy()
    return participants.drop_duplicates("participant_id")


def _filter_observations(frame: Any, config: dict[str, Any]) -> Any:
    excluded_participants = {
        normalize_participant_id(value)
        for value in get_nested(config, "dataset.excluded_participants", ["P14"])
    }
    excluded_speeches = {
        clean_scalar(value) for value in get_nested(config, "dataset.excluded_speech_ids", ["1327"])
    }
    if not excluded_speeches:
        excluded_speeches = {PRACTICE_SPEECH_ID}
    return frame[
        ~frame["participant_id"].isin(excluded_participants)
        & ~frame["speech_id"].astype(str).isin(excluded_speeches)
    ].copy()


def _add_gaze_metrics(frame: Any) -> Any:
    pd = _require_pandas()
    out = frame.copy()
    for target, source in GAZE_METRIC_MAP.items():
        if source in out.columns:
            out[target] = pd.to_numeric(out[source], errors="coerce")
        else:
            out[target] = pd.NA
    out["skip"] = out["fixation_count"].fillna(0).eq(0).astype("int8")
    out["refixation_count"] = out["fixation_count"].fillna(0).sub(1).clip(lower=0)
    if "word_go_past_time" in out.columns:
        out["go_past_time"] = pd.to_numeric(out["word_go_past_time"], errors="coerce")
    for metric in ("regression_in", "regression_out"):
        if metric not in out.columns:
            out[metric] = pd.NA
    return out


def _limit_sample(frame: Any, sample_participants: int | None, sample_speeches: int | None) -> Any:
    out = frame
    if sample_participants:
        keep_participants = sorted(out["participant_id"].dropna().unique())[:sample_participants]
        out = out[out["participant_id"].isin(keep_participants)].copy()
    if sample_speeches:
        speech_counts = (
            out.groupby("speech_id")["participant_id"]
            .nunique()
            .reset_index(name="participant_count")
            .sort_values(["participant_count", "speech_id"], ascending=[False, True])
        )
        keep_speeches = set(speech_counts["speech_id"].astype(str).head(sample_speeches).tolist())
        covered = set(out[out["speech_id"].astype(str).isin(keep_speeches)]["participant_id"].unique())
        for participant_id in sorted(set(out["participant_id"].unique()) - covered):
            participant_speeches = sorted(
                out[out["participant_id"] == participant_id]["speech_id"].astype(str).unique()
            )
            keep_speeches.update(participant_speeches[:sample_speeches])
        out = out[out["speech_id"].astype(str).isin(keep_speeches)].copy()
    return out


def _sort_observations(frame: Any) -> Any:
    pd = _require_pandas()
    out = frame.copy()
    for column in ["speechId", "paragraphId", "sentenceId", "wordId"]:
        out[f"_{column}_sort"] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values(
        [
            "participant_id",
            "_speechId_sort",
            "_paragraphId_sort",
            "_sentenceId_sort",
            "_wordId_sort",
            "word_id",
        ],
        kind="mergesort",
    )


def assert_unique_participant_word(frame: Any) -> None:
    duplicates = frame.duplicated(["participant_id", "word_id"], keep=False)
    if bool(duplicates.any()):
        count = int(duplicates.sum())
        raise ValueError(f"final word table must have one row per participant_id, word_id; {count} dupes")


def _make_words(observations: Any) -> Any:
    pd = _require_pandas()
    first_cols = [
        "speech_id",
        "paragraph_id",
        "sentence_id",
        "word_id",
        "speechId",
        "paragraphId",
        "sentenceId",
        "wordId",
        "word",
    ]
    text = observations[first_cols].drop_duplicates("word_id").copy()
    text = text.sort_values(["speechId", "paragraphId", "sentenceId", "wordId"], kind="mergesort")
    text = text.rename(columns={"word": "word_form"})
    text["word_form"] = text["word_form"].map(normalize_word)
    text["word_length_chars"] = text["word_form"].map(len)
    syllables = text["word_form"].map(syllable_count)
    text["syllable_count"] = [item[0] for item in syllables]
    text["syllable_source"] = [item[1] for item in syllables]

    text["word_index_in_sentence"] = text.groupby("sentence_id", sort=False).cumcount()
    text["word_index_in_paragraph"] = text.groupby("paragraph_id", sort=False).cumcount()
    text["sentence_length_words"] = text.groupby("sentence_id")["word_id"].transform("count")
    text["paragraph_length_words"] = text.groupby("paragraph_id")["word_id"].transform("count")
    text["word_position_in_sentence_norm"] = text["word_index_in_sentence"] / (
        text["sentence_length_words"].sub(1).replace(0, pd.NA)
    )
    text["word_position_in_sentence_norm"] = text["word_position_in_sentence_norm"].fillna(0.0)
    text["word_position_in_paragraph_norm"] = text["word_index_in_paragraph"] / (
        text["paragraph_length_words"].sub(1).replace(0, pd.NA)
    )
    text["word_position_in_paragraph_norm"] = text["word_position_in_paragraph_norm"].fillna(0.0)
    return _add_text_offsets(text)


def _add_text_offsets(words: Any) -> Any:
    out = words.copy()
    for scope in ("sentence", "paragraph"):
        out[f"word_start_in_{scope}"] = 0
        out[f"word_end_in_{scope}"] = 0
        group_column = f"{scope}_id"
        for _, group in out.groupby(group_column, sort=False):
            position = 0
            for index, value in group["word_form"].items():
                word = str(value)
                out.at[index, f"word_start_in_{scope}"] = position
                position += len(word)
                out.at[index, f"word_end_in_{scope}"] = position
                position += 1
    return out


def _make_sentences(words: Any) -> Any:
    return (
        words.sort_values(["speechId", "paragraphId", "sentenceId", "wordId"], kind="mergesort")
        .groupby("sentence_id", sort=False)
        .agg(
            speech_id=("speech_id", "first"),
            paragraph_id=("paragraph_id", "first"),
            speechId=("speechId", "first"),
            paragraphId=("paragraphId", "first"),
            sentenceId=("sentenceId", "first"),
            sentence_text=("word_form", lambda values: " ".join(map(str, values))),
            sentence_length_words=("word_id", "count"),
            mean_word_length_chars=("word_length_chars", "mean"),
        )
        .reset_index()
    )


def _make_paragraphs(words: Any) -> Any:
    return (
        words.sort_values(["speechId", "paragraphId", "sentenceId", "wordId"], kind="mergesort")
        .groupby("paragraph_id", sort=False)
        .agg(
            speech_id=("speech_id", "first"),
            speechId=("speechId", "first"),
            paragraphId=("paragraphId", "first"),
            paragraph_text=("word_form", lambda values: " ".join(map(str, values))),
            paragraph_length_words=("word_id", "count"),
            sentence_count=("sentence_id", "nunique"),
            mean_word_length_chars=("word_length_chars", "mean"),
        )
        .reset_index()
    )


def _make_word_observations(observations: Any, words: Any, participants: Any) -> Any:
    columns = [
        "participant_id",
        "speech_id",
        "paragraph_id",
        "sentence_id",
        "word_id",
        "FFD",
        "GD",
        "TRT",
        "fixation_count",
        "skip",
        "refixation_count",
        "go_past_time",
        "regression_in",
        "regression_out",
    ]
    available = [column for column in columns if column in observations.columns]
    final = observations[available].copy()
    text_features = words[
        [
            "word_id",
            "word_form",
            "word_length_chars",
            "syllable_count",
            "syllable_source",
            "word_index_in_sentence",
            "word_index_in_paragraph",
            "word_position_in_sentence_norm",
            "word_position_in_paragraph_norm",
            "sentence_length_words",
            "paragraph_length_words",
        ]
    ]
    final = final.merge(text_features, on="word_id", how="left", validate="many_to_one")
    participant_features = participants[
        [column for column in participants.columns if column not in {"subj"}]
    ].copy()
    final = final.merge(participant_features, on="participant_id", how="left", validate="many_to_one")
    final = _aggregate_duplicate_participant_words(final)
    assert_unique_participant_word(final)
    return final


def _aggregate_duplicate_participant_words(frame: Any) -> Any:
    pd = _require_pandas()
    if not bool(frame.duplicated(["participant_id", "word_id"]).any()):
        out = frame.copy()
        out["source_observation_count"] = 1
        return out

    key = ["participant_id", "word_id"]
    numeric_cols = [
        column
        for column in frame.select_dtypes(include=["number", "bool"]).columns
        if column not in key and column != "dyslexia_labeled"
    ]
    aggregations: dict[str, str] = {}
    for column in frame.columns:
        if column in key:
            continue
        if column == "dyslexia_labeled":
            aggregations[column] = "first"
        elif column in numeric_cols:
            aggregations[column] = "mean"
        else:
            aggregations[column] = "first"
    out = frame.groupby(key, as_index=False, sort=False).agg(aggregations)
    counts = frame.groupby(key, as_index=False, sort=False).size().rename(
        columns={"size": "source_observation_count"}
    )
    out = out.merge(counts, on=key, how="left", validate="one_to_one")
    if "skip" in out.columns:
        out["skip"] = pd.to_numeric(out["skip"], errors="coerce").fillna(0).round().astype("int8")
    return out


def _numeric_ia_series(series: Any) -> Any:
    pd = _require_pandas()
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _ia_crosscheck(config: dict[str, Any], repo_root: Path, observations: Any) -> dict[str, Any]:
    pd = _require_pandas()
    root = _legacy_root(config, repo_root)
    pattern = str(get_nested(config, "dataset.ia_reports_glob", "InterestAreaReports/IA_report_P*.txt"))
    files = sorted(root.glob(pattern))
    files_available = len(files)
    participant_ids = set(observations["participant_id"].dropna().astype(str))
    if participant_ids:
        files = [
            path
            for path in files
            if normalize_participant_id(path.stem.replace("IA_report_", "")) in participant_ids
        ]
    report: dict[str, Any] = {
        "available": bool(files),
        "files_available": files_available,
        "files_seen": len(files),
        "matched_rows": 0,
        "duplicate_ia_keys": 0,
        "duplicate_observation_keys": 0,
        "metrics": {},
    }
    if not files:
        report["reason"] = "ia_reports_missing"
        return report

    frames = []
    usecols = [
        "RECORDING_SESSION_LABEL",
        "speechid",
        "paragraphid",
        "IA_ID",
        *sorted(set(IA_CROSSCHECK_MAP.values())),
    ]
    for path in files:
        header = pd.read_csv(path, sep="\t", nrows=0, encoding="utf-8-sig")
        cols = [column for column in usecols if column in header.columns]
        frames.append(pd.read_csv(path, sep="\t", usecols=cols, encoding="utf-8-sig"))
    ia = pd.concat(frames, ignore_index=True)
    ia["participant_id"] = ia["RECORDING_SESSION_LABEL"].map(normalize_participant_id)
    ia["speech_id"] = ia["speechid"].map(clean_scalar)
    ia["source_paragraph_id"] = ia["paragraphid"].map(clean_scalar)
    ia["source_word_id"] = ia["IA_ID"].map(clean_scalar)

    obs = observations.copy()
    obs["source_paragraph_id"] = obs["paragraphId"].map(clean_scalar)
    obs["source_word_id"] = obs["wordId"].map(clean_scalar)
    keys = ["participant_id", "speech_id", "source_paragraph_id", "source_word_id"]
    report["duplicate_ia_keys"] = int(ia.duplicated(keys, keep=False).sum())
    report["duplicate_observation_keys"] = int(obs.duplicated(keys, keep=False).sum())
    ia_unique = ia[~ia.duplicated(keys, keep=False)].copy()
    obs_unique = obs[~obs.duplicated(keys, keep=False)].copy()
    merged = obs_unique.merge(ia_unique, on=keys, how="inner", suffixes=("", "_ia"))
    report["matched_rows"] = int(len(merged))
    if merged.empty:
        report["reason"] = "no_unique_key_matches"
        return report

    for metric, ia_column in IA_CROSSCHECK_MAP.items():
        if metric not in merged.columns or ia_column not in merged.columns:
            continue
        left = pd.to_numeric(merged[metric], errors="coerce")
        right = _numeric_ia_series(merged[ia_column])
        valid = left.notna() & right.notna()
        diff = (left[valid] - right[valid]).abs()
        report["metrics"][metric] = {
            "compared_rows": int(valid.sum()),
            "exact_or_tolerance_matches": int((diff <= 1e-6).sum()),
            "max_abs_diff": None if diff.empty else float(diff.max()),
        }
    return report


def _write_split_tables(
    output_dir: Path, config: dict[str, Any], participants: Any, words: Any
) -> dict[str, Any]:
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    label_ready = (
        "dyslexia_labeled" in participants.columns
        and not participants["dyslexia_labeled"].isna().any()
        and participants["dyslexia_labeled"].nunique() == 2
    )
    if not label_ready:
        report["skipped"] = "participant labels missing or single-class"
        return report

    folds = int(get_nested(config, "cv.participant_grouped_folds", 5))
    seed = int(get_nested(config, "cv.random_seeds", [17])[0])
    participant_folds = participant_grouped_folds(participants, n_splits=folds, seed=seed)
    participant_folds.to_csv(split_dir / "participant_grouped_folds.csv", index=False)
    lopo = leave_one_participant_out(participants)
    lopo.to_csv(split_dir / "leave_one_participant_out.csv", index=False)
    loso = leave_one_speech_out(words)
    loso.to_csv(split_dir / "leave_one_speech_out.csv", index=False)

    p32 = str(get_nested(config, "dataset.p32_sensitivity_participant", "P32"))
    participants[participants["participant_id"] != p32].to_csv(
        split_dir / "participants_excluding_p32.csv", index=False
    )
    weak = participants["label_provenance_strength"].astype(str).str.contains(
        "weak|self", case=False, na=False, regex=True
    )
    participants[~weak].to_csv(split_dir / "participants_strong_or_unspecified_provenance.csv", index=False)
    report.update(
        {
            "participant_grouped_rows": int(len(participant_folds)),
            "leave_one_participant_rows": int(len(lopo)),
            "leave_one_speech_rows": int(len(loso)),
            "p32_excluded_participants": int((participants["participant_id"] != p32).sum()),
            "weak_provenance_excluded": int(weak.sum()),
        }
    )
    return report


def build_feature_tables(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    repo_root: str | Path = ".",
    sample_participants: int | None = None,
    sample_speeches: int | None = None,
) -> dict[str, Any]:
    """Build the reproducible CopCo feature-table layer under ``output_dir``."""

    pd = _require_pandas()
    root = Path(repo_root).resolve()
    out = Path(output_dir).resolve()
    table_dir = out / "tables"
    report_dir = out / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    missing_inputs: list[str] = []
    derived57_module = str(get_nested(config, "dataset.derived57_module", "derived57"))
    try:
        __import__(derived57_module)
        source = derived57_module
        missing_inputs.append("derived57_imported_but_no_generic_reader_used; legacy schema reader used")
    except Exception:
        source = "legacy_copco_processing"
        missing_inputs.append(f"{derived57_module}:python_module_missing")

    raw_observations = _read_extracted_features(config, root, sample_participants)
    sample_report = dict(raw_observations.attrs.get("sample_report", {}))
    observations = add_stable_ids(raw_observations)
    observations = _filter_observations(observations, config)
    observations = _add_gaze_metrics(observations)
    observations = _limit_sample(observations, sample_participants, sample_speeches)
    observations = _sort_observations(observations)

    participants = _prepare_participants(_read_participants(config, root), observations)
    if sample_participants:
        participants = participants[participants["participant_id"].isin(observations["participant_id"])]
    words = _make_words(observations)
    sentences = _make_sentences(words)
    paragraphs = _make_paragraphs(words)
    word_observations = _make_word_observations(observations, words, participants)
    duplicate_source_rows = int(word_observations["source_observation_count"].sub(1).clip(lower=0).sum())
    duplicate_source_groups = int(word_observations["source_observation_count"].gt(1).sum())

    words.to_parquet(table_dir / "words.parquet", index=False)
    sentences.to_parquet(table_dir / "sentences.parquet", index=False)
    paragraphs.to_parquet(table_dir / "paragraphs.parquet", index=False)
    participants.to_parquet(table_dir / "participants.parquet", index=False)
    word_observations.to_parquet(table_dir / "word_observations.parquet", index=False)

    ia_report = _ia_crosscheck(config, root, observations)
    _write_json(report_dir / "ia_cross_checks.json", ia_report)
    split_report = _write_split_tables(out, config, participants, words)

    label_counts: dict[str, int] = {}
    if "dyslexia_labeled" in participants.columns:
        counts = participants["dyslexia_labeled"].dropna().astype(int).value_counts()
        label_counts = {
            "typical": int(counts.get(0, 0)),
            "dyslexia_labeled": int(counts.get(1, 0)),
        }
    expected = get_nested(config, "dataset.expected_participants", {})
    label_count_warning = None
    if expected and label_counts:
        expected_counts = {
            "typical": int(expected.get("typical", -1)),
            "dyslexia_labeled": int(expected.get("dyslexia_labeled", -1)),
        }
        if label_counts != expected_counts:
            label_count_warning = {"expected": expected_counts, "observed": label_counts}

    resource_report = [status.as_dict() for status in public_resource_statuses()]
    manifest = {
        "run_type": "build_features",
        "source": source,
        "output_dir": str(out),
        "git_sha": _git_sha(root),
        "config_sha256": _config_hash(config),
        "sample": {"participants": sample_participants, "speeches": sample_speeches},
        "sample_strategy": sample_report,
        "missing_inputs": missing_inputs,
        "row_counts": {
            "raw_observations_loaded": int(len(raw_observations)),
            "word_observations": int(len(word_observations)),
            "words": int(len(words)),
            "sentences": int(len(sentences)),
            "paragraphs": int(len(paragraphs)),
            "participants": int(len(participants)),
        },
        "source_duplicate_observations_aggregated": {
            "extra_rows_collapsed": duplicate_source_rows,
            "participant_word_groups_with_duplicates": duplicate_source_groups,
        },
        "label_counts": label_counts,
        "label_count_warning": label_count_warning,
        "ia_cross_checks": ia_report,
        "split_tables": split_report,
        "public_resources": resource_report,
        "claim_language": "dyslexia-labeled reader; not clinical diagnosis or screening",
    }
    _write_json(out / "manifest.json", manifest)

    summary = pd.DataFrame(
        [
            {"table": "word_observations", "rows": len(word_observations)},
            {"table": "words", "rows": len(words)},
            {"table": "sentences", "rows": len(sentences)},
            {"table": "paragraphs", "rows": len(paragraphs)},
            {"table": "participants", "rows": len(participants)},
        ]
    )
    summary.to_csv(report_dir / "table_summary.csv", index=False)
    return manifest
