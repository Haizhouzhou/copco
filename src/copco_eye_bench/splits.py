"""Cross-validation split construction and leakage checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LeakageIssue:
    group_column: str
    overlapping_values: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "group_column": self.group_column,
            "overlap_count": len(self.overlapping_values),
            "overlapping_values": list(self.overlapping_values[:20]),
        }


def assert_no_group_leakage(train: object, test: object, group_columns: Iterable[str]) -> None:
    """Raise when a train/test split shares values in forbidden group columns."""

    issues: list[LeakageIssue] = []
    for column in group_columns:
        train_values = set(train[column].dropna().astype(str))
        test_values = set(test[column].dropna().astype(str))
        overlap = tuple(sorted(train_values.intersection(test_values)))
        if overlap:
            issues.append(LeakageIssue(column, overlap))
    if issues:
        details = "; ".join(f"{issue.group_column}={len(issue.overlapping_values)}" for issue in issues)
        raise ValueError(f"group leakage detected: {details}")


def _round_robin_stratified_groups(groups: list[str], labels: list[int], n_splits: int) -> list[list[str]]:
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    by_label: dict[int, list[str]] = {}
    for group, label in sorted(zip(groups, labels, strict=True)):
        by_label.setdefault(int(label), []).append(group)
    for label_groups in by_label.values():
        for index, group in enumerate(label_groups):
            folds[index % n_splits].append(group)
    return folds


def participant_grouped_folds(participants: object, *, n_splits: int, seed: int) -> object:
    """Build deterministic participant-grouped fold assignments."""

    import pandas as pd

    required = {"participant_id", "dyslexia_labeled"}
    missing = sorted(required.difference(participants.columns))
    if missing:
        raise ValueError(f"participant split table missing columns: {missing}")

    frame = participants[["participant_id", "dyslexia_labeled"]].drop_duplicates().copy()
    frame["dyslexia_labeled"] = frame["dyslexia_labeled"].astype(int)
    groups = frame["participant_id"].astype(str).tolist()
    labels = frame["dyslexia_labeled"].astype(int).tolist()
    class_counts = frame["dyslexia_labeled"].value_counts()
    usable_splits = min(n_splits, len(frame), int(class_counts.min()) if len(class_counts) > 1 else 1)
    usable_splits = max(2, usable_splits)

    try:
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=usable_splits, shuffle=True, random_state=seed)
        fold_groups = [[] for _ in range(usable_splits)]
        for fold, (_, test_idx) in enumerate(splitter.split(groups, labels)):
            fold_groups[fold] = [groups[index] for index in test_idx]
    except Exception:
        fold_groups = _round_robin_stratified_groups(groups, labels, usable_splits)

    rows: list[dict[str, object]] = []
    all_groups = set(groups)
    label_by_group = dict(zip(groups, labels, strict=True))
    for fold, test_groups in enumerate(fold_groups):
        test_set = set(test_groups)
        for split, split_groups in (("train", all_groups - test_set), ("test", test_set)):
            for participant_id in sorted(split_groups):
                rows.append(
                    {
                        "cv_regime": f"participant_grouped_{usable_splits}fold",
                        "fold": fold,
                        "split": split,
                        "participant_id": participant_id,
                        "dyslexia_labeled": label_by_group[participant_id],
                    }
                )
    return pd.DataFrame(rows)


def leave_one_participant_out(participants: object) -> object:
    import pandas as pd

    frame = participants[["participant_id", "dyslexia_labeled"]].drop_duplicates().copy()
    rows: list[dict[str, object]] = []
    all_groups = set(frame["participant_id"].astype(str))
    label_by_group = dict(zip(frame["participant_id"].astype(str), frame["dyslexia_labeled"], strict=True))
    for fold, participant_id in enumerate(sorted(all_groups)):
        for split, split_groups in (("train", all_groups - {participant_id}), ("test", {participant_id})):
            for group in sorted(split_groups):
                rows.append(
                    {
                        "cv_regime": "leave_one_participant_out",
                        "fold": fold,
                        "split": split,
                        "participant_id": group,
                        "dyslexia_labeled": int(label_by_group[group]),
                    }
                )
    return pd.DataFrame(rows)


def leave_one_speech_out(words: object) -> object:
    import pandas as pd

    speeches = sorted(set(words["speech_id"].dropna().astype(str)))
    rows: list[dict[str, object]] = []
    all_speeches = set(speeches)
    for fold, speech_id in enumerate(speeches):
        for split, split_speeches in (("train", all_speeches - {speech_id}), ("test", {speech_id})):
            for group in sorted(split_speeches):
                rows.append(
                    {
                        "cv_regime": "leave_one_speech_out",
                        "fold": fold,
                        "split": split,
                        "speech_id": group,
                    }
                )
    return pd.DataFrame(rows)
