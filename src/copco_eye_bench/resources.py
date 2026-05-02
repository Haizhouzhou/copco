"""Public-resource detection and deterministic fallbacks."""

from __future__ import annotations

import importlib.util
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


VOWEL_RE = re.compile(r"[aeiouyæøåAEIOUYÆØÅ]+")


@dataclass(frozen=True)
class ResourceStatus:
    name: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "available": self.available, "detail": self.detail}


def module_status(module: str, *, label: str | None = None) -> ResourceStatus:
    spec = importlib.util.find_spec(module)
    if spec is None:
        return ResourceStatus(label or module, False, "python_module_missing")
    return ResourceStatus(label or module, True, spec.origin or "namespace_package")


def file_status(path: str | Path, *, label: str) -> ResourceStatus:
    candidate = Path(path)
    if candidate.exists():
        return ResourceStatus(label, True, str(candidate))
    return ResourceStatus(label, False, f"missing_file:{candidate}")


def public_resource_statuses(extra_paths: Iterable[tuple[str, str | Path]] = ()) -> list[ResourceStatus]:
    statuses = [
        module_status("pyphen", label="pyphen_da_DK"),
        module_status("dacy", label="dacy"),
        module_status("spacy", label="spacy"),
        module_status("torch", label="torch"),
        module_status("transformers", label="transformers"),
        module_status("sentence_transformers", label="sentence_transformers"),
        module_status("sklearn", label="scikit_learn"),
        module_status("statsmodels", label="statsmodels"),
        module_status("lightgbm", label="lightgbm"),
        module_status("xgboost", label="xgboost"),
    ]
    statuses.extend(file_status(path, label=label) for label, path in extra_paths)
    return statuses


def normalize_word(word: str) -> str:
    return unicodedata.normalize("NFC", str(word or ""))


def vowel_cluster_syllable_count(word: str) -> int:
    normalized = normalize_word(word)
    clusters = VOWEL_RE.findall(normalized)
    return max(1, len(clusters)) if normalized else 0


def syllable_count(word: str) -> tuple[int, str]:
    """Return Danish syllable count with Pyphen when available, else a flagged fallback."""

    normalized = normalize_word(word)
    if not normalized:
        return 0, "empty"
    try:
        import pyphen

        dictionary = pyphen.Pyphen(lang="da_DK")
        hyphenated = dictionary.inserted(normalized)
        return max(1, hyphenated.count("-") + 1), "pyphen_da_DK"
    except Exception:
        return vowel_cluster_syllable_count(normalized), "vowel_cluster_fallback"
