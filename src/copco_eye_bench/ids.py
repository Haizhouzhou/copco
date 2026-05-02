"""Stable identifier construction for CopCo tables."""

from __future__ import annotations

import math
import re
from typing import Any


EXCLUDED_PARTICIPANTS = frozenset({"P14"})
PRACTICE_SPEECH_ID = "1327"


def clean_scalar(value: Any) -> str:
    """Convert source values to stable string tokens without pandas dtype artifacts."""

    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip().strip('"')
    if text.endswith(".0") and re.fullmatch(r"-?\d+\.0", text):
        text = text[:-2]
    return text


def normalize_participant_id(value: Any) -> str:
    """Normalize participant identifiers while preserving the CopCo ``P01`` style."""

    text = clean_scalar(value)
    if not text:
        return text
    upper = text.upper()
    match = re.fullmatch(r"P?0*(\d+)", upper)
    if match:
        return f"P{int(match.group(1)):02d}"
    return upper


def speech_id_from_source(speech_id: Any) -> str:
    return clean_scalar(speech_id)


def paragraph_id_from_source(speech_id: Any, paragraph_id: Any) -> str:
    sid = speech_id_from_source(speech_id)
    return f"{sid}_p{clean_scalar(paragraph_id)}"


def sentence_id_from_source(speech_id: Any, paragraph_id: Any, sentence_id: Any) -> str:
    pid = paragraph_id_from_source(speech_id, paragraph_id)
    return f"{pid}_s{clean_scalar(sentence_id)}"


def word_id_from_source(speech_id: Any, paragraph_id: Any, sentence_id: Any, word_id: Any) -> str:
    sid = sentence_id_from_source(speech_id, paragraph_id, sentence_id)
    return f"{sid}_w{clean_scalar(word_id)}"


def add_stable_ids(frame: Any) -> Any:
    """Return a copy of a pandas frame with stable CopCo ID columns added."""

    required = {"speechId", "paragraphId", "sentenceId", "wordId"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"cannot construct stable IDs; missing columns: {missing}")

    out = frame.copy()
    participant_source = "participant_id" if "participant_id" in out.columns else "part"
    if participant_source in out.columns:
        out["participant_id"] = out[participant_source].map(normalize_participant_id)
    out["speech_id"] = out["speechId"].map(speech_id_from_source)
    out["paragraph_id"] = [
        paragraph_id_from_source(speech, paragraph)
        for speech, paragraph in zip(out["speechId"], out["paragraphId"], strict=True)
    ]
    out["sentence_id"] = [
        sentence_id_from_source(speech, paragraph, sentence)
        for speech, paragraph, sentence in zip(
            out["speechId"], out["paragraphId"], out["sentenceId"], strict=True
        )
    ]
    out["word_id"] = [
        word_id_from_source(speech, paragraph, sentence, word)
        for speech, paragraph, sentence, word in zip(
            out["speechId"], out["paragraphId"], out["sentenceId"], out["wordId"], strict=True
        )
    ]
    return out
