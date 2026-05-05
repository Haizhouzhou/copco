from __future__ import annotations

import pytest

from copco_eye_bench.lm_features import (
    WordSpan,
    align_token_offsets_to_word_spans,
    validate_token_offsets_to_word_spans,
)


def test_token_offsets_align_to_word_spans_by_overlap() -> None:
    spans = [WordSpan("w1", 0, 3), WordSpan("w2", 4, 10)]
    offsets = [(0, 0), (0, 2), (2, 3), (4, 7), (7, 10), (11, 12)]
    assert align_token_offsets_to_word_spans(offsets, spans) == [
        None,
        "w1",
        "w1",
        "w2",
        "w2",
        None,
    ]


def test_alignment_validation_accepts_danish_punctuation_and_compounds() -> None:
    text = '"Københavns Universitet" e-mail-løsningen virkede ikke.'
    spans = [
        WordSpan("s1_p1_s1_w1", 0, 11, '"Københavns', "45", "45_p1", "45_p1_s1"),
        WordSpan("s1_p1_s1_w2", 12, 24, 'Universitet"', "45", "45_p1", "45_p1_s1"),
        WordSpan("s1_p1_s1_w3", 25, 41, "e-mail-løsningen", "45", "45_p1", "45_p1_s1"),
        WordSpan("s1_p1_s1_w4", 42, 49, "virkede", "45", "45_p1", "45_p1_s1"),
        WordSpan("s1_p1_s1_w5", 50, 55, "ikke.", "45", "45_p1", "45_p1_s1"),
    ]
    offsets = [
        (0, 0),
        (0, 1),
        (1, 6),
        (6, 11),
        (12, 18),
        (18, 24),
        (25, 32),
        (32, 41),
        (42, 49),
        (50, 54),
        (54, 55),
    ]
    assignments = align_token_offsets_to_word_spans(offsets, spans)
    report = validate_token_offsets_to_word_spans(
        text, offsets, spans, assignments, context_id="45_p1"
    )
    assert report["status"] == "ok"
    assert report["word_subword_counts"]["s1_p1_s1_w3"] == 2


def test_alignment_validation_accepts_apostrophes_and_sentence_boundaries() -> None:
    text = "Hun sagde 'nå'. Så læste medlæseren videre."
    spans = [
        WordSpan("p1_s1_w1", 0, 3, "Hun"),
        WordSpan("p1_s1_w2", 4, 9, "sagde"),
        WordSpan("p1_s1_w3", 10, 15, "'nå'."),
        WordSpan("p1_s2_w1", 16, 18, "Så"),
        WordSpan("p1_s2_w2", 19, 24, "læste"),
        WordSpan("p1_s2_w3", 25, 35, "medlæseren"),
        WordSpan("p1_s2_w4", 36, 43, "videre."),
    ]
    offsets = [
        (0, 3),
        (4, 9),
        (10, 11),
        (11, 13),
        (13, 15),
        (16, 18),
        (19, 24),
        (25, 30),
        (30, 35),
        (36, 43),
    ]
    assignments = align_token_offsets_to_word_spans(offsets, spans)
    report = validate_token_offsets_to_word_spans(
        text, offsets, spans, assignments, context_id="paragraph_with_two_sentences"
    )
    assert report["status"] == "ok"
    assert report["word_count"] == 7


def test_alignment_validation_fails_for_zero_subword_word() -> None:
    text = "Hej verden"
    spans = [WordSpan("w1", 0, 3, "Hej"), WordSpan("w2", 4, 10, "verden")]
    offsets = [(0, 3)]
    report = validate_token_offsets_to_word_spans(text, offsets, spans, context_id="p1")
    assert report["status"] == "error"
    assert any(error.startswith("zero_subword_words") for error in report["errors"])


def test_alignment_validation_fails_for_reconstructed_text_mismatch() -> None:
    text = "Hej verden"
    spans = [WordSpan("w1", 0, 3, "Hej"), WordSpan("w2", 4, 10, "verden!")]
    offsets = [(0, 3), (4, 10)]
    report = validate_token_offsets_to_word_spans(text, offsets, spans, context_id="p1")
    assert report["status"] == "error"
    assert "reconstructed_text_mismatch" in report["errors"]


@pytest.mark.parametrize(
    "spans, expected",
    [
        ([WordSpan("", 0, 3, "Hej")], "missing_stable_word_ids"),
        ([WordSpan("w1", 0, 3, "Hej"), WordSpan("w1", 4, 10, "verden")], "duplicate_stable_word_ids"),
    ],
)
def test_alignment_validation_fails_for_bad_stable_ids(
    spans: list[WordSpan], expected: str
) -> None:
    text = "Hej verden"
    offsets = [(0, 3), (4, 10)]
    report = validate_token_offsets_to_word_spans(text, offsets, spans, context_id="p1")
    assert report["status"] == "error"
    assert any(error.startswith(expected) for error in report["errors"])
