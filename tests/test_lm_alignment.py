from __future__ import annotations

from copco_eye_bench.lm_features import WordSpan, align_token_offsets_to_word_spans


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
