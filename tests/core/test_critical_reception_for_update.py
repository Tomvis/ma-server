"""
Unit tests for `critical_reception_for_update`.

The CR a provider reports comes from the files' review tags, which music-rater
rewrites whenever it corrects a match: it is the truth, so it replaces the stored CR
outright, losses included (MUSIC-80). A provider that reports none (a transient probe
failure, a stub, a streaming provider) leaves the stored CR alone, and so does an
empty CR: a split album's untagged half would otherwise erase its tagged half's CR on
every sync. An absent amg_dr is "not reported" (the tag left the files), never a loss.
"""

from __future__ import annotations

from music_assistant_models.media_items.metadata import (
    CriticalReception,
    ReviewLink,
    ReviewSourceEntry,
)

from music_assistant.helpers.critical_reception import critical_reception_for_update


def _cr(
    amg_dr: float | None = None, sources: list[ReviewSourceEntry] | None = None
) -> CriticalReception:
    return CriticalReception(amg_dr=amg_dr, sources=sources)


def _amg(**fields: object) -> ReviewSourceEntry:
    return ReviewSourceEntry(source="AMG", **fields)


def test_unreported_cr_keeps_stored() -> None:
    """No CR reported keeps the stored one."""
    stored = _cr(sources=[_amg(rating=4.5)])
    assert critical_reception_for_update(None, stored) is stored


def test_empty_cr_keeps_stored() -> None:
    """An empty CR (an untagged split-album half) is not a report."""
    stored = _cr(sources=[_amg(rating=4.5)])
    assert critical_reception_for_update(_cr(), stored) is stored
    assert critical_reception_for_update(_cr(sources=[]), stored) is stored


def test_nothing_stored_takes_new() -> None:
    """With nothing stored, the reported CR is taken."""
    new = _cr(sources=[_amg(rating=4.5)])
    assert critical_reception_for_update(new, None) == new


def test_corrected_accolades_replace_stored_despite_net_loss() -> None:
    """A dropped mismatched accolade plus gained review text lands as reported."""
    stored = _cr(
        sources=[
            _amg(
                rating=4.5,
                accolades=["Album of the Year (2013)", "Review"],
                links=[
                    ReviewLink(label="Album of the Year (2013)", url="https://a/aoty"),
                    ReviewLink(label="Review", url="https://a/wrong-review"),
                ],
            )
        ]
    )
    new = _cr(
        sources=[
            _amg(
                rating=4.5,
                accolades=["Album of the Year (2013)"],
                links=[ReviewLink(label="Album of the Year (2013)", url="https://a/aoty")],
                review="Text.",
            )
        ]
    )
    assert critical_reception_for_update(new, stored) == new


def test_unmatched_review_drops_rating_and_text() -> None:
    """A review music-rater unmatched takes its rating and text with it."""
    stored = _cr(sources=[_amg(rating=4.0, accolades=["Review", "YMIO"], review="Text.")])
    new = _cr(sources=[_amg(accolades=["YMIO"], favorite=True)])
    assert critical_reception_for_update(new, stored) == new


def test_unreported_source_is_dropped() -> None:
    """A source the files no longer carry is dropped."""
    stored = _cr(sources=[_amg(rating=4.5), ReviewSourceEntry(source="TPS", rating=8.0)])
    new = _cr(sources=[ReviewSourceEntry(source="TPS", rating=8.5)])
    assert critical_reception_for_update(new, stored) == new


def test_absent_amg_dr_keeps_stored_amg_dr() -> None:
    """An absent amg_dr keeps the stored one."""
    stored = _cr(amg_dr=7.0, sources=[_amg(rating=4.0)])
    new = _cr(sources=[_amg(rating=4.0, review="Text.")])
    assert critical_reception_for_update(new, stored) == _cr(
        amg_dr=7.0, sources=[_amg(rating=4.0, review="Text.")]
    )


def test_reported_amg_dr_wins() -> None:
    """A reported amg_dr replaces the stored one."""
    stored = _cr(amg_dr=7.0, sources=[_amg(rating=4.0)])
    new = _cr(amg_dr=9.0, sources=[_amg(rating=4.0)])
    assert critical_reception_for_update(new, stored) == new


def test_unchanged_report_equals_stored() -> None:
    """Sync relies on this to skip the write once the stored CR caught up."""
    stored = _cr(amg_dr=7.0, sources=[_amg(rating=4.0, review="Text.")])
    new = _cr(sources=[_amg(rating=4.0, review="Text.")])
    assert critical_reception_for_update(new, stored) == stored


def test_does_not_mutate_new() -> None:
    """Carrying amg_dr over copies instead of mutating the report."""
    stored = _cr(amg_dr=7.0, sources=[_amg(rating=4.0)])
    new = _cr(sources=[_amg(rating=4.0)])
    critical_reception_for_update(new, stored)
    assert new.amg_dr is None
