"""Unit tests for `_critical_reception_is_richer`.

The function gates wholesale replacement of an album's stored critical_reception
during sync and listen-later updates. Replacement is whole-CR, so the function
must reject any change that loses data on either dimension (amg_dr or sources)
— otherwise that field would silently disappear.
"""

from __future__ import annotations

from music_assistant_models.media_items.metadata import CriticalReception, ReviewSourceEntry

from music_assistant.models.music_provider import _critical_reception_is_richer


def _cr(
    amg_dr: float | None = None, sources: list[ReviewSourceEntry] | None = None
) -> CriticalReception:
    return CriticalReception(amg_dr=amg_dr, sources=sources)


def test_returns_false_when_new_is_none() -> None:
    """`new is None` never replaces an existing CR (would erase data)."""
    existing = _cr(amg_dr=12.5)
    assert _critical_reception_is_richer(None, existing) is False


def test_returns_true_when_existing_is_none() -> None:
    """Any populated CR is richer than nothing."""
    new = _cr(amg_dr=12.5)
    assert _critical_reception_is_richer(new, None) is True


def test_regresses_when_new_drops_amg_dr() -> None:
    """New gains a source but loses amg_dr — wholesale replace would erase 12.5."""
    existing = _cr(amg_dr=12.5, sources=[ReviewSourceEntry(source="AMG", rating=4.5)])
    new = _cr(
        amg_dr=None,
        sources=[
            ReviewSourceEntry(source="AMG", rating=4.5),
            ReviewSourceEntry(source="TPS", rating=8.5),
        ],
    )
    assert _critical_reception_is_richer(new, existing) is False


def test_regresses_when_new_drops_a_source() -> None:
    """New gains amg_dr but loses a source — wholesale replace would erase TPS."""
    existing = _cr(
        amg_dr=None,
        sources=[
            ReviewSourceEntry(source="AMG", rating=4.5),
            ReviewSourceEntry(source="TPS", rating=8.5),
        ],
    )
    new = _cr(amg_dr=12.5, sources=[ReviewSourceEntry(source="AMG", rating=4.5)])
    assert _critical_reception_is_richer(new, existing) is False


def test_amg_dr_added_with_same_sources_is_richer() -> None:
    """A fresh amg_dr on an otherwise-identical CR is a strict upgrade."""
    existing = _cr(amg_dr=None, sources=[ReviewSourceEntry(source="AMG", rating=4.5)])
    new = _cr(amg_dr=12.5, sources=[ReviewSourceEntry(source="AMG", rating=4.5)])
    assert _critical_reception_is_richer(new, existing) is True


def test_extra_source_added_is_richer() -> None:
    """Adding a brand-new source to an otherwise-equal CR is a strict upgrade."""
    existing = _cr(amg_dr=12.5, sources=[ReviewSourceEntry(source="AMG", rating=4.5)])
    new = _cr(
        amg_dr=12.5,
        sources=[
            ReviewSourceEntry(source="AMG", rating=4.5),
            ReviewSourceEntry(source="TPS", rating=8.5),
        ],
    )
    assert _critical_reception_is_richer(new, existing) is True


def test_same_source_gains_field_is_richer() -> None:
    """A refresh that fills in a previously-blank field counts as richer."""
    existing = _cr(sources=[ReviewSourceEntry(source="AMG", rating=4.5)])
    new = _cr(
        sources=[
            ReviewSourceEntry(source="AMG", rating=4.5, accolades=["Album of the Year (2024)"])
        ]
    )
    assert _critical_reception_is_richer(new, existing) is True


def test_identical_cr_is_not_richer() -> None:
    """No-change refresh must not trigger replacement."""
    existing = _cr(amg_dr=12.5, sources=[ReviewSourceEntry(source="AMG", rating=4.5)])
    new = _cr(amg_dr=12.5, sources=[ReviewSourceEntry(source="AMG", rating=4.5)])
    assert _critical_reception_is_richer(new, existing) is False


def test_same_source_count_but_field_swap_is_not_richer() -> None:
    """New has same source count but different fields (lost rating, gained favorite).

    Total field count is equal, so wholesale replacement would lose the rating
    data — must not classify as richer.
    """
    existing = _cr(sources=[ReviewSourceEntry(source="AMG", rating=4.5)])
    new = _cr(sources=[ReviewSourceEntry(source="AMG", favorite=True)])
    assert _critical_reception_is_richer(new, existing) is False
