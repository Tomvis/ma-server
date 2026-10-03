"""
Unit tests for `critical_reception_is_richer`.

The function decides whether an incoming critical_reception is worth acting on: the
album library sync uses it to widen the update condition (write the row at all), and
`AlbumsController._update_library_item` uses it to decide whether to restore the stored
CR over the deep-merge result. A False answer means "keep what is stored", so the
function must reject any change that loses data on either dimension (amg_dr or sources)
— otherwise that field would silently disappear.
"""

from __future__ import annotations

from music_assistant_models.media_items.metadata import (
    CriticalReception,
    ReviewLink,
    ReviewSourceEntry,
)

from music_assistant.helpers.critical_reception import critical_reception_is_richer


def _cr(
    amg_dr: float | None = None, sources: list[ReviewSourceEntry] | None = None
) -> CriticalReception:
    return CriticalReception(amg_dr=amg_dr, sources=sources)


def test_returns_false_when_new_is_none() -> None:
    """`new is None` never replaces an existing CR (would erase data)."""
    existing = _cr(amg_dr=12.5)
    assert critical_reception_is_richer(None, existing) is False


def test_returns_true_when_existing_is_none() -> None:
    """Any populated CR is richer than nothing."""
    new = _cr(amg_dr=12.5)
    assert critical_reception_is_richer(new, None) is True


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
    assert critical_reception_is_richer(new, existing) is False


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
    assert critical_reception_is_richer(new, existing) is False


def test_amg_dr_added_with_same_sources_is_richer() -> None:
    """A fresh amg_dr on an otherwise-identical CR is a strict upgrade."""
    existing = _cr(amg_dr=None, sources=[ReviewSourceEntry(source="AMG", rating=4.5)])
    new = _cr(amg_dr=12.5, sources=[ReviewSourceEntry(source="AMG", rating=4.5)])
    assert critical_reception_is_richer(new, existing) is True


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
    assert critical_reception_is_richer(new, existing) is True


def test_same_source_gains_field_is_richer() -> None:
    """A refresh that fills in a previously-blank field counts as richer."""
    existing = _cr(sources=[ReviewSourceEntry(source="AMG", rating=4.5)])
    new = _cr(
        sources=[
            ReviewSourceEntry(source="AMG", rating=4.5, accolades=["Album of the Year (2024)"])
        ]
    )
    assert critical_reception_is_richer(new, existing) is True


def test_identical_cr_is_not_richer() -> None:
    """No-change refresh must not trigger replacement."""
    existing = _cr(amg_dr=12.5, sources=[ReviewSourceEntry(source="AMG", rating=4.5)])
    new = _cr(amg_dr=12.5, sources=[ReviewSourceEntry(source="AMG", rating=4.5)])
    assert critical_reception_is_richer(new, existing) is False


def test_same_source_count_but_field_swap_is_not_richer() -> None:
    """
    New has same source count but different fields (lost rating, gained favorite).

    Total field count is equal, so wholesale replacement would lose the rating
    data — must not classify as richer.
    """
    existing = _cr(sources=[ReviewSourceEntry(source="AMG", rating=4.5)])
    new = _cr(sources=[ReviewSourceEntry(source="AMG", favorite=True)])
    assert critical_reception_is_richer(new, existing) is False


def test_changed_accolade_set_plus_added_links_is_richer() -> None:
    """
    A file-tag re-probe that swaps an accolade AND adds review links is richer.

    The accolade set legitimately changes over time (an award revised, a stale one
    dropped, a new honorable mention added in a later year). Such a re-tag is no
    longer a subset of the stored set, but it grows the total field count (it adds
    links), so it must win — not get blocked by the old strict-superset rule.
    """
    existing = _cr(
        sources=[
            ReviewSourceEntry(
                source="AMG",
                rating=4.5,
                accolades=["Record of the Month (Mar 2021)", "Honorable Mention (2021)"],
            )
        ]
    )
    new = _cr(
        sources=[
            ReviewSourceEntry(
                source="AMG",
                rating=4.5,
                accolades=["Honorable Mention (2021)", "Honorable Mention (2022)"],
                links=[ReviewLink(label="Honorable Mention", url="https://example.com/post")],
            )
        ]
    )
    assert critical_reception_is_richer(new, existing) is True


def test_changed_accolade_set_with_net_loss_is_not_richer() -> None:
    """
    A re-tag that swaps an accolade but shrinks the total field count is rejected.

    Even with the superset rule relaxed, the aggregate field-count guard still
    protects against net data loss: dropping two accolades for one, with nothing
    added to compensate, lowers the total and must keep the stored copy.
    """
    existing = _cr(
        sources=[
            ReviewSourceEntry(
                source="AMG",
                rating=4.5,
                accolades=[
                    "Record of the Month (Mar 2021)",
                    "Honorable Mention (2021)",
                    "Album of the Year (2021)",
                ],
            )
        ]
    )
    new = _cr(
        sources=[
            ReviewSourceEntry(source="AMG", rating=4.5, accolades=["Honorable Mention (2022)"])
        ]
    )
    assert critical_reception_is_richer(new, existing) is False


def test_dropping_all_accolades_is_not_richer() -> None:
    """
    A new source that keeps the rating but blanks a populated accolades list regresses.

    The populated-stays-populated guard is independent of the superset relaxation:
    going from accolades=[...] to no accolades erases data on the wholesale replace.
    """
    existing = _cr(
        sources=[
            ReviewSourceEntry(source="AMG", rating=4.5, accolades=["Album of the Year (2024)"])
        ]
    )
    new = _cr(sources=[ReviewSourceEntry(source="AMG", rating=4.5)])
    assert critical_reception_is_richer(new, existing) is False


def test_gaining_review_text_is_richer_and_losing_it_is_not() -> None:
    """Review text (3.6.0+) counts like any field: gained wins, dropped keeps stored."""
    bare = _cr(sources=[ReviewSourceEntry(source="AMG", rating=4.0)])
    reviewed = _cr(sources=[ReviewSourceEntry(source="AMG", rating=4.0, review="Text.")])
    edited = _cr(sources=[ReviewSourceEntry(source="AMG", rating=4.0, review="Edited.")])
    assert critical_reception_is_richer(reviewed, bare) is True
    assert critical_reception_is_richer(bare, reviewed) is False
    assert critical_reception_is_richer(edited, reviewed) is True
