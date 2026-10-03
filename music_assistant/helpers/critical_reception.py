"""Richness comparison rules for album critical_reception (review) metadata."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from music_assistant_models.media_items.metadata import ReviewSourceEntry


# The value-bearing fields of a ReviewSourceEntry, excluding the `source` identifier.
# Every richness/preservation check below is driven off this one roster, so adding a
# field to ReviewSourceEntry only has to be recorded here.
_REVIEW_SOURCE_FIELDS: Final[tuple[str, ...]] = (
    "rating",
    "favorite",
    "accolades",
    "links",
    "authors",
    "review",
)
# The subset of the above whose values are lists; these compare None and [] as equal.
_REVIEW_SOURCE_LIST_FIELDS: Final[frozenset[str]] = frozenset({"accolades", "links", "authors"})


def critical_reception_is_richer(new: object, existing: object) -> bool:
    """
    Return True if `new` carries more or fresher critical_reception data than `existing`.

    "Richer" means either strictly more populated fields, or the same shape with at
    least one field value that actually changed (e.g. a refreshed rating). Both callers
    treat a False answer as "keep what is stored": the album library sync uses it to
    decide whether the row is worth writing at all, and `_update_library_item` uses it to
    decide whether to restore the stored CR over the merge result. So any regression — a
    lost amg_dr, a dropped source, or a per-source field that goes from populated to
    blank — has to be rejected; the stored copy stays in that case.
    """
    if new is None:
        return False
    if existing is None:
        return True
    new_amg_dr = getattr(new, "amg_dr", None)
    cur_amg_dr = getattr(existing, "amg_dr", None)
    new_sources = getattr(new, "sources", None) or []
    cur_sources = getattr(existing, "sources", None) or []
    # Regression on either dimension would erase data on wholesale replace.
    if cur_amg_dr is not None and new_amg_dr is None:
        return False
    if len(new_sources) < len(cur_sources):
        return False
    # Per-source field-level regression check: same source name on both sides,
    # populated field on `existing` must still be populated on `new`.
    if not _sources_preserve_data(new_sources, cur_sources):
        return False
    new_total = (1 if new_amg_dr is not None else 0) + _total_source_field_count(new_sources)
    cur_total = (1 if cur_amg_dr is not None else 0) + _total_source_field_count(cur_sources)
    if new_total > cur_total:
        return True
    if new_total < cur_total:
        return False
    # Same field count and no field-level regression: accept when at least one
    # value actually changed (refreshed rating, swapped accolade, new amg_dr) so
    # meaningful updates don't get stuck behind an equal-shape stored copy.
    if new_amg_dr != cur_amg_dr:
        return True
    new_by_name = _sources_by_name(new_sources)
    return any(
        _source_signature(new_by_name.get(getattr(s, "source", ""))) != _source_signature(s)
        for s in cur_sources
    )


def _field_weight(value: Any) -> int:
    """
    Richness weight of one ReviewSourceEntry field value.

    Scalars count once when set (``favorite=False`` and ``rating=0.0`` are set);
    list fields count once per element, so gaining an accolade counts as richer.
    """
    if value is None:
        return 0
    if isinstance(value, list | tuple | set):
        return len(value)
    return 1


def _field_is_populated(value: Any) -> bool:
    """Return True when a ReviewSourceEntry field is populated (not None / empty)."""
    return _field_weight(value) > 0


def _source_filled_field_count(source: ReviewSourceEntry | None) -> int:
    """
    Count populated fields on a single ReviewSourceEntry.

    Used as a per-source richness signal so a refresh that gains a rating,
    favorite flag, or extra accolade label wins over the stored copy even when
    the total source count hasn't changed.
    """
    if source is None:
        return 0
    return sum(_field_weight(getattr(source, field)) for field in _REVIEW_SOURCE_FIELDS)


def _total_source_field_count(sources: Sequence[ReviewSourceEntry] | None) -> int:
    """Sum of filled fields across every entry in a sources iterable."""
    if not sources:
        return 0
    return sum(_source_filled_field_count(s) for s in sources)


def _sources_by_name(sources: Any) -> dict[str, Any]:
    """
    Map source identifier -> ReviewSourceEntry for a sources iterable.

    Drops empty / blank source names. On duplicate source names (which the
    downstream SQL filters assume away), keep the FIRST entry — the strict
    richness check then compares each cur entry against the same stable
    reference. A dict comprehension would otherwise keep the *last* entry,
    making the richness check asymmetric on inputs that violate the
    "one entry per source" invariant.
    """
    result: dict[str, Any] = {}
    for s in sources or []:
        if s is None:
            continue
        name = getattr(s, "source", "") or ""
        if not name or name in result:
            continue
        result[name] = s
    return result


def _source_preserves_data(new_source: Any, cur_source: Any) -> bool:
    """
    Return True when `new_source` keeps every populated field from `cur_source`.

    A field that's populated on the stored copy must still be populated on the
    incoming one — losing a rating, dropping all accolades, etc. is a regression the
    callers must not act on: it should neither pull the album into the sync's update
    branch nor let `_update_library_item` keep the merge result over `stored_cr`.

    List-valued fields (accolades, links, authors) only have to stay non-empty,
    not stay a superset. For file-tag-derived CR the provider re-probe is
    authoritative: an accolade set that legitimately changes over time (an award
    revised, a new honorable mention added in a later year, a stale one dropped)
    is a valid refresh, not a regression — and gating it behind a strict superset
    would also block an additive refresh (e.g. one that adds review links) whenever
    the set happened to change. Net data loss across the whole CR is still guarded
    by the aggregate field-count check in `critical_reception_is_richer`.
    """
    if cur_source is None:
        return True
    if new_source is None:
        return False
    for field in _REVIEW_SOURCE_FIELDS:
        cur_val = getattr(cur_source, field, None)
        new_val = getattr(new_source, field, None)
        if not _field_is_populated(cur_val):
            continue
        if not _field_is_populated(new_val):
            return False
    return True


def _sources_preserve_data(new_sources: Any, cur_sources: Any) -> bool:
    """Return True when every existing source's populated fields survive on the new side."""
    new_by_name = _sources_by_name(new_sources)
    for cur in cur_sources or []:
        name = getattr(cur, "source", "")
        if not _source_preserves_data(new_by_name.get(name), cur):
            return False
    return True


def _source_signature(source: ReviewSourceEntry | None) -> tuple[Any, ...]:
    """Return a comparable snapshot of every value-bearing field on a ReviewSourceEntry."""
    if source is None:
        return ()
    return (
        source.source,
        *(
            tuple(getattr(source, field) or ())
            if field in _REVIEW_SOURCE_LIST_FIELDS
            else getattr(source, field)
            for field in _REVIEW_SOURCE_FIELDS
        ),
    )
