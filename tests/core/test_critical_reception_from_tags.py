"""Tests for critical_reception_from_tags: music-rater's tag payload -> CriticalReception (MUSIC-79)."""

from __future__ import annotations

from music_assistant_models.media_items.metadata import (
    CriticalReception,
    ReviewLink,
    ReviewSourceEntry,
)

from music_assistant.helpers.tags import critical_reception_from_tags

# Shape of music-rater's GET /tags/for-album `fields`: raw tag names, multi-values as lists.
_PAYLOAD = {
    "AMG_RATING": "4.0",
    "AMG_FAVORITE": "true",
    "AMG_LINKS": [
        "Album of the Year (2025) — https://www.angrymetalguy.com/aoty-2025/",
        "Review — https://www.angrymetalguy.com/ii-apostles-of-the-flesh-review/",
    ],
    "AMG_ACCOLADES": ["Album of the Year (2025)", "Review"],
    "AMG_AUTHOR": ["Spicie Forrest"],
    "AMG_REVIEW": "Before you ask; it's pronounced “two eyes.”",
    "TPS_RATING": "7.5",
    "TPS_ACCOLADES": ["Review"],
    "GENRE": ["Darkwave", "Gothic Rock"],
    "MR_PARSER_VERSION": "1.98.0",
    "MR_TAG_SCHEMA_VERSION": "3.6.0",
}


def test_payload_maps_to_critical_reception() -> None:
    """Raw tag names and list values parse exactly like the file tags do."""
    assert critical_reception_from_tags(_PAYLOAD) == CriticalReception(
        amg_dr=None,
        sources=[
            ReviewSourceEntry(
                source="AMG",
                rating=4.0,
                favorite=True,
                accolades=["Album of the Year (2025)", "Review"],
                links=[
                    ReviewLink(
                        label="Album of the Year (2025)",
                        url="https://www.angrymetalguy.com/aoty-2025/",
                    ),
                    ReviewLink(
                        label="Review",
                        url="https://www.angrymetalguy.com/ii-apostles-of-the-flesh-review/",
                    ),
                ],
                authors=["Spicie Forrest"],
                review="Before you ask; it's pronounced “two eyes.”",
            ),
            ReviewSourceEntry(source="TPS", rating=7.5, accolades=["Review"]),
        ],
    )


def test_amg_dr_is_read() -> None:
    """The legacy AMG_ALBUM_DYNAMIC_RANGE key still lands on amg_dr."""
    cr = critical_reception_from_tags({"AMG_ALBUM_DYNAMIC_RANGE": "7"})
    assert cr == CriticalReception(amg_dr=7.0, sources=None)


def test_no_review_fields_is_none() -> None:
    """Genre and audit stamps alone are not critical reception."""
    assert critical_reception_from_tags({"GENRE": ["Doom"], "MR_PARSER_VERSION": "1"}) is None
    assert critical_reception_from_tags({}) is None
