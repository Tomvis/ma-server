"""Tests for the critical_reception filter SQL builder in the albums controller.

These verify both the structure of the generated WHERE fragments and that they
behave correctly against a real (in-memory) SQLite engine using the JSON1 extension.
"""

import json
import sqlite3

from music_assistant.controllers.media.albums import _apply_critical_reception_filters
from music_assistant.controllers.media.base import SORT_KEYS


def _build(**kwargs: object) -> tuple[list[str], dict[str, object]]:
    """Run the filter builder with sensible defaults filled in."""
    parts: list[str] = []
    params: dict[str, object] = {}
    base = {
        "dr_buckets": None,
        "amg_ratings": None,
        "amg_favorite": None,
        "amg_labels": None,
        "amg_untagged": None,
        "tps_ratings": None,
        "tps_favorite": None,
        "tps_labels": None,
        "tps_untagged": None,
    }
    base.update(kwargs)
    _apply_critical_reception_filters(query_parts=parts, query_params=params, **base)  # type: ignore[arg-type]
    return parts, params


def _exec_with_filters(con: sqlite3.Connection, **kwargs: object) -> list[int]:
    parts, params = _build(**kwargs)
    sql = "SELECT item_id FROM albums"
    if parts:
        sql += " WHERE " + " AND ".join(parts)
    sql += " ORDER BY item_id"
    return [row[0] for row in con.execute(sql, params).fetchall()]


def _make_db() -> sqlite3.Connection:
    """In-memory SQLite seeded with a fixed mix of albums for filter tests."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE albums (item_id INTEGER PRIMARY KEY, metadata json)")
    rows = [
        # 1: AMG 4.5 + AOTY-2024 + DR 16 (excellent)
        (
            1,
            {
                "critical_reception": {
                    "dr": 16.0,
                    "sources": [
                        {"source": "AMG", "rating": 4.5, "labels": ["AOTY-2024"]},
                    ],
                }
            },
        ),
        # 2: TPS 8.5 + AOTM-2024-03 + DR 11 (good)
        (
            2,
            {
                "critical_reception": {
                    "dr": 11.0,
                    "sources": [
                        {
                            "source": "TPS",
                            "rating": 8.5,
                            "labels": ["AOTM-2024-03"],
                        }
                    ],
                }
            },
        ),
        # 3: AMG list-pick (favorite) only, DR 6 (poor)
        (
            3,
            {
                "critical_reception": {
                    "dr": 6.0,
                    "sources": [{"source": "AMG", "favorite": True}],
                }
            },
        ),
        # 4: TPS 9.2 + Record of the Month + DR 8 (fair)
        (
            4,
            {
                "critical_reception": {
                    "dr": 8.0,
                    "sources": [
                        {
                            "source": "TPS",
                            "rating": 9.2,
                            "labels": ["RECORD_OF_THE_MONTH"],
                        }
                    ],
                }
            },
        ),
        # 5: completely untagged
        (5, {}),
        # 6: only DR, no sources, DR 13 (good)
        (6, {"critical_reception": {"dr": 13.0}}),
    ]
    for item_id, meta in rows:
        con.execute(
            "INSERT INTO albums (item_id, metadata) VALUES (?, ?)",
            (item_id, json.dumps(meta)),
        )
    return con


def test_no_filters_emits_nothing() -> None:
    parts, params = _build()
    assert parts == []
    assert params == {}


def test_dr_bucket_excellent_matches_only_album_with_dr_ge_14() -> None:
    con = _make_db()
    assert _exec_with_filters(con, dr_buckets=["excellent"]) == [1]


def test_dr_bucket_good_includes_dr_only_album() -> None:
    """Album 6 has DR 13 (good) and no sources — still must match."""
    con = _make_db()
    assert _exec_with_filters(con, dr_buckets=["good"]) == [2, 6]


def test_dr_bucket_poor_below_threshold() -> None:
    con = _make_db()
    assert _exec_with_filters(con, dr_buckets=["poor"]) == [3]


def test_dr_bucket_untagged_keeps_albums_without_dr() -> None:
    """Albums with no DR (5) must surface for the 'untagged' DR bucket."""
    con = _make_db()
    assert _exec_with_filters(con, dr_buckets=["untagged"]) == [5]


def test_dr_buckets_combine_with_or() -> None:
    con = _make_db()
    assert _exec_with_filters(con, dr_buckets=["excellent", "good"]) == [1, 2, 6]


def test_amg_rating_floor_buckets() -> None:
    """AMG rating 4.5 lives in selector 4 (floor), not selector 5."""
    con = _make_db()
    assert _exec_with_filters(con, amg_ratings=[4]) == [1]
    assert _exec_with_filters(con, amg_ratings=[5]) == []


def test_tps_rating_three_step_buckets() -> None:
    """TPS uses 1/3/5/7/9 selectors; selector 7 covers [7, 9), selector 9 covers [9, 11)."""
    con = _make_db()
    assert _exec_with_filters(con, tps_ratings=[7]) == [2]
    assert _exec_with_filters(con, tps_ratings=[9]) == [4]
    assert _exec_with_filters(con, tps_ratings=[7, 9]) == [2, 4]


def test_amg_favorite_only() -> None:
    con = _make_db()
    assert _exec_with_filters(con, amg_favorite=True) == [3]


def test_label_filter_matches_year_suffixed_kinds() -> None:
    con = _make_db()
    assert _exec_with_filters(con, amg_labels=["aoty"]) == [1]
    assert _exec_with_filters(con, tps_labels=["aotm"]) == [2]
    assert _exec_with_filters(con, tps_labels=["record_of_the_month"]) == [4]


def test_amg_untagged_matches_albums_without_amg_entry() -> None:
    """Albums 2, 4, 5, 6 have no AMG entry."""
    con = _make_db()
    assert _exec_with_filters(con, amg_untagged=True) == [2, 4, 5, 6]


def test_filters_combine_as_and_across_fields() -> None:
    """DR good AND TPS [7,9) → only album 2 (DR 11 + TPS 8.5)."""
    con = _make_db()
    assert _exec_with_filters(con, dr_buckets=["good"], tps_ratings=[7]) == [2]


def test_unknown_label_kind_silently_dropped() -> None:
    """A bogus label kind should not produce SQL nor crash; it's just ignored."""
    parts, params = _build(amg_labels=["totally-made-up"])
    assert parts == []
    assert params == {}


def test_sort_keys_are_registered_for_albums() -> None:
    """The new sort keys must be present in the global SORT_KEYS table."""
    for key in (
        "dr",
        "dr_desc",
        "amg_rating",
        "amg_rating_desc",
        "tps_rating",
        "tps_rating_desc",
    ):
        assert key in SORT_KEYS, f"sort key {key!r} missing"
