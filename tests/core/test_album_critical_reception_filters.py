"""
Tests for the critical_reception filter SQL builder in the albums controller.

These verify both the structure of the generated WHERE fragments and that they
behave correctly against a real (in-memory) SQLite engine using the JSON1 extension.
"""

import inspect
import json
import sqlite3
from typing import get_type_hints

from music_assistant_models.enums import MediaType, SortDirection, SortField

from music_assistant.controllers.music.constants import LEGACY_SORT_KEYS
from music_assistant.controllers.music.media.albums import (
    AlbumsController,
    _apply_critical_reception_filters,
)
from music_assistant.controllers.music.sorting import MEDIA_TYPE_SORT_FIELDS
from music_assistant.helpers.api import parse_arguments


def _build(**kwargs: object) -> tuple[list[str], dict[str, object]]:
    """Run the filter builder with sensible defaults filled in."""
    parts: list[str] = []
    params: dict[str, object] = {}
    base = {
        "dr_buckets": None,
        "amg_ratings": None,
        "amg_favorite": None,
        "amg_accolades": None,
        "amg_untagged": None,
        "tps_ratings": None,
        "tps_favorite": None,
        "tps_accolades": None,
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
    # DR values now live at metadata.dynamic_range (the canonical, measured value).
    # critical_reception holds only review-derived data (AMG/TPS sources + amg_dr).
    rows = [
        # 1: AMG 4.5 + Album of the Year + YMIO column + DR 16 (excellent)
        (
            1,
            {
                "dynamic_range": 16.0,
                "critical_reception": {
                    "sources": [
                        {
                            "source": "AMG",
                            "rating": 4.5,
                            "accolades": [
                                "Album of the Year (2024)",
                                "YMIO",
                            ],
                        },
                    ],
                },
            },
        ),
        # 2: TPS 8.5 + dated Honorable Mention + Lost in Time + DR 11 (good)
        (
            2,
            {
                "dynamic_range": 11.0,
                "critical_reception": {
                    "sources": [
                        {
                            "source": "TPS",
                            "rating": 8.5,
                            "accolades": ["Honorable Mention (2024)", "Lost in Time"],
                        }
                    ],
                },
            },
        ),
        # 3: AMG list-pick (favorite) + TYMHM column, DR 6 (poor)
        (
            3,
            {
                "dynamic_range": 6.0,
                "critical_reception": {
                    "sources": [{"source": "AMG", "favorite": True, "accolades": ["TYMHM"]}],
                },
            },
        ),
        # 4: TPS 9.2 + undated Honorable Mention + RFU + DR 8 (fair)
        (
            4,
            {
                "dynamic_range": 8.0,
                "critical_reception": {
                    "sources": [
                        {
                            "source": "TPS",
                            "rating": 9.2,
                            "accolades": ["Honorable Mention", "RFU"],
                        }
                    ],
                },
            },
        ),
        # 5: completely untagged
        (5, {}),
        # 6: only DR, no sources, DR 13 (good)
        (6, {"dynamic_range": 13.0}),
    ]
    for item_id, meta in rows:
        con.execute(
            "INSERT INTO albums (item_id, metadata) VALUES (?, ?)",
            (item_id, json.dumps(meta)),
        )
    return con


def test_no_filters_emits_nothing() -> None:
    """An unfiltered call contributes no WHERE fragments and binds no params."""
    parts, params = _build()
    assert parts == []
    assert params == {}


def test_dr_bucket_excellent_matches_only_album_with_dr_ge_14() -> None:
    """The excellent bucket selects only albums whose dynamic range is 14 or above."""
    con = _make_db()
    assert _exec_with_filters(con, dr_buckets=["excellent"]) == [1]


def test_dr_bucket_good_includes_dr_only_album() -> None:
    """Album 6 has DR 13 (good) and no sources — still must match."""
    con = _make_db()
    assert _exec_with_filters(con, dr_buckets=["good"]) == [2, 6]


def test_dr_bucket_poor_below_threshold() -> None:
    """The poor bucket selects only albums below its upper dynamic-range threshold."""
    con = _make_db()
    assert _exec_with_filters(con, dr_buckets=["poor"]) == [3]


def test_dr_bucket_untagged_keeps_albums_without_dr() -> None:
    """Albums with no DR (5) must surface for the 'untagged' DR bucket."""
    con = _make_db()
    assert _exec_with_filters(con, dr_buckets=["untagged"]) == [5]


def test_dr_buckets_combine_with_or() -> None:
    """Multiple DR buckets union rather than intersect, so an album matching either is kept."""
    con = _make_db()
    assert _exec_with_filters(con, dr_buckets=["excellent", "good"]) == [1, 2, 6]


def test_amg_rating_half_star_buckets() -> None:
    """AMG selectors are exact half-star steps: 4.5 is its own bucket, not part of 4."""
    con = _make_db()
    assert _exec_with_filters(con, amg_ratings=[4.5]) == [1]
    assert _exec_with_filters(con, amg_ratings=[4]) == []
    assert _exec_with_filters(con, amg_ratings=[5]) == []
    # picking both halves of a star spans what the old whole-star selector covered
    assert _exec_with_filters(con, amg_ratings=[4, 4.5]) == [1]


def test_rating_selectors_survive_the_api_parse_layer() -> None:
    """
    A JSON payload mixing ints and half stars must reach the controller intact.

    JSON has no float literal, so the 4-star bucket arrives as ``4``; against a
    strict ``list[float]`` annotation the api parser fails the union and drops the
    whole argument, silently turning the filter off. Hence ``list[float | int]``.
    """
    func = AlbumsController.library_items
    sig = inspect.signature(func)
    sig = sig.replace(parameters=[p for n, p in sig.parameters.items() if n != "self"])
    parsed = parse_arguments(
        sig, get_type_hints(func), {"amg_ratings": [4, 4.5], "tps_ratings": [7]}
    )
    assert parsed["amg_ratings"] == [4, 4.5]
    assert parsed["tps_ratings"] == [7]


def test_amg_rating_selector_vocabulary() -> None:
    """The whole 0.5..5.0 half-star scale binds; off-step values are dropped."""
    _, params = _build(amg_ratings=[0.5])
    assert params == {"amg_rb_lo_0": 0.5, "amg_rb_hi_0": 1.0}
    parts, params = _build(amg_ratings=[4.3])
    assert parts == []
    assert params == {}


def test_tps_rating_half_point_steps() -> None:
    """
    TPS selectors are single half-point steps over 0.5..10, like AMG's half stars.

    Album 2 is TPS 8.5 and album 4 an off-grid 9.2, which lands in the 9 step.
    """
    con = _make_db()
    assert _exec_with_filters(con, tps_ratings=[8.5]) == [2]
    assert _exec_with_filters(con, tps_ratings=[9]) == [4]
    assert _exec_with_filters(con, tps_ratings=[8, 9.5]) == []
    assert _exec_with_filters(con, tps_ratings=[8.5, 9]) == [2, 4]


def test_tps_rating_selector_vocabulary() -> None:
    """The whole 0.5..10 half-point scale binds; off-step values are dropped."""
    _, params = _build(tps_ratings=[10])
    assert params == {"tps_rb_lo_0": 10.0, "tps_rb_hi_0": 10.5}
    parts, params = _build(tps_ratings=[8.75])
    assert parts == []
    assert params == {}


def test_amg_favorite_only() -> None:
    """The AMG favourite flag keeps only albums whose AMG entry is marked favourite."""
    con = _make_db()
    assert _exec_with_filters(con, amg_favorite=True) == [3]


def test_dated_award_kinds_match_by_prefix() -> None:
    """
    Dated awards inline their date, so the kind prefix-matches every variant.

    Album 2 (dated) and album 4 (undated) both carry "Honorable Mention…" and match the
    single honorable_mention kind.
    """
    con = _make_db()
    assert _exec_with_filters(con, amg_accolades=["aoty"]) == [1]
    assert _exec_with_filters(con, tps_accolades=["honorable_mention"]) == [2, 4]


def test_accolade_filter_matches_exact_review_column() -> None:
    """Review-column accolades match exactly (no prefix), scoped to their own source."""
    con = _make_db()
    assert _exec_with_filters(con, amg_accolades=["tymhm"]) == [3]
    assert _exec_with_filters(con, amg_accolades=["ymio"]) == [1]


def test_accolade_filter_matches_value_with_spaces() -> None:
    """A multi-word column like 'Lost in Time' (kind 'lit') matches its stored string."""
    con = _make_db()
    assert _exec_with_filters(con, tps_accolades=["lit"]) == [2]
    assert _exec_with_filters(con, tps_accolades=["rfu"]) == [4]


def test_accolades_combine_with_or_within_source() -> None:
    """Multiple requested kinds OR together: tymhm (3) and ymio (1) both surface."""
    con = _make_db()
    assert _exec_with_filters(con, amg_accolades=["tymhm", "ymio"]) == [1, 3]


def test_accolade_filter_is_source_scoped() -> None:
    """A column matches only under the source that carries it (albums 2 and 4 are TPS)."""
    con = _make_db()
    assert _exec_with_filters(con, tps_accolades=["tymhm"]) == []
    assert _exec_with_filters(con, tps_accolades=["ymio"]) == []
    assert _exec_with_filters(con, amg_accolades=["lit", "rfu"]) == []


def test_empty_accolade_list_emits_nothing() -> None:
    """An all-unknown accolade list produces no SQL nor params; it's simply ignored."""
    parts, params = _build(amg_accolades=[""])
    assert parts == []
    assert params == {}


def test_amg_untagged_matches_albums_without_amg_entry() -> None:
    """Albums 2, 4, 5, 6 have no AMG entry."""
    con = _make_db()
    assert _exec_with_filters(con, amg_untagged=True) == [2, 4, 5, 6]


def test_filters_combine_as_and_across_fields() -> None:
    """DR good AND TPS [8.5, 9) → only album 2 (DR 11 + TPS 8.5)."""
    con = _make_db()
    assert _exec_with_filters(con, dr_buckets=["good"], tps_ratings=[8.5]) == [2]


def test_unknown_accolade_kind_silently_dropped() -> None:
    """A bogus accolade kind should not produce SQL nor crash; it's just ignored."""
    parts, params = _build(amg_accolades=["totally-made-up"])
    assert parts == []
    assert params == {}


def _make_amg_dr_fallback_db() -> sqlite3.Connection:
    """
    Separate fixture so amg_dr-fallback assertions don't perturb the main fixture.

    Mirrors the streaming-album case that motivated the COALESCE: a Listen Later
    row with no measured DR (never played → never analyzed) but with an AMG-tagged
    DR carried in via the listen_later_add `critical_reception` payload.
    """
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE albums (item_id INTEGER PRIMARY KEY, metadata json)")
    rows = [
        # 10: amg_dr only (good via fallback)
        (10, {"critical_reception": {"amg_dr": 12.0, "sources": []}}),
        # 11: amg_dr only (poor via fallback)
        (11, {"critical_reception": {"amg_dr": 5.0, "sources": []}}),
        # 12: measured 16 + amg_dr 5 — measured must win (COALESCE picks the first
        # non-null), so this album buckets as excellent, not poor.
        (
            12,
            {
                "dynamic_range": 16.0,
                "critical_reception": {"amg_dr": 5.0, "sources": []},
            },
        ),
        # 13: neither — still untagged.
        (13, {"critical_reception": {"sources": []}}),
    ]
    for item_id, meta in rows:
        con.execute(
            "INSERT INTO albums (item_id, metadata) VALUES (?, ?)",
            (item_id, json.dumps(meta)),
        )
    return con


def test_dr_bucket_falls_back_to_amg_dr() -> None:
    """An album with only amg_dr buckets by that value (matches the badge fallback)."""
    con = _make_amg_dr_fallback_db()
    assert _exec_with_filters(con, dr_buckets=["good"]) == [10]
    assert _exec_with_filters(con, dr_buckets=["poor"]) == [11]


def test_dr_bucket_measured_wins_over_amg_dr() -> None:
    """When both values exist, measured wins (COALESCE picks dynamic_range first)."""
    con = _make_amg_dr_fallback_db()
    # Album 12 has measured=16 (excellent) and amg_dr=5 — must bucket as excellent.
    assert _exec_with_filters(con, dr_buckets=["excellent"]) == [12]
    assert _exec_with_filters(con, dr_buckets=["poor"]) == [11]


def test_dr_bucket_untagged_requires_both_null() -> None:
    """`untagged` matches only albums where both measured and amg_dr are null."""
    con = _make_amg_dr_fallback_db()
    # Albums 10, 11, 12 all have *some* DR value; only 13 has neither.
    assert _exec_with_filters(con, dr_buckets=["untagged"]) == [13]


def test_sort_keys_are_registered_for_albums() -> None:
    """
    The album-scoped sort keys live on AlbumsController.extra_sort_keys, typed as album-only.

    They reference albums.metadata JSON paths that only exist on this table: each maps to a
    fork SortField that only the album listings offer, so resolve_sort rejects it elsewhere.
    """
    fields = {
        SortField.DR,
        SortField.AMG_RATING,
        SortField.TPS_RATING,
        SortField.LISTEN_LATER_ADDED_AT,
    }
    for field in fields:
        for key, direction in ((field.value, SortDirection.ASC), (f"{field.value}_desc", None)):
            assert key in AlbumsController.extra_sort_keys, f"sort key {key!r} missing"
            assert LEGACY_SORT_KEYS[key] == (field, direction or SortDirection.DESC)
    for media_type, offered in MEDIA_TYPE_SORT_FIELDS.items():
        if media_type != MediaType.ALBUM:
            assert not fields & set(offered), f"album-only sort offered for {media_type}"
    assert fields <= set(MEDIA_TYPE_SORT_FIELDS[MediaType.ALBUM])


def test_typed_album_sort_renders_the_extra_sort_key() -> None:
    """A typed fork sort field renders the extra_sort_keys SQL of its direction."""
    controller = AlbumsController.__new__(AlbumsController)
    keys = AlbumsController.extra_sort_keys
    assert controller._get_sort_sql(SortField.DR, SortDirection.ASC) == keys["dr"]
    assert controller._get_sort_sql(SortField.DR, None) == keys["dr_desc"]
    assert (
        controller._get_sort_sql(SortField.AMG_RATING, SortDirection.DESC)
        == keys["amg_rating_desc"]
    )
    assert (
        controller._get_sort_sql(SortField.LISTEN_LATER_ADDED_AT, None)
        == keys["listen_later_added_at_desc"]
    )
