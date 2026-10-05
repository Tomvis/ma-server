"""Manage MediaItems of type Album."""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from time import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import aiohttp
from music_assistant_models.auth import Scope
from music_assistant_models.enums import (
    AlbumType,
    EventType,
    ExternalID,
    MediaType,
    ProviderFeature,
)
from music_assistant_models.errors import (
    InsufficientPermissions,
    InvalidDataError,
    LoginFailed,
    MediaNotFoundError,
    MusicAssistantError,
    ProviderUnavailableError,
    ResourceTemporarilyUnavailable,
    RetriesExhausted,
)
from music_assistant_models.helpers import create_safe_string
from music_assistant_models.media_items import (
    Album,
    AlbumSummary,
    Artist,
    ItemMapping,
    MediaItemImage,
    ProviderMapping,
    Track,
    UniqueList,
)
from music_assistant_models.media_items.metadata import CriticalReception, MediaItemMetadata

from music_assistant.constants import (
    DB_TABLE_ALBUM_ARTISTS,
    DB_TABLE_ALBUM_LISTEN_LATER,
    DB_TABLE_ALBUM_TRACKS,
    DB_TABLE_ALBUMS,
)
from music_assistant.controllers.music.helpers import (
    fill_track_from_recording,
    metadata_for_update,
    provider_mappings_for_update,
    provider_mappings_from_urls,
    search_name_match_clause,
)
from music_assistant.controllers.webserver.helpers.auth_middleware import get_current_user
from music_assistant.helpers.compare import (
    ALBUM_RETAIL_SUFFIX_KEYS,
    AlbumMatchEvidence,
    album_tracks_have_positions,
    compare_album_evidence,
    compare_artists,
    compare_strings,
    loose_compare_strings,
    strip_album_retail_suffix,
)
from music_assistant.helpers.critical_reception import critical_reception_is_richer
from music_assistant.helpers.database import UNSET
from music_assistant.helpers.datetime import utc_timestamp
from music_assistant.helpers.external_ids import (
    barcode_to_upc,
    is_valid_barcode,
    is_valid_isrc,
    normalize_external_id,
)
from music_assistant.helpers.json import json_loads, serialize_to_json
from music_assistant.helpers.tags import ACCOLADE_KINDS
from music_assistant.helpers.uri import share_url_provider
from music_assistant.helpers.util import try_parse_float
from music_assistant.models.music_provider import PROVIDER_FETCH_ERRORS, MusicProvider
from music_assistant.providers.musicbrainz.provider import (
    is_digital_release,
    relation_urls,
    release_matches_album,
)

from .base import (
    AUTHORITATIVE_REFRESH,
    EXTERNAL_ID_LOOKUP_ERRORS,
    MAX_EXTERNAL_ID_MATCH_LOOKUPS,
    LibraryItemSyncDetails,
    MediaControllerBase,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from music_assistant import MusicAssistant
    from music_assistant.providers.musicbrainz import MusicbrainzProvider
    from music_assistant.providers.musicbrainz.models import (
        MusicBrainzBarcodeRelease,
        MusicBrainzRecording,
        MusicBrainzRelease,
    )


# expected failures from a provider album-track lookup: a missing item, an unavailable or
# rate limiting provider or a transient transport outage. Each leaves that tracklist
# unavailable so the (best-effort, multi-provider) match can continue rather than aborting
# the whole operation.
_ALBUM_TRACK_LOOKUP_ERRORS = (
    MediaNotFoundError,
    ProviderUnavailableError,
    ResourceTemporarilyUnavailable,
    RetriesExhausted,
    TimeoutError,
    aiohttp.ClientError,
)
# what costs one provider its album listing without failing the album itself
_SKIPPED_ALBUM_LISTING_ERRORS: tuple[type[Exception], ...] = (
    MusicAssistantError,
    *PROVIDER_FETCH_ERRORS,
    *_ALBUM_TRACK_LOOKUP_ERRORS,
)

# how many seconds the duration of one and the same track may differ between sources
_TRACK_DURATION_TOLERANCE = 8

# how many of a release group's official editions are looked up on the music providers,
# likeliest first: each one costs a MusicBrainz release lookup and a barcode fan-out
_MAX_EDITION_LOOKUPS = 3


@dataclass
class _BaseTracksMemo:
    """Single-slot memo holding the tracklist of one base album, resolved on first use."""

    resolved: bool = False
    tracks: list[Track] | None = None


# Bare-text search cutoff: library_items returns up to this many title-matches
# on page 1 before declining to top up with artist-only matches. library_count
# mirrors the cutoff so counts and lists track 1-1 — above the cutoff, only
# title-matching albums are visible, so the count must drop the artist union.
_SEARCH_ARTIST_PASS_CUTOFF = 25

# DR quality thresholds (mirrors src/helpers/album_tags.ts on the frontend).
_DR_BUCKET_RANGES: dict[str, tuple[float, float | None]] = {
    "excellent": (14, None),
    "good": (10, 14),
    "fair": (7, 10),
    "poor": (0, 7),
}

# Map a filter accolade-kind -> how it matches each string in the JSON accolades[]
# array (3.2.0+ merged shape). Derived from the ACCOLADE_KINDS vocabulary that
# *writes* those strings, so a renamed display name can't silently turn these
# filters into zero-row queries. Dated honors inline their date in the display
# string ("Album of the Year (2024)"), so they match by prefix on the bare award
# name — one kind covers every year/month variant; undated review-column kinds
# match exactly. "review" is excluded: it's the default column, not a filter.
# (mode, value): mode is "exact" or "prefix".
_ACCOLADE_KIND_MATCH: dict[str, tuple[str, str]] = {
    kind: ("prefix" if dated else "exact", display)
    for kind, (display, dated) in ACCOLADE_KINDS.items()
    if kind != "review"
}


# Table-valued scan of the album's critical-reception sources array; every CR
# filter clause and rating sort key below builds on this one fragment so a
# future schema move only needs a single edit.
_CR_SOURCES_EACH = "json_each(albums.metadata, '$.critical_reception.sources')"

# The canonical (measured) album dynamic range. Extracted for the same reason as
# _CR_SOURCES_EACH: it feeds the DR bucket filter, both dr sort keys, the summary
# query and the sync-details query, so a schema move stays a single edit.
_DR_JSON = "json_extract(albums.metadata, '$.dynamic_range')"

# The listen-later shelf is per-user: it lives in the album_listen_later association
# table, keyed on (item_id, userid), not on the `albums.listen_later` column that a
# fork database still carries. Every read below is bound to :listen_later_userid.
#
# The legacy columns are still on the albums row and still selected by `albums.*`, so
# the computed columns MUST come first in the SELECT list. With duplicate column names
# aiosqlite/sqlite3 Row resolves to the FIRST match, so an override appended after
# `albums.*` would be silently ignored and every read would fall back to the global
# bit. tests/core/test_album_listen_later_per_user.py pins that ordering.
_LISTEN_LATER_ADDED_AT_SQL = (
    f"(SELECT added_at FROM {DB_TABLE_ALBUM_LISTEN_LATER} "
    "WHERE item_id = albums.item_id AND userid = :listen_later_userid)"
)
_LISTEN_LATER_FLAG_SQL = (
    f"EXISTS(SELECT 1 FROM {DB_TABLE_ALBUM_LISTEN_LATER} "
    "WHERE item_id = albums.item_id AND userid = :listen_later_userid)"
)
# the membership test used by the `listen_later=` filter; phrased over albums.item_id
# so it composes with the other WHERE fragments without needing a JOIN
_LISTEN_LATER_MEMBER_SQL = (
    f"albums.item_id IN (SELECT item_id FROM {DB_TABLE_ALBUM_LISTEN_LATER} "
    "WHERE userid = :listen_later_userid)"
)


def _current_userid() -> str:
    """
    Return the calling user's id for a listen-later query, or "" when there is none.

    Fails closed on purpose. `userid` is NOT NULL and no account has an empty id, so ""
    matches no association row: a background task or an unauthenticated caller sees an
    empty shelf rather than the union of everyone's saves.
    """
    user = get_current_user()
    return user.user_id if user else ""


# Rating-bucket vocabulary per CR source: the selector values a client may send and
# the width of the bucket each selector spans. AMG rates in half stars, the full 0.5-5.0
# scale it publishes (Unlistenable .. Iconic), so a selector is one exact step: 4.5 covers
# [4.5, 5.0) and 4 covers [4.0, 4.5), keeping 4-star and 4.5-star albums separable. TPS
# selectors step in twos over its /10 scale.
_CR_RATING_SPECS: dict[str, tuple[frozenset[float], float]] = {
    "AMG": (frozenset(n / 2 for n in range(1, 11)), 0.5),
    "TPS": (frozenset({1.0, 3.0, 5.0, 7.0, 9.0}), 2.0),
}


def _cr_source_exists(source_sql: str, extra_sql: str = "") -> str:
    """
    EXISTS clause over the CR sources array, filtered to one source.

    :param source_sql: SQL fragment the source must equal (literal or bound param).
    :param extra_sql: Optional extra condition, including its leading `` AND ``.
    """
    return (
        f"EXISTS(SELECT 1 FROM {_CR_SOURCES_EACH} "
        f"WHERE json_extract(value, '$.source') = {source_sql}{extra_sql})"
    )


def _cr_rating_sort_key(source: str, direction: str) -> str:
    """ORDER BY fragment sorting on the given CR source's rating."""
    return (
        "(SELECT json_extract(value, '$.rating') "
        f"FROM {_CR_SOURCES_EACH} "
        f"WHERE json_extract(value, '$.source') = '{source}' LIMIT 1) {direction} NULLS LAST"
    )


def _like_prefix(value: str) -> str:
    """
    Escape LIKE wildcards in a literal, then append % for a prefix match.

    The clause that uses the result pairs it with ESCAPE so the doubled
    backslash escapes are honored against SQLite's % / _ wildcards.
    """
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}%"


def _coerce_float_list(values: Iterable[Any] | None) -> list[float]:
    """
    Coerce a heterogeneous list to floats; drop entries that won't survive float().

    The api commands annotate these as ``list[float | int]``, and the int arm is not
    redundant: JSON has no float literal, so a client picking the 4-star bucket sends
    ``4``, and against a strict ``list[float]`` the api parser fails the union and
    silently drops the *whole* argument (turning the filter off) rather than widening
    the value. The JSON deserializer also preserves stray ``null`` / non-numeric
    entries; without this filter a malformed payload would raise out of the
    bucket-clause builders.

    The ``bool`` guard is not redundant: ``try_parse_float(True)`` returns 1.0.
    """
    return [
        f
        for v in (values or ())
        if not isinstance(v, bool) and (f := try_parse_float(v, None)) is not None
    ]


def _track_position(track: Track) -> str:
    """
    Return the disc/track identity of a track, counting an unset disc as disc 1.

    A provider that serves single-disc releases has no disc to report: Bandcamp
    numbers every track disc 0, while the same album tagged locally is disc 1. Keyed
    on the raw number the two never line up, so the positional half of the album-track
    de-duplication below silently stops working and each track reads as one the
    library lacks.
    """
    return f"{track.disc_number or 1}.{track.track_number}"


_FEAT_SUFFIX = re.compile(
    r"\s*[([]?\s*(?:feat\.?|ft\.?|featuring|with)\s+[^)\]]*[)\]]?\s*$", re.IGNORECASE
)
_RELEASE_SUFFIX = re.compile(
    r"\s*[([](?:bonus track|bonus|remaster(?:ed)?(?: \d{4})?|live|demo|instrumental"
    r"|[a-z ]*cover|[a-z ]*version|edit|radio edit)[^)\]]*[)\]]\s*$",
    re.IGNORECASE,
)
_LEADING_TRACK_NO = re.compile(r"^\s*\d{1,2}\s*[-_.]\s*")


def _track_name_variants(name: str) -> set[str]:
    """
    Return the spellings a provider may use for a track the library already holds.

    A streaming listing decorates a title in ways a local tag does not: a "(feat. X)"
    credit the tagger folded into the artist, a "(Bonus Track)" / "(2015 Remaster)"
    marker for the edition it sells, or -- on Bandcamp, which serves whatever the
    artist typed -- the "01-Artist-Title" filename itself. Compared on the raw title
    alone each of those reads as a track the library is missing.
    """
    variants = {name}
    for pattern in (_FEAT_SUFFIX, _RELEASE_SUFFIX, _LEADING_TRACK_NO):
        if stripped := pattern.sub("", name).strip():
            variants.add(stripped)
    # "01-Amestigon-Demiurg": a filename carrying its own artist field
    parts = name.split("-")
    if len(parts) >= 3 and parts[0].strip().isdigit() and (rest := "-".join(parts[2:]).strip()):
        variants.add(rest)
    return variants


def _names_same_track(provider_name: str, library_names: Iterable[str]) -> bool:
    """
    Return True if a provider track title names a track the library already holds.

    The exact lowercase compare this backs up matches only identical spellings, so a
    typographic apostrophe, an ellipsis character, a diacritic or "ft." against
    "feat." is enough to admit a second copy of a track that is already listed.
    """
    library_names = list(library_names)
    return any(
        compare_strings(variant, library_name, strict=True)
        or loose_compare_strings(variant, library_name)
        or compare_strings(variant, library_name, strict=False)
        for variant in _track_name_variants(provider_name)
        for library_name in library_names
    )


def _rating_bucket_clause(
    source: str, values: Iterable[float], param_prefix: str
) -> tuple[str, dict[str, Any]]:
    """
    Build the WHERE fragment for the given source's rating buckets.

    Selectors outside the source's vocabulary (see ``_CR_RATING_SPECS``) are dropped;
    each surviving selector N matches the half-open range [N, N + width) — one exact
    half-star step for AMG, a two-point band for TPS.
    """
    allowed, width = _CR_RATING_SPECS[source]
    valid = sorted({v for v in _coerce_float_list(values) if v in allowed})
    if not valid:
        return "", {}
    bucket_clauses: list[str] = []
    params: dict[str, Any] = {}
    for i, lo in enumerate(valid):
        lo_key, hi_key = f"{param_prefix}_lo_{i}", f"{param_prefix}_hi_{i}"
        params[lo_key] = lo
        params[hi_key] = lo + width
        bucket_clauses.append(
            f"(json_extract(value, '$.rating') >= :{lo_key} "
            f"AND json_extract(value, '$.rating') < :{hi_key})"
        )
    inner = " OR ".join(bucket_clauses)
    sub = _cr_source_exists(f"'{source}'", f" AND ({inner})")
    return sub, params


def _source_favorite_clause(source: str, param_prefix: str) -> tuple[str, dict[str, Any]]:
    """Match albums where the given source has favorite=true."""
    src_key = f"{param_prefix}_src"
    sub = _cr_source_exists(f":{src_key}", " AND json_extract(value, '$.favorite') = 1")
    return sub, {src_key: source}


def _source_accolades_clause(
    source: str, kinds: list[str], param_prefix: str
) -> tuple[str, dict[str, Any]]:
    """
    Match albums where the given source carries one of the requested accolade kinds.

    Each kind resolves (via ``_ACCOLADE_KIND_MATCH``) to either an exact match on a
    review-column accolade (``TYMHM``) or a LIKE-prefix match on a dated award accolade
    (``Album of the Year (2024)`` — the date is inlined, so the bare award name as a
    prefix covers every year/month variant). Both are matched against the source's
    merged ``accolades[]`` array. Unknown kinds are ignored.
    """
    src_key = f"{param_prefix}_src"
    params: dict[str, Any] = {src_key: source}
    conditions: list[str] = []
    for i, kind in enumerate(kinds):
        match = _ACCOLADE_KIND_MATCH.get(kind)
        if match is None:
            continue
        mode, value = match
        key = f"{param_prefix}_{i}"
        if mode == "prefix":
            params[key] = _like_prefix(value)
            conditions.append(f"accolade_each.value LIKE :{key} ESCAPE '\\'")
        else:
            params[key] = value
            conditions.append(f"accolade_each.value = :{key}")
    if not conditions:
        return "", {}
    cond_or = " OR ".join(conditions)
    sub = (
        f"EXISTS(SELECT 1 FROM {_CR_SOURCES_EACH} src "
        f"WHERE json_extract(src.value, '$.source') = :{src_key} "
        "AND EXISTS(SELECT 1 FROM "
        "json_each(json_extract(src.value, '$.accolades')) accolade_each "
        f"WHERE {cond_or}))"
    )
    return sub, params


def _source_untagged_clause(source: str, param_prefix: str) -> tuple[str, dict[str, Any]]:
    """Match albums that do not carry an entry for the given source."""
    src_key = f"{param_prefix}_src"
    sub = (
        f"NOT EXISTS(SELECT 1 FROM {_CR_SOURCES_EACH} "
        f"WHERE json_extract(value, '$.source') = :{src_key})"
    )
    return sub, {src_key: source}


def _apply_critical_reception_filters(  # noqa: PLR0913
    *,
    query_parts: list[str],
    query_params: dict[str, Any],
    dr_buckets: list[str] | None,
    amg_ratings: list[float | int] | None,
    amg_favorite: bool | None,
    amg_accolades: list[str] | None,
    amg_untagged: bool | None,
    tps_ratings: list[float | int] | None,
    tps_favorite: bool | None,
    tps_accolades: list[str] | None,
    tps_untagged: bool | None,
    match_mode: str = "all",
) -> None:
    """
    Append SQL clauses for critical_reception filters into the supplied lists.

    Each list-shaped filter OR-combines its values internally (a DR bucket selector
    matches if the album falls in any of the chosen buckets). The ``match_mode``
    parameter controls how the resulting top-level clauses combine with each other:

    - ``"all"`` (default): clauses are appended individually to ``query_parts`` so
      they AND with each other and with the rest of the surrounding query.
    - ``"any"``: clauses are collected locally and appended as one OR group so an
      album matches if it satisfies *any* active critical-reception filter, while
      the OR group itself still ANDs with non-reception filters (favorite, genre,
      provider, etc.). Falls back to "all" semantics when zero or one clause is
      active, since OR over a single clause is identical to AND.

    :param match_mode: ``"all"`` or ``"any"``; any other value is treated as ``"all"``.
    """
    # Collect clauses locally; we'll decide how to splice them into query_parts
    # based on match_mode at the end.
    local_parts: list[str] = []

    # DR buckets — combine into a single OR clause referencing one extracted value.
    # Prefers the canonical (measured) `$.dynamic_range` and falls back to the
    # AMG-review-reported `$.critical_reception.amg_dr` so an album with only the
    # review value still buckets the same way the on-cover badge displays it (the
    # badge does the same fallback in useAlbumTags / parseAlbumTags on the
    # frontend). `untagged` matches albums where both values are null.
    if dr_buckets:
        kinds = [b for b in dr_buckets if b in _DR_BUCKET_RANGES or b == "untagged"]
        if kinds:
            dr_path = (
                "COALESCE("
                f"{_DR_JSON}, "
                "json_extract(albums.metadata, '$.critical_reception.amg_dr')"
                ")"
            )
            or_parts: list[str] = []
            for i, kind in enumerate(kinds):
                if kind == "untagged":
                    or_parts.append(f"{dr_path} IS NULL")
                    continue
                lo, hi = _DR_BUCKET_RANGES[kind]
                lo_key, hi_key = f"dr_{kind}_lo_{i}", f"dr_{kind}_hi_{i}"
                query_params[lo_key] = lo
                if hi is None:
                    or_parts.append(f"{dr_path} >= :{lo_key}")
                else:
                    query_params[hi_key] = hi
                    or_parts.append(f"({dr_path} >= :{lo_key} AND {dr_path} < :{hi_key})")
            local_parts.append("(" + " OR ".join(or_parts) + ")")

    # Per-source review filters: rating buckets, the favorite flag, accolade-kind
    # filters (merged review-column types + award labels) and the untagged flag.
    # Both sources take the identical shape, so drive them from one tuple instead
    # of keeping a literal AMG/TPS twin of every block.
    for source, ratings, favorite, accolades_list, untagged, prefix in (
        ("AMG", amg_ratings, amg_favorite, amg_accolades, amg_untagged, "amg"),
        ("TPS", tps_ratings, tps_favorite, tps_accolades, tps_untagged, "tps"),
    ):
        if ratings:
            clause, params = _rating_bucket_clause(source, list(ratings), f"{prefix}_rb")
            if clause:
                local_parts.append(clause)
                query_params.update(params)
        if favorite:
            clause, params = _source_favorite_clause(source, f"{prefix}_fav")
            local_parts.append(clause)
            query_params.update(params)
        if accolades_list:
            clause, params = _source_accolades_clause(source, list(accolades_list), f"{prefix}_acc")
            if clause:
                local_parts.append(clause)
                query_params.update(params)
        if untagged:
            clause, params = _source_untagged_clause(source, f"{prefix}_unt")
            local_parts.append(clause)
            query_params.update(params)

    if not local_parts:
        return
    if match_mode == "any" and len(local_parts) > 1:
        # Single OR group ANDed against everything else in query_parts. Parenthesize
        # to keep the precedence intact when the caller joins with " AND ".
        query_parts.append("(" + " OR ".join(local_parts) + ")")
    else:
        query_parts.extend(local_parts)


def _apply_album_specific_filters(  # noqa: PLR0913
    *,
    query_parts: list[str],
    query_params: dict[str, Any],
    album_types: list[AlbumType] | None,
    listen_later: bool | None,
    dr_buckets: list[str] | None,
    amg_ratings: list[float | int] | None,
    amg_favorite: bool | None,
    amg_accolades: list[str] | None,
    amg_untagged: bool | None,
    tps_ratings: list[float | int] | None,
    tps_favorite: bool | None,
    tps_accolades: list[str] | None,
    tps_untagged: bool | None,
    match_mode: str,
) -> None:
    """
    Append the album-specific (album_type / listen_later / critical-reception) clauses.

    Shared by :meth:`AlbumsController.library_items` and
    :meth:`AlbumsController.library_count` so the listed rows and the displayed count
    apply identical album filters; only the search clause is handled per-method.
    """
    if album_types:
        query_parts.append("albums.album_type IN :album_types")
        query_params["album_types"] = [x.value for x in album_types]
    if listen_later is not None:
        # Scoped to the calling user: the shelf is per-user, so an unscoped query would
        # show everyone in the household everyone else's saves. `NOT (...)` rather than
        # a negated subquery so an album nobody saved still matches listen_later=False.
        query_parts.append(
            _LISTEN_LATER_MEMBER_SQL if listen_later else f"NOT ({_LISTEN_LATER_MEMBER_SQL})"
        )
        query_params["listen_later_userid"] = _current_userid()
    _apply_critical_reception_filters(
        query_parts=query_parts,
        query_params=query_params,
        dr_buckets=dr_buckets,
        amg_ratings=amg_ratings,
        amg_favorite=amg_favorite,
        amg_accolades=amg_accolades,
        amg_untagged=amg_untagged,
        tps_ratings=tps_ratings,
        tps_favorite=tps_favorite,
        tps_accolades=tps_accolades,
        tps_untagged=tps_untagged,
        match_mode=match_mode,
    )


@dataclass(slots=True)
class AlbumSyncDetails(LibraryItemSyncDetails):
    """
    Lightweight sync snapshot of a library album.

    Carries the enhanced-branch fields the provider sync loop needs to decide whether
    to refresh review data or demote a listen-later row, so it can do both without
    hydrating a full Album (which is what upstream's lightweight snapshot avoids).
    """

    listen_later: bool = False
    critical_reception: CriticalReception | None = None
    dynamic_range: float | None = None


class AlbumsController(MediaControllerBase[Album]):
    """Controller managing MediaItems of type Album."""

    db_table = DB_TABLE_ALBUMS
    media_type = MediaType.ALBUM
    item_cls = Album
    summary_item_cls = AlbumSummary
    # Sort keys that reference columns/JSON paths unique to the albums table.
    # `NULLS LAST` keeps unsaved listen-later rows out of the way; the dr/amg/tps
    # sorts target `albums.metadata` JSON fields that only exist on this table.
    extra_sort_keys: Mapping[str, str] = MappingProxyType(
        {
            # the *association's* timestamp, not albums.listen_later_added_at: that
            # column is the retired household-wide stamp, so sorting on it would order
            # one user's shelf by when somebody else saved the album.
            "listen_later_added_at": f"{_LISTEN_LATER_ADDED_AT_SQL} ASC NULLS LAST",
            "listen_later_added_at_desc": f"{_LISTEN_LATER_ADDED_AT_SQL} DESC NULLS LAST",
            # `dr` sorts on the canonical (measured) album dynamic range — not the
            # AMG-review-reported value, which lives at $.critical_reception.amg_dr.
            # NULLS LAST so albums without a measured DR don't float to the top of
            # an ASC sort (a long tail of un-analyzed albums would otherwise hide
            # the entries the user actually wants to see).
            "dr": f"{_DR_JSON} ASC NULLS LAST",
            "dr_desc": f"{_DR_JSON} DESC NULLS LAST",
            "amg_rating": _cr_rating_sort_key("AMG", "ASC"),
            "amg_rating_desc": _cr_rating_sort_key("AMG", "DESC"),
            "tps_rating": _cr_rating_sort_key("TPS", "ASC"),
            "tps_rating_desc": _cr_rating_sort_key("TPS", "DESC"),
        }
    )

    def __init__(self, mass: MusicAssistant) -> None:
        """Initialize class."""
        super().__init__(mass)
        # register (extra) api handlers
        api_base = self.api_base
        self.mass.register_api_command(
            f"music/{api_base}/album_tracks", self.tracks, required_scope=Scope.LIBRARY_READ
        )
        self.mass.register_api_command(
            f"music/{api_base}/album_versions", self.versions, required_scope=Scope.LIBRARY_READ
        )

    @property
    def base_query(self) -> tuple[str, dict[str, Any]]:
        """Return the base SELECT query for albums and its bound query params."""
        query = f"""
        SELECT
            {_LISTEN_LATER_FLAG_SQL} AS listen_later,
            {_LISTEN_LATER_ADDED_AT_SQL} AS listen_later_added_at,
            albums.*,
            {self._external_ids_query()} AS external_ids,
            {self._favorite_query()} AS favorite,
            {self._provider_mappings_query()} AS provider_mappings,
            (SELECT JSON_GROUP_ARRAY(
                json_object(
                'item_id', artists.item_id,
                'provider', 'library',
                    'name', artists.name,
                    'sort_name', artists.sort_name,
                    'media_type', 'artist'
                )) FROM artists JOIN album_artists on album_artists.album_id = albums.item_id  WHERE artists.item_id = album_artists.artist_id) AS artists
            FROM albums"""
        return query, {"listen_later_userid": _current_userid()}

    @property
    def summary_query(self) -> tuple[str, dict[str, Any]]:
        """Return the slim SELECT query used for album summary listings."""
        artists_query = self._artist_mappings_summary_query(DB_TABLE_ALBUM_ARTISTS, "album_id")
        query = f"""
        SELECT
            {self._summary_base_columns()},
            albums.version,
            albums.year,
            albums.album_type,
            {_LISTEN_LATER_FLAG_SQL} AS listen_later,
            {_LISTEN_LATER_ADDED_AT_SQL} AS listen_later_added_at,
            json_extract(albums.metadata, '$.critical_reception') AS critical_reception,
            {_DR_JSON} AS dynamic_range,
            {self._provider_mappings_query()} AS provider_mappings,
            {artists_query} AS artists
            FROM albums"""
        return query, {"listen_later_userid": _current_userid()}

    async def get(
        self,
        item_id: str,
        provider_instance_id_or_domain: str,
        allow_update_metadata: bool = True,
        recursive: bool = True,
    ) -> Album:
        """Return (full) details for a single media item."""
        album = await super().get(
            item_id,
            provider_instance_id_or_domain,
            allow_update_metadata=allow_update_metadata,
        )
        if not recursive:
            return album

        # append artist details to full album item (resolve ItemMappings)
        album_artists: UniqueList[Artist | ItemMapping] = UniqueList()
        for artist in album.artists:
            if not isinstance(artist, ItemMapping):
                album_artists.append(artist)
                continue
            with contextlib.suppress(MediaNotFoundError):
                album_artists.append(
                    await self.mass.music.artists.get(
                        artist.item_id, artist.provider, allow_update_metadata=False
                    )
                )
        album.artists = album_artists
        return album

    async def library_items(  # noqa: PLR0913
        self,
        favorite: bool | None = None,
        search: str | None = None,
        limit: int = 500,
        offset: int = 0,
        order_by: str = "sort_name",
        provider: str | list[str] | None = None,
        genre: int | list[int] | None = None,
        played_only: bool = False,
        album_types: list[AlbumType] | None = None,
        listen_later: bool | None = None,
        dr_buckets: list[str] | None = None,
        amg_ratings: list[float | int] | None = None,
        amg_favorite: bool | None = None,
        amg_accolades: list[str] | None = None,
        amg_untagged: bool | None = None,
        tps_ratings: list[float | int] | None = None,
        tps_favorite: bool | None = None,
        tps_accolades: list[str] | None = None,
        tps_untagged: bool | None = None,
        critical_reception_match: str = "all",
        *,
        summary: bool = True,
        reachable_via: list[str] | None = None,
        **kwargs: Any,
    ) -> list[Album]:
        """
        Get in-database albums.

        :param favorite: Only include the current user's likes (True) or dislikes (False).
        :param search: Filter by search query.
        :param limit: Maximum number of items to return.
        :param offset: Number of items to skip.
        :param order_by: Order by field (e.g. 'sort_name', 'timestamp_added').
        :param provider: Filter by provider instance ID (single string or list).
        :param album_types: Filter by album types.
        :param genre: Filter by genre id(s).
        :param summary: When True (default), return slim summary items containing only the
            fields needed for a list view. Set to False to get fully hydrated items.
        :param dr_buckets: Filter by DR quality bucket (excellent/good/fair/poor/untagged).
        :param amg_ratings / tps_ratings: Filter by review-source rating buckets
            (AMG: half stars 0.5..5.0, each selector one exact step; TPS: 1/3/5/7/9,
            each selector a two-point band on the /10 scale).
        :param amg_favorite / tps_favorite: Keep only entries flagged as favourite.
        :param amg_accolades / tps_accolades: Filter by accolade kind (aoty,
            record_of_the_month, honorable_mention, score_revised, tymhm, sitf,
            ymio, lit, rfu).
        :param amg_untagged / tps_untagged: Keep only albums missing that source.
        :param critical_reception_match: ``"all"`` (default) ANDs all DR/AMG/TPS clauses;
            ``"any"`` ORs them so an album matches if it satisfies at least one.
        :param reachable_via: Restrict results to items with a provider mapping reachable
            through one of these provider instance ids (OR semantics). See
            `MediaControllerBase.library_items` for the full semantics.
        """
        reachable_via = self._resolve_reachable_via(reachable_via)
        if reachable_via is not None and not reachable_via:
            return []
        extra_query_params: dict[str, Any] = {}
        extra_query_parts: list[str] = []
        extra_join_parts: list[str] = []
        artist_table_joined = False
        # album-specific filters (album_type / listen_later / critical_reception),
        # shared with library_count so count and list stay in sync.
        _apply_album_specific_filters(
            query_parts=extra_query_parts,
            query_params=extra_query_params,
            album_types=album_types,
            listen_later=listen_later,
            dr_buckets=dr_buckets,
            amg_ratings=amg_ratings,
            amg_favorite=amg_favorite,
            amg_accolades=amg_accolades,
            amg_untagged=amg_untagged,
            tps_ratings=tps_ratings,
            tps_favorite=tps_favorite,
            tps_accolades=tps_accolades,
            tps_untagged=tps_untagged,
            match_mode=critical_reception_match,
        )
        if order_by and "album_artist_name" in order_by:
            # join artist table to allow sorting on artist name
            extra_join_parts.append(
                "JOIN album_artists ON album_artists.album_id = albums.item_id "
                "JOIN artists ON artists.item_id = album_artists.artist_id "
            )
            artist_table_joined = True
        if search and " - " in search:
            # handle combined artist + title search
            artist_str, title_str = search.split(" - ", 1)
            search = None
            title_str = create_safe_string(title_str, True, True)
            artist_str = create_safe_string(artist_str, True, True)
            extra_query_parts.append(
                search_name_match_clause("albums", title_str, "search_title", extra_query_params)
            )
            artist_clause = "AND " + search_name_match_clause(
                "artists", artist_str, "search_artist", extra_query_params
            )
            # use join with artists table to filter on artist name
            extra_join_parts.append(
                "JOIN album_artists ON album_artists.album_id = albums.item_id "
                "JOIN artists ON artists.item_id = album_artists.artist_id " + artist_clause
                if not artist_table_joined
                else artist_clause
            )
            artist_table_joined = True
        result = await self.get_library_items_by_query(
            favorite=favorite,
            search=search,
            genre_ids=genre,
            limit=limit,
            offset=offset,
            order_by=order_by,
            provider_filter=self._provider_filter_considering_reachability(provider, reachable_via),
            extra_query_parts=extra_query_parts,
            extra_query_params=extra_query_params,
            extra_join_parts=extra_join_parts,
            played_only=played_only,
            # listen-later items intentionally don't flip in_library, so they'd
            # be filtered out by the standard library JOIN. Relax the filter
            # when this query is scoped to the listen-later list.
            in_library_only=listen_later is not True,
            summary=summary,
            reachable_via=reachable_via,
        )

        # Calculate how many more items we need to reach the original limit
        remaining_limit = limit - len(result)

        if (
            search
            and len(result) < _SEARCH_ARTIST_PASS_CUTOFF
            and not offset
            and remaining_limit > 0
        ):
            # append artist items to result
            search = create_safe_string(search, True, True)
            artist_clause = "AND " + search_name_match_clause(
                "artists", search, "search_artist", extra_query_params
            )
            extra_join_parts.append(
                "JOIN album_artists ON album_artists.album_id = albums.item_id "
                "JOIN artists ON artists.item_id = album_artists.artist_id " + artist_clause
                if not artist_table_joined
                else artist_clause
            )
            existing_uris = {item.uri for item in result}

            for album in await self.get_library_items_by_query(
                favorite=favorite,
                search=None,
                genre_ids=genre,
                limit=remaining_limit,
                order_by=order_by,
                provider_filter=self._provider_filter_considering_reachability(
                    provider, reachable_via
                ),
                extra_query_parts=extra_query_parts,
                extra_query_params=extra_query_params,
                extra_join_parts=extra_join_parts,
                in_library_only=listen_later is not True,
                summary=summary,
                reachable_via=reachable_via,
            ):
                # prevent duplicates (when artist is also in the title)
                if album.uri not in existing_uris:
                    result.append(album)
                    # Stop if we've reached the original limit
                    if len(result) >= limit:
                        break
        return result

    async def library_count(  # noqa: PLR0913
        self,
        favorite: bool | None = None,
        search: str | None = None,
        provider: str | list[str] | None = None,
        genre: int | list[int] | None = None,
        played_only: bool = False,
        album_types: list[AlbumType] | None = None,
        listen_later: bool | None = None,
        dr_buckets: list[str] | None = None,
        amg_ratings: list[float | int] | None = None,
        amg_favorite: bool | None = None,
        amg_accolades: list[str] | None = None,
        amg_untagged: bool | None = None,
        tps_ratings: list[float | int] | None = None,
        tps_favorite: bool | None = None,
        tps_accolades: list[str] | None = None,
        tps_untagged: bool | None = None,
        critical_reception_match: str = "all",
        # Caller's effective page-size hint. library_items's artist top-up only
        # fires on page 1 when page 1 has < cutoff hits AND remaining_limit > 0.
        # When the caller is paging at < cutoff items per page, the artist-only
        # delta is never visible — gating the union below on min(cutoff, limit)
        # keeps count parity with what the user actually sees.
        limit: int | None = None,
        # Accept the legacy `*_only` names so older clients keep working without
        # an immediate API contract bump. These are normalized into the boolean
        # filters below; new callers should use `favorite=` / `listen_later=`.
        favorite_only: bool = False,
        listen_later_only: bool = False,
        **kwargs: Any,
    ) -> int:
        """
        Return the total number of items in the library matching the filters.

        Filter args mirror :meth:`library_items` so count and list track each
        other 1-1 on any filtered view.
        """
        if favorite_only and favorite is None:
            favorite = True
        if listen_later_only and listen_later is None:
            listen_later = True
        # Build the non-search filter set once so we can apply it identically
        # to whichever search clause(s) the count requires below.
        base_params: dict[str, Any] = {}
        base_parts: list[str] = []
        base_joins: list[str] = []
        # in_library JOIN matches library_items: listen-later entries intentionally
        # don't flip in_library, so the JOIN is dropped when the caller is asking
        # for the listen-later subset.
        in_library_only = listen_later is not True
        self._apply_filters(
            query_parts=base_parts,
            query_params=base_params,
            favorite=favorite,
            search=None,
            genre_ids=self._preprocess_genre_ids(genre),
            provider_filter=self._ensure_provider_filter(provider),
            played_only=played_only,
            in_library_only=in_library_only,
        )
        _apply_album_specific_filters(
            query_parts=base_parts,
            query_params=base_params,
            album_types=album_types,
            listen_later=listen_later,
            dr_buckets=dr_buckets,
            amg_ratings=amg_ratings,
            amg_favorite=amg_favorite,
            amg_accolades=amg_accolades,
            amg_untagged=amg_untagged,
            tps_ratings=tps_ratings,
            tps_favorite=tps_favorite,
            tps_accolades=tps_accolades,
            tps_untagged=tps_untagged,
            match_mode=critical_reception_match,
        )

        async def _count(
            *,
            extra_parts: list[str] | None = None,
            extra_params_extra: dict[str, Any] | None = None,
            extra_joins: list[str] | None = None,
        ) -> int:
            parts = base_parts + (extra_parts or [])
            params = {**base_params, **(extra_params_extra or {})}
            joins = base_joins + (extra_joins or [])
            return await self._execute_count(parts, params, joins)

        # No search → single count with the base filters.
        if not search:
            return await _count()

        # "Artist - Album" mode: library_items splits on " - " and AND-joins title
        # against the album_artists table. There's no artist-pass fallback in this
        # branch, so the count is just the JOIN-filtered total.
        if " - " in search:
            artist_str, title_str = search.split(" - ", 1)
            title_safe = create_safe_string(title_str, True, True)
            artist_safe = create_safe_string(artist_str, True, True)
            # Reuse the shared matcher rather than hand-rolling LIKE: it picks the
            # FTS5 index or a LIKE scan depending on term length, and library_items
            # builds its clauses the same way — spelling it out here would desync
            # the count from the list the moment that helper changes.
            search_params: dict[str, Any] = {}
            title_clause = search_name_match_clause(
                "albums", title_safe, "search_title", search_params
            )
            artist_clause = search_name_match_clause(
                "artists", artist_safe, "search_artist", search_params
            )
            return await _count(
                extra_parts=[title_clause],
                extra_params_extra=search_params,
                extra_joins=[
                    "JOIN album_artists ON album_artists.album_id = albums.item_id "
                    "JOIN artists ON artists.item_id = album_artists.artist_id "
                    f"AND {artist_clause}"
                ],
            )

        # Bare-text mode: library_items pages return title-matches first, then
        # — only when the title-match result on page 1 fell under the cutoff —
        # top up with artist-only matches. So:
        #   • title_count >= cutoff → only title rows are ever visible.
        #     Counting the OR-union here would inflate by the artist-only delta.
        #   • title_count <  cutoff → page 1 includes the artist-only delta.
        #     |title ∪ artist| == |title| + |artist \ title|, which is exactly  # noqa: RUF003
        #     what the user sees, so the union count is right.
        search_safe = create_safe_string(search, True, True)
        bare_params: dict[str, Any] = {}
        title_clause = search_name_match_clause("albums", search_safe, "search", bare_params)
        title_count = await _count(
            extra_parts=[title_clause],
            extra_params_extra=bare_params,
        )
        # `limit` mirrors library_items' remaining_limit: the artist top-up never
        # fires once page 1 is already full of title hits, so a caller paging at
        # limit < cutoff never sees artist-only matches. Cap the cutoff to limit
        # so the count agrees with what the user actually scrolls through.
        effective_cutoff = (
            min(_SEARCH_ARTIST_PASS_CUTOFF, limit) if limit else _SEARCH_ARTIST_PASS_CUTOFF
        )
        if title_count >= effective_cutoff:
            return title_count
        union_params: dict[str, Any] = dict(bare_params)
        artist_clause = search_name_match_clause(
            "artists", search_safe, "search_artist", union_params
        )
        return await _count(
            extra_parts=[
                f"({title_clause} "
                "OR EXISTS(SELECT 1 FROM album_artists "
                "JOIN artists ON artists.item_id = album_artists.artist_id "
                "WHERE album_artists.album_id = albums.item_id "
                f"AND {artist_clause}))"
            ],
            extra_params_extra=union_params,
        )

    async def remove_item_from_library(self, item_id: str | int, recursive: bool = True) -> None:
        """Delete item from the library(database)."""
        db_id = int(item_id)  # ensure integer
        # recursively also remove album tracks
        for db_track in await self.get_library_album_tracks(db_id):
            if not recursive:
                raise MusicAssistantError("Album still has tracks linked")
            with contextlib.suppress(MediaNotFoundError):
                await self.mass.music.tracks.remove_item_from_library(db_track.item_id)
        # delete entry(s) from albumtracks table
        await self.mass.music.database.delete(DB_TABLE_ALBUM_TRACKS, {"album_id": db_id})
        # delete entry(s) from album artists table
        await self.mass.music.database.delete(DB_TABLE_ALBUM_ARTISTS, {"album_id": db_id})
        # delete this album from every user's listen-later shelf
        await self.mass.music.database.delete(DB_TABLE_ALBUM_LISTEN_LATER, {"item_id": db_id})
        # delete the album itself from db
        # this will raise if the item still has references and recursive is false
        await super().remove_item_from_library(item_id)

    async def set_listen_later(
        self, item_id: str | int, listen_later: bool, userid: str | None = None
    ) -> None:
        """
        Set the listen_later flag on a library album, for one user.

        Independent of `favorite` — this is the Roon-style "save for later" pile,
        not a library/favorites add. Stamps `added_at` with the current epoch on
        flip-to-true so the dedicated view can sort newest-first.

        The shelf is per-user: this writes a row in the album_listen_later
        association table, not the (retired) `albums.listen_later` column.

        :param item_id: Library album item_id (database id).
        :param listen_later: Whether the album should be on the user's shelf.
        :param userid: Whose shelf to write. Defaults to the calling user. It is a
            parameter rather than always-ambient so a caller with no request context
            (a maintenance script replaying the retired household shelf, a test) can be
            explicit; an ambient-only lookup would make those callers impossible to
            write without faking a request context.
        """
        if userid is None:
            user = get_current_user()
            if user is None:
                raise InsufficientPermissions("listen-later requires a signed-in user")
            userid = user.user_id
        db_id = int(item_id)
        if listen_later:
            await self.mass.music.database.insert_or_replace(
                DB_TABLE_ALBUM_LISTEN_LATER,
                {"item_id": db_id, "userid": userid, "added_at": int(utc_timestamp())},
            )
        else:
            await self.mass.music.database.delete(
                DB_TABLE_ALBUM_LISTEN_LATER, {"item_id": db_id, "userid": userid}
            )
        await self._signal_listen_later_change(db_id)

    async def clear_listen_later_for_all_users(self, item_id: str | int) -> None:
        """
        Take an album off every user's listen-later shelf.

        For the household-wide transitions, where the album stops being a "saved for
        later" candidate for everyone at once rather than for one person: it entered
        the library proper, or it is being deleted. Clearing only the calling user's
        row would leave the album on everyone else's shelf *and* in the library, which
        is exactly the both-views state the mutual exclusivity rule exists to prevent —
        and the sync loop that triggers it has no calling user at all.

        :param item_id: Library album item_id (database id).
        """
        await self.mass.music.database.delete(
            DB_TABLE_ALBUM_LISTEN_LATER, {"item_id": int(item_id)}
        )

    async def has_listen_later_anchor(self, item_id: str | int) -> bool:
        """
        Return whether the album sits on *any* user's listen-later shelf.

        Deliberately household-wide and independent of the calling user: this answers
        "does anybody still want this album", which is what the provider sync loop asks
        before deleting a row whose provider dropped it. Asking it per-user would let a
        background sync — which has no calling user, so reads an empty shelf — delete an
        album that another account has saved.

        :param item_id: Library album item_id (database id).
        """
        return bool(
            await self.mass.music.database.get_row(
                DB_TABLE_ALBUM_LISTEN_LATER, {"item_id": int(item_id)}
            )
        )

    async def listen_later_uris_all_users(self) -> set[str]:
        """
        Return the library uri of every album on any user's listen-later shelf.

        Household-wide, for the Discover row's refresh-debounce snapshot: that signal
        is broadcast to every client, so what it tracks is "did the shelf change for
        anyone", not "for me".
        """
        rows = await self.mass.music.database.get_rows_from_query(
            f"SELECT DISTINCT item_id FROM {DB_TABLE_ALBUM_LISTEN_LATER}", limit=0
        )
        return {f"library://album/{row['item_id']}" for row in rows}

    async def set_release_group(
        self,
        album_item_id: int,
        release_group_mbid: str,
    ) -> None:
        """
        Persist a MusicBrainz release-group ID on a library album, idempotently.

        :param album_item_id: Library album item_id (database id).
        :param release_group_mbid: MusicBrainz release-group UUID to set.
        """
        if not release_group_mbid:
            return
        try:
            album = await self.get_library_item(album_item_id)
        except MusicAssistantError as err:
            self.logger.debug("set_release_group: cannot load album %s: %s", album_item_id, err)
            return
        # Refuse to overwrite — keeps tag-sourced or already-enriched IDs authoritative.
        if album.get_external_id(ExternalID.MB_RELEASEGROUP):
            self.logger.debug(
                "set_release_group: album %s already has MB_RELEASEGROUP — keeping",
                album_item_id,
            )
            return
        album.add_external_id(ExternalID.MB_RELEASEGROUP, release_group_mbid)
        await self.update_item_in_library(album_item_id, album)
        self.logger.debug(
            "set_release_group: wrote %s onto album %s", release_group_mbid, album_item_id
        )

    async def tracks(
        self,
        item_id: str,
        provider_instance_id_or_domain: str,
        in_library_only: bool = False,
    ) -> list[Track]:
        """Return album tracks for the given provider album id."""
        # always check if we have a library item for this album
        library_album = await self.get_library_item_by_prov_id(
            item_id, provider_instance_id_or_domain
        )
        if not library_album:
            album_tracks = await self._get_provider_album_tracks(
                item_id, provider_instance_id_or_domain
            )
            await self._backfill_album_on_tracks(
                album_tracks, item_id, provider_instance_id_or_domain
            )
            return album_tracks

        # respect the current user's provider filter (if any) for both the
        # in-library tracks and the live provider fetches below
        allowed_providers = self._ensure_provider_filter(None)
        db_items = await self.get_library_album_tracks(
            library_album.item_id, provider_filter=allowed_providers
        )
        result: list[Track] = list(db_items)
        if in_library_only:
            # return in-library items only
            return sorted(db_items, key=lambda x: (x.disc_number, x.track_number))

        # return all (unique) items from all providers
        # because we are returning the items from all providers combined,
        # we need to make sure that we don't return duplicates
        unique_ids = self._album_track_unique_ids(db_items)
        library_names = [x.name for x in db_items]
        # where each provider track landed in the result, so a playable copy from another
        # provider can take the place of an unplayable one
        provider_slots: dict[str, int] = {}
        lookup_error: Exception | None = None
        for provider_mapping in library_album.provider_mappings:
            if not provider_mapping.available or (
                allowed_providers is not None
                and provider_mapping.provider_instance not in allowed_providers
            ):
                continue
            # an unavailable mapped instance hands the lookup to another account of the service
            own_instance = self.mass.get_provider(provider_mapping.provider_instance)
            own_lookup = (
                own_instance is not None
                and own_instance.instance_id == provider_mapping.provider_instance
            )
            provider_tracks, err = await self._album_tracks_from_provider(
                library_album, provider_mapping
            )
            if err is not None:
                # one provider that no longer lists the album must not hide the other
                # providers' tracks
                lookup_error = err
                if own_lookup and isinstance(err, MediaNotFoundError):
                    await self.mass.music.mark_provider_mapping_unavailable(
                        library_album, provider_mapping
                    )
                continue
            for provider_track in provider_tracks:
                # In some cases (looking at you YTM) the disc/track number is not obtained from
                # library_tracks. Ensure to update the disc/track number when interacting with
                # album tracks
                db_track = next(
                    (
                        x
                        for x in db_items
                        if x.sort_name == provider_track.sort_name
                        and x.version == provider_track.version
                    ),
                    None,
                )
                if (
                    db_track
                    and db_track.track_number == 0
                    and db_track.track_number != provider_track.track_number
                ):
                    await self._set_album_track(
                        db_id=int(library_album.item_id),
                        db_track_id=int(db_track.item_id),
                        track=provider_track,
                    )
                if provider_track.item_id in unique_ids:
                    continue
                if _track_position(provider_track) in unique_ids:
                    continue
                unique_id = f"{provider_track.name.lower()}.{provider_track.version.lower()}"
                slot = provider_slots.get(unique_id)
                if unique_id in unique_ids and (
                    slot is None or result[slot].available or not provider_track.available
                ):
                    continue
                if _names_same_track(provider_track.name, library_names):
                    continue
                unique_ids.add(unique_id)
                provider_track.album = library_album
                # always prefer album image
                album_images = [library_album.image] if library_album.image else []
                track_images: list[MediaItemImage] = provider_track.metadata.images or []
                provider_track.metadata.images = UniqueList(album_images + track_images)
                if slot is None:
                    provider_slots[unique_id] = len(result)
                    result.append(provider_track)
                else:
                    result[slot] = provider_track
        if lookup_error is not None and not any(track.available for track in result):
            # nothing could be played at all, so surface the reason instead of an empty list
            raise lookup_error
        # NOTE: we need to return the results sorted on disc/track here
        # to ensure the correct order at playback
        return sorted(result, key=lambda x: (x.disc_number, x.track_number))

    async def versions(
        self,
        item_id: str,
        provider_instance_id_or_domain: str,
    ) -> UniqueList[Album]:
        """Return all versions of an album we can find on all providers."""
        album = await self.get_provider_item(item_id, provider_instance_id_or_domain)
        streaming_search_query = (
            f"{album.artists[0].name} - {album.name}" if album.artists else album.name
        )
        result: UniqueList[Album] = UniqueList()
        for provider_id in self.mass.music.get_unique_providers():
            provider = self.mass.get_provider(provider_id)
            if not provider or not isinstance(provider, MusicProvider):
                continue
            if MediaType.ALBUM not in provider.supported_media_types:
                continue
            # TODO: filter by artists in db for non-streaming providers
            search_query = streaming_search_query if provider.is_streaming_provider else album.name
            # One failing provider must not break the whole lookup: the album page calls
            # this on every visit, so an unisolated raise here takes the page down instead
            # of just dropping that provider's versions. Providers reach third-party APIs
            # with an open-ended failure surface (Bandcamp's search intermittently answers
            # HTML, which surfaces as an aiohttp ContentTypeError), so this catches broadly
            # and logs at warning: the result is degraded but still useful.
            try:
                result.extend(
                    prov_item
                    for prov_item in await self.search(search_query, provider_id)
                    if loose_compare_strings(album.name, prov_item.name)
                    and compare_artists(prov_item.artists, album.artists, any_match=True)
                    # make sure that the 'base' version is NOT included
                    and not album.provider_mappings.intersection(prov_item.provider_mappings)
                )
                if ProviderFeature.ALBUM_VERSIONS in provider.supported_features:
                    # Call the specialized function in addition to searching to
                    # handle cases where the provider hasn't merged the album
                    # variants
                    if mapped_id := next(
                        (
                            p.item_id
                            for p in album.provider_mappings
                            if p.provider_instance == provider.instance_id
                        ),
                        None,
                    ):
                        result.extend(await provider.get_album_versions(mapped_id))
            except Exception:
                self.logger.warning(
                    "Failed to get album versions from provider %s", provider_id, exc_info=True
                )
        return result

    async def resolve_musicbrainz_release_group(
        self, release_group_id: str, allow_update_metadata: bool = True
    ) -> Album:
        """
        Return the album a MusicBrainz release group is, on one of the user's music providers.

        The group's likeliest official editions are tried in turn, and the album is the first
        one a music provider has, found through the links MusicBrainz keeps or, failing
        those, by the edition's barcode. An album already in the library is returned as the
        library album.

        :param release_group_id: MusicBrainz release group id.
        :param allow_update_metadata: Whether the album's metadata may be refreshed on the way.
        :raises ProviderUnavailableError: The MusicBrainz provider is not loaded.
        :raises MediaNotFoundError: None of the user's music providers has the album.
        """
        musicbrainz = cast("MusicbrainzProvider | None", self.mass.get_provider("musicbrainz"))
        if musicbrainz is None:
            raise ProviderUnavailableError("MusicBrainz is not available")
        # a heavily reissued album has more editions than one page holds, of which the
        # likeliest few are wanted rather than the whole set
        editions = [
            release
            for release in await musicbrainz.browse_releases_by_release_group(
                release_group_id, complete=False
            )
            if release.status == "Official"
        ]
        # the digital editions of one group carry different ids on the providers, so each
        # edition's own links and barcode are tried, never those of the whole group; the
        # editions linked to the user's own services go first, as only those resolve by link
        services = {source.domain for source in self.mass.music.providers if source.available}
        ranked = sorted(editions, key=lambda edition: _streaming_edition_rank(edition, services))
        for edition in ranked[:_MAX_EDITION_LOOKUPS]:
            try:
                release = await musicbrainz.get_release_details(edition.id)
            except InvalidDataError as err:
                # a stale edition costs nothing but its turn
                self.logger.debug("Release %s could not be looked up: %s", edition.id, err)
                continue
            # a link names one loaded instance of a service, while the user may only be
            # handed an album from a music source it may see
            linked = _within_sources(
                await provider_mappings_from_urls(
                    self.mass, relation_urls(release.relations), MediaType.ALBUM, set()
                ),
                self.mass.music.providers,
            )
            if album := await self._first_available_album(
                linked, release_group_id, allow_update_metadata
            ):
                return album
            # a barcode lookup fans out over every provider, so it is spent only when none
            # of the links resolves; a linked provider is asked too, as its link may be stale
            by_barcode = await self._album_candidates_by_barcode(release)
            if album := await self._first_available_album(
                by_barcode, release_group_id, allow_update_metadata
            ):
                return album
        msg = f"Release group {release_group_id} is not available on any music provider"
        raise MediaNotFoundError(
            msg,
            translation_key="album_not_available_on_music_services",
            translation_owner=self.translation_owner,
        )

    async def get_library_album_tracks(
        self,
        item_id: str | int,
        provider_filter: list[str] | None = None,
    ) -> list[Track]:
        """
        Return in-database album tracks for the given database album.

        :param item_id: The library item ID of the album.
        :param provider_filter: Optional provider instance ID(s) to limit the result to.
        """
        db_id = int(item_id)  # ensure integer
        # pass the album id as preferred album so the track_album subquery in the
        # base query returns this album's disc/track numbers for tracks that
        # appear on multiple albums
        return await self.mass.music.tracks.get_library_items_by_query(
            provider_filter=provider_filter,
            extra_query_parts=[
                f"tracks.item_id IN (SELECT track_id FROM {DB_TABLE_ALBUM_TRACKS} "
                "WHERE album_id = :album_id)"
            ],
            extra_query_params={"album_id": db_id, "preferred_album_id": db_id},
        )

    async def link_album_tracks(
        self,
        album: Album,
        db_tracks: Sequence[Track],
        release: MusicBrainzRelease | None,
        *,
        link_providers: bool = True,
    ) -> None:
        """
        Carry an album's MusicBrainz identity and provider links over to its library tracks.

        The library tracks at a matching position on the release get its recording id and
        ISRCs. For each provider the album is mapped to but one or more of its library
        tracks are not, the provider's tracklist is matched to the library tracks by ISRC
        or position and the matching provider tracks are linked.

        :param album: The library album.
        :param db_tracks: The album's library tracks.
        :param release: The album's MusicBrainz release, if it was identified.
        :param link_providers: Whether to link the tracks to the album's providers too.
        """
        if not db_tracks:
            return
        # one tracklist per streaming service, its first mapping standing for it; a local
        # server pushes its own track mappings when it syncs, so its tracklist is left alone
        album_mappings: dict[str, ProviderMapping] = {}
        for mapping in sorted(
            album.provider_mappings,
            key=lambda x: (x.provider_domain, x.provider_instance, x.item_id),
        ):
            provider = self.mass.get_provider(
                mapping.provider_instance, provider_type=MusicProvider
            )
            if provider is None or not provider.is_streaming_provider:
                continue
            album_mappings.setdefault(mapping.provider_domain, mapping)
        track_domains = [
            {x.provider_domain for x in track.provider_mappings} for track in db_tracks
        ]
        async with self.mass.music.database.deferred_commit():
            if release is not None:
                await self._link_tracks_to_release(db_tracks, release)
            if not link_providers:
                return
            for domain, mapping in album_mappings.items():
                if all(domain in domains for domains in track_domains):
                    continue
                await self._link_tracks_to_provider_album(db_tracks, mapping)

    async def add_item_mapping_as_album_to_library(self, item: ItemMapping) -> Album:
        """
        Add an ItemMapping as an Album to the library.

        This is only used in special occasions as is basically adds an album
        to the db without a lot of mandatory data, such as artists.
        """
        album = self.album_from_item_mapping(item)
        return await self.add_item_to_library(album)

    async def match_provider(
        self, db_album: Album, provider: MusicProvider, strict: bool = True
    ) -> list[ProviderMapping]:
        """
        Try to find a match on the given (streaming) provider for a (database) album.

        Links albums of different providers/qualities together. A provider that supports
        barcode lookups is asked for the album by barcode before it is searched. Sparse
        provider search results only rule out a confident non-match; a candidate that
        still looks ambiguous is confirmed against the full provider album, its tracklist
        and, as a last resort, MusicBrainz before its provider mapping is accepted.
        """
        return await self._match_provider(db_album, provider, strict, _BaseTracksMemo())

    async def match_providers(self, db_album: Album) -> None:
        """
        Try to find match on all (streaming) providers for the provided (database) album.

        This is used to link objects of different providers/qualities together.
        """
        if db_album.provider != "library":
            return  # Matching only supported for database items
        if not db_album.artists:
            return  # guard

        # resolve the base tracklist at most once for the whole match operation
        base_tracks_memo = _BaseTracksMemo()
        # try to find match on all providers
        cur_provider_domains = {
            x.provider_domain for x in db_album.provider_mappings if x.available
        }
        # the links MusicBrainz keeps name the album on the other providers outright, so
        # those providers are linked first and only the remaining ones are searched
        if musicbrainz := self._musicbrainz_link_provider():
            cur_provider_domains |= await self._link_musicbrainz_entity(
                db_album, lambda: musicbrainz.resolve_release(db_album)
            )
        for provider in self.mass.music.providers:
            if provider.domain in cur_provider_domains:
                continue
            if ProviderFeature.SEARCH not in provider.supported_features:
                continue
            if MediaType.ALBUM not in provider.supported_media_types:
                continue
            if not provider.is_streaming_provider:
                # matching on unique providers is pointless as they push (all) their content to MA
                continue
            if match := await self._match_provider(db_album, provider, True, base_tracks_memo):
                # 100% match, we update the db with the additional provider mapping(s)
                await self.add_provider_mappings(db_album.item_id, match)
                cur_provider_domains.add(provider.domain)

    def album_from_item_mapping(self, item: ItemMapping) -> Album:
        """Create an Album object from an ItemMapping object."""
        # a library item mapping references an item already in the library, which has no
        # mapping to itself, so only a resolvable provider yields a real provider mapping
        provider_mappings: list[dict[str, Any]] = []
        if prov := self.mass.get_provider(item.provider):
            provider_mappings.append(
                {
                    "item_id": item.item_id,
                    "provider_domain": prov.domain,
                    "provider_instance": prov.instance_id,
                    "available": item.available,
                }
            )
        return Album.from_dict({**item.to_dict(), "provider_mappings": provider_mappings})

    async def _add_library_item(self, item: Album, overwrite_existing: bool = False) -> int:
        """Add a new record to the database."""
        if not isinstance(item, Album):  # TODO: Remove this once the codebase is fully typed
            msg = "Not a valid Album object (ItemMapping can not be added to db)"  # type: ignore[unreachable]
            raise InvalidDataError(msg)
        db_id = await self.mass.music.database.insert(
            self.db_table,
            {
                "name": item.name,
                "sort_name": item.sort_name,
                "version": item.version,
                "album_type": item.album_type,
                "year": item.year,
                "metadata": serialize_to_json(item.metadata),
                "search_name": create_safe_string(item.name, True, True),
                "search_sort_name": create_safe_string(item.sort_name or "", True, True),
                "timestamp_added": int(item.date_added.timestamp()) if item.date_added else UNSET,
            },
        )
        # update/set external id lookup table
        await self.set_external_ids(db_id, item.external_ids)
        # update/set provider_mappings table
        await self.set_provider_mappings(db_id, item.provider_mappings)
        # set track artist(s)
        await self._set_album_artists(db_id, item.artists)
        self.logger.debug("added %s to database (id: %s)", item.name, db_id)
        return db_id

    async def _update_library_item(
        self, item_id: str | int, update: Album, overwrite: bool = False
    ) -> None:
        """Update existing record in the database."""
        db_id = int(item_id)  # ensure integer
        cur_item = await self.get_library_item(db_id)
        # 2.10 factored the stored-vs-incoming decision into metadata_for_update, which
        # also fixes the case this branch used to only paper over: an overwrite carrying
        # a bare stub (providers embed those in track payloads) now merges instead of
        # wiping. The critical-reception arbitration still has to run on top of it --
        # model.update() deep-merges CR into a superset, but a wholesale overwrite still
        # replaces a rich stored CR with an empty one, silently losing the
        # filesystem-tag-derived value. Judge whatever the helper returned against the
        # stored CR and put the stored one back only when the incoming one regressed, so
        # a purely additive update (a newly surfaced TPS source) still survives.
        if cur_item.metadata is None:
            # `metadata` has a default_factory so this should never happen, but the
            # stored_cr read below already guarded for it — keep the two consistent
            # instead of guarding one line and dereferencing on the next.
            cur_item.metadata = MediaItemMetadata()
        stored_cr = cur_item.metadata.critical_reception
        metadata = metadata_for_update(cur_item.metadata, update.metadata, overwrite)
        if update.metadata is not None and not critical_reception_is_richer(
            metadata.critical_reception, stored_cr
        ):
            metadata.critical_reception = stored_cr
        if getattr(update, "album_type", AlbumType.UNKNOWN) != AlbumType.UNKNOWN:
            album_type = update.album_type
        else:
            album_type = cur_item.album_type
        cur_item.external_ids.update(update.external_ids)
        name = update.name if overwrite else cur_item.name
        sort_name = update.sort_name if overwrite else cur_item.sort_name or update.sort_name
        authoritative = AUTHORITATIVE_REFRESH.get() is update
        if overwrite and authoritative:
            # a re-tagged release can lose its edition, which only the files can tell us
            version = update.version
        elif overwrite:
            version = update.version or cur_item.version
        else:
            version = cur_item.version or update.version
        await self.mass.music.database.update(
            self.db_table,
            {"item_id": db_id},
            {
                "name": name,
                "sort_name": sort_name,
                "version": version,
                "year": (update.year or cur_item.year)
                if overwrite
                else (cur_item.year or update.year),
                "album_type": album_type.value,
                "metadata": serialize_to_json(metadata),
                "search_name": create_safe_string(name, True, True),
                "search_sort_name": create_safe_string(sort_name or "", True, True),
                "timestamp_added": int(update.date_added.timestamp())
                if update.date_added
                else UNSET,
            },
        )
        # update/set external id lookup table
        await self.set_external_ids(
            db_id, update.external_ids if overwrite else cur_item.external_ids
        )
        # update/set provider_mappings table
        if overwrite:
            # 2.10's helper is better than what this branch had here: it replaces only
            # the mappings of the providers the incoming item comes from, so a moved
            # file drops its stale row without discarding the mappings other providers
            # wrote for the same album.
            provider_mappings: Iterable[ProviderMapping] = provider_mappings_for_update(
                cur_item.provider_mappings, update.provider_mappings, overwrite
            )
        else:
            # The helper's merge branch is a plain set union, which keeps *both* rows of
            # a mapping that differs only in in_library and leaves the loser to chance.
            # Merge by (provider_instance, item_id) instead. Seed from the update side so
            # refreshed fields (available, url, audio_format, details, is_unique) actually
            # land in the DB; layer cur_item's mappings on top for keys update doesn't
            # know about, and promote in_library=True from cur on collision so a
            # concurrent in_library flip can never get downgraded to False by an update
            # payload that hasn't seen it yet.
            merged: dict[tuple[str, str], ProviderMapping] = {
                (pm.provider_instance, pm.item_id): pm for pm in update.provider_mappings
            }
            for pm in cur_item.provider_mappings:
                key = (pm.provider_instance, pm.item_id)
                existing_pm = merged.get(key)
                if existing_pm is None:
                    merged[key] = pm
                elif pm.in_library and not existing_pm.in_library:
                    # Promote in_library without mutating the caller-supplied `update`
                    # mapping in place (these objects are seeded by reference from
                    # update.provider_mappings); replace with a copy instead.
                    merged[key] = replace(existing_pm, in_library=True)
            provider_mappings = list(merged.values())
        await self.set_provider_mappings(db_id, provider_mappings, overwrite)
        # Library membership and listen-later are mutually exclusive: once an album
        # gains an in_library mapping it graduates out of the listen-later pile, so a
        # saved album that's later added to the library (or synced in) never lingers in
        # both. New rows default listen_later=0, so this only matters on update.
        if any(pm.in_library for pm in provider_mappings):
            # Clear listen-later inline rather than via set_listen_later(): the
            # surrounding update_item_in_library/add_item_to_library re-reads the row
            # and emits a single MEDIA_ITEM_UPDATED once the whole update (incl. the
            # artists set below) has landed. Calling set_listen_later here would add
            # two redundant re-reads plus a premature event for a half-updated row.
            #
            # Unconditional now, and household-wide: `cur_item.listen_later` only ever
            # reported the *calling* user's shelf, so gating on it would leave the album
            # on every other account's shelf while it sits in the library. The delete is
            # a no-op when no shelf row exists, which is the common case.
            await self.clear_listen_later_for_all_users(db_id)
        # set album artist(s)
        artists = update.artists if overwrite else cur_item.artists + update.artists
        await self._set_album_artists(db_id, artists, overwrite=overwrite)
        self.logger.debug("updated %s in database: (id %s)", update.name, db_id)

    async def _get_provider_album_tracks(
        self, item_id: str, provider_instance_id_or_domain: str
    ) -> list[Track]:
        """Return album tracks for the given provider album id."""
        if prov := self.mass.get_provider(provider_instance_id_or_domain):
            prov = cast("MusicProvider", prov)
            return await prov.get_album_tracks(item_id)
        return []

    async def _album_tracks_from_provider(
        self, library_album: Album, provider_mapping: ProviderMapping
    ) -> tuple[list[Track], Exception | None]:
        """
        Return one provider's listing for a library album, plus the error if it failed.

        A failed provider yields no tracks and its error, so the caller can skip it and
        still raise when no provider lists the album at all.

        :param library_album: The library album being listed, used for logging.
        :param provider_mapping: The album's mapping on the provider to fetch from.
        """
        try:
            tracks = await self._get_provider_album_tracks(
                provider_mapping.item_id, provider_mapping.provider_instance
            )
        except LoginFailed:
            # an account problem is the user's to fix, not a listing to skip
            raise
        except _SKIPPED_ALBUM_LISTING_ERRORS as err:
            # a mapping outlives what it points at: a subsonic server reissues its
            # ids on a rescan, a streaming release is delisted. losing that one
            # provider's listing is the whole cost -- raising here instead fails
            # the album outright, including the tracks held in the library.
            self.logger.warning(
                "Unable to fetch tracks for album %s from provider %s: %s",
                library_album.name,
                provider_mapping.provider_instance,
                err,
            )
            return [], err
        except Exception as err:
            # same cost, but the provider did not wrap its failure. A provider talking to
            # a third-party API can leak a raw client error -- Bandcamp answers HTML
            # instead of JSON and aiohttp raises ContentTypeError, which is no
            # MusicAssistantError and so sailed straight past the arm above, emptying the
            # whole album for someone whose copy is sitting right there in the library.
            # Louder than the expected case: an unwrapped error is a bug in that provider
            # (or here), not ordinary staleness.
            self.logger.warning(
                "album %s: skipping album tracks from %s (unwrapped provider error)",
                library_album.name,
                provider_mapping.provider_instance,
                exc_info=True,
            )
            return [], err
        return tracks, None

    async def _backfill_album_on_tracks(
        self, album_tracks: list[Track], item_id: str, provider_instance_id_or_domain: str
    ) -> None:
        """
        Fill in the parent album and its image on provider album tracks that omit them.

        :param album_tracks: The album tracks as listed by the provider.
        :param item_id: The provider album id.
        :param provider_instance_id_or_domain: The provider the album tracks come from.
        """
        # some album-track listings omit the parent album and its image; backfill both
        # from the provider album so the queue shows the album name and artwork.
        if not album_tracks or (album_tracks[0].album and album_tracks[0].image):
            return
        prov_album = await self.get_provider_item(item_id, provider_instance_id_or_domain)
        album_mapping = ItemMapping.from_item(prov_album)
        for track in album_tracks:
            if prov_album.image and not track.image:
                track.metadata.add_image(prov_album.image)
            if track.album is None:
                track.album = album_mapping

    async def _verify_musicbrainz_mapping(self, mapping: ProviderMapping) -> bool:
        """Return True if a linked album exists on the provider, checked for Apple Music only."""
        # MusicBrainz links Apple Music albums per storefront, so a linked album may not
        # exist in the user's storefront; the other providers' catalogs are worldwide
        if mapping.provider_domain != "apple_music":
            return True
        try:
            await self.get_provider_item(
                mapping.item_id, mapping.provider_instance, allow_fallback=False
            )
        except MusicAssistantError, aiohttp.ClientError, TimeoutError:
            return False
        return True

    async def _link_tracks_to_release(
        self, db_tracks: Sequence[Track], release: MusicBrainzRelease
    ) -> None:
        """Fill the recording ids and ISRCs of a release in on the library tracks at its positions."""
        recordings = {
            (medium.position, track.position): track.recording
            for medium in release.media
            for track in medium.tracks
            if track.position and track.recording
        }
        for db_track in db_tracks:
            # a digital release stores its single disc as disc 0 or 1
            recording = recordings.get((db_track.disc_number or 1, db_track.track_number))
            if recording is None or not _recording_matches_track(recording, db_track):
                continue
            changed = fill_track_from_recording(db_track, recording)
            if not changed and db_track.metadata.last_musicbrainz_lookup is not None:
                continue
            db_track.metadata.last_musicbrainz_lookup = int(time())
            await self.mass.music.tracks.update_item_in_library(db_track.item_id, db_track)

    async def _link_tracks_to_provider_album(
        self, db_tracks: Sequence[Track], mapping: ProviderMapping
    ) -> None:
        """Link the tracks of a provider album to the library tracks they are."""
        try:
            provider_tracks = await self._get_provider_album_tracks(
                mapping.item_id, mapping.provider_instance
            )
        except _ALBUM_TRACK_LOOKUP_ERRORS as err:
            self.logger.debug(
                "Album tracks unavailable for %s on %s: %s",
                mapping.item_id,
                mapping.provider_instance,
                err,
            )
            return
        for db_track in db_tracks:
            if any(
                x.provider_domain == mapping.provider_domain for x in db_track.provider_mappings
            ):
                continue
            if provider_track := _matching_provider_track(db_track, provider_tracks):
                await self.mass.music.tracks.add_unclaimed_provider_mappings(
                    db_track.item_id, provider_track.provider_mappings
                )

    @staticmethod
    def _album_track_unique_ids(db_items: Iterable[Track]) -> set[str]:
        """Return the identifiers by which provider album tracks are matched to library tracks."""
        unique_ids: set[str] = {_track_position(x) for x in db_items}
        unique_ids.update({f"{x.name.lower()}.{x.version.lower()}" for x in db_items})
        for db_item in db_items:
            unique_ids.update(x.item_id for x in db_item.provider_mappings)
        return unique_ids

    def _library_match_names(self, item: Album | ItemMapping) -> list[str]:
        """Return the normalized album names, with and without a spelled-out retail suffix."""
        base_name = create_safe_string(strip_album_retail_suffix(item.name), True, True)
        return [base_name, *(f"{base_name}{suffix}" for suffix in ALBUM_RETAIL_SUFFIX_KEYS)]

    async def _confirm_library_candidate(self, db_item: Album, item: Album | ItemMapping) -> bool:
        """
        Return True if a library album is the same album as the one being added.

        An edition that cannot be decided on the albums' own metadata is escalated to
        tracklists and MusicBrainz, so an ambiguous album is linked to the album it
        belongs to instead of becoming a second library entry.
        """
        if not isinstance(item, Album):
            return await super()._confirm_library_candidate(db_item, item)
        evidence = compare_album_evidence(db_item, item, strict=True)
        if evidence != AlbumMatchEvidence.INSUFFICIENT:
            return evidence == AlbumMatchEvidence.MATCH
        provider = self.mass.get_provider(item.provider, provider_type=MusicProvider)
        if provider is None or provider.instance_id != item.provider:
            # only the exact provider instance the album came from may be fingerprinted,
            # never a same-domain fallback pointing at a different account/server
            return False
        evidence = await self._resolve_album_evidence(
            db_item, item, provider, True, _BaseTracksMemo()
        )
        return evidence == AlbumMatchEvidence.MATCH

    async def _match_provider(
        self,
        db_album: Album,
        provider: MusicProvider,
        strict: bool,
        base_tracks_memo: _BaseTracksMemo,
    ) -> list[ProviderMapping]:
        """Match one provider by barcode, then by search, and return the confirmed mappings."""
        self.logger.debug("Trying to match album %s on provider %s", db_album.name, provider.name)
        matches: list[ProviderMapping] = []
        if ProviderFeature.ALBUM_BY_EXTERNAL_ID in provider.supported_features:
            matches = await self._match_provider_by_barcode(
                db_album, provider, strict, base_tracks_memo
            )
        # a barcode hit makes the search unnecessary
        search_results: list[Album] = []
        if not matches:
            search_str = (
                f"{db_album.artists[0].name} - {db_album.name}"
                if db_album.artists
                else db_album.name
            )
            search_results = await self.search(search_str, provider.instance_id)
        for search_result_item in search_results:
            if not search_result_item.available:
                continue
            # a sparse search result only rules out a confident non-match; a MATCH or an
            # ambiguous (INSUFFICIENT) candidate is confirmed against the full album below
            if (
                compare_album_evidence(db_album, search_result_item, strict=strict)
                == AlbumMatchEvidence.NO_MATCH
            ):
                continue
            # search results can be simplified objects, so fetch the full provider album
            prov_album = await self.get_provider_item(
                search_result_item.item_id,
                search_result_item.provider,
                fallback=search_result_item,
            )
            evidence = await self._resolve_album_evidence(
                db_album, prov_album, provider, strict, base_tracks_memo
            )
            if evidence == AlbumMatchEvidence.MATCH:
                matches.extend(prov_album.provider_mappings)
        if not matches:
            self.logger.debug(
                "Could not find match for Album %s on provider %s",
                db_album.name,
                provider.name,
            )
        return matches

    async def _match_provider_by_barcode(
        self,
        db_album: Album,
        provider: MusicProvider,
        strict: bool,
        base_tracks_memo: _BaseTracksMemo,
    ) -> list[ProviderMapping]:
        """Return the mappings of the provider album one of the base album's barcodes resolves to."""
        # the order only makes the choice of looked-up barcodes deterministic
        for barcode in sorted(_canonical_album_barcodes(db_album))[:MAX_EXTERNAL_ID_MATCH_LOOKUPS]:
            try:
                prov_album = await provider.get_album_by_external_id(barcode, ExternalID.BARCODE)
                if prov_album is None:
                    continue
                # a lookup result can be a simplified object, so fetch the full provider album
                prov_album = await self.get_provider_item(
                    prov_album.item_id, prov_album.provider, fallback=prov_album
                )
            except EXTERNAL_ID_LOOKUP_ERRORS as err:
                self.logger.debug(
                    "Barcode %s lookup on provider %s failed: %s", barcode, provider.name, err
                )
                continue
            if not prov_album.available:
                continue
            # the queried barcode is the query, not evidence: it is left out of the scored
            # copy so name, artist, year and the tracklist decide, while a second,
            # independently agreeing barcode still counts
            candidate = replace(
                prov_album,
                external_ids={
                    (kind, value)
                    for kind, value in prov_album.external_ids
                    if not (kind == ExternalID.BARCODE and barcode_to_upc(value) == barcode)
                },
            )
            evidence = await self._resolve_album_evidence(
                db_album, candidate, provider, strict, base_tracks_memo
            )
            if evidence == AlbumMatchEvidence.MATCH:
                return list(prov_album.provider_mappings)
        return []

    async def _album_candidates_by_barcode(
        self, release: MusicBrainzRelease
    ) -> list[ProviderMapping]:
        """
        Return the mappings of the albums the music providers find by a release's barcode.

        Only an album that is the release, by title and primary artist, counts.

        :param release: The MusicBrainz release, with its barcode and artist credits.
        """
        if not release.barcode or not is_valid_barcode(release.barcode):
            return []
        upc = barcode_to_upc(release.barcode)
        providers = [
            provider
            for provider in self.mass.music.providers
            if provider.supports_feature(ProviderFeature.ALBUM_BY_EXTERNAL_ID)
        ]
        hits = await asyncio.gather(
            *(self._album_by_barcode(provider, upc) for provider in providers)
        )
        candidates: list[ProviderMapping] = []
        for provider, hit in zip(providers, hits, strict=True):
            if hit is None:
                continue
            # a barcode gets reused, and a provider answers with the first album carrying it
            if not release_matches_album(release, hit):
                self.logger.debug(
                    "Barcode %s on provider %s is album %s, not %s",
                    upc,
                    provider.name,
                    hit.name,
                    release.title,
                )
                continue
            candidates.extend(hit.provider_mappings)
        return candidates

    async def _album_by_barcode(self, provider: MusicProvider, upc: str) -> Album | None:
        """Return the album a provider finds by a barcode, if it has one and answers."""
        try:
            return await provider.get_album_by_external_id(upc, ExternalID.BARCODE)
        except EXTERNAL_ID_LOOKUP_ERRORS as err:
            self.logger.debug(
                "Barcode %s lookup on provider %s failed: %s", upc, provider.name, err
            )
            return None

    async def _first_available_album(
        self,
        candidates: Sequence[ProviderMapping],
        release_group_id: str,
        allow_update_metadata: bool,
    ) -> Album | None:
        """Return the first candidate its provider still serves as an album, if any."""
        for candidate in candidates:
            try:
                if library_album := await self.get_library_item_by_prov_id(
                    candidate.item_id, candidate.provider_instance
                ):
                    return await self.get(
                        library_album.item_id,
                        "library",
                        allow_update_metadata=allow_update_metadata,
                    )
                # the candidate names one of the user's own sources, so an unavailable one
                # must not fall back to another account of the same service
                return await self.get_provider_item(
                    candidate.item_id,
                    candidate.provider_instance,
                    allow_fallback=False,
                    strict_provider_instance=True,
                )
            except EXTERNAL_ID_LOOKUP_ERRORS as err:
                self.logger.debug(
                    "Release group %s is not available as album %s on %s: %s",
                    release_group_id,
                    candidate.item_id,
                    candidate.provider_instance,
                    err,
                )
        return None

    async def _resolve_album_evidence(
        self,
        db_album: Album,
        prov_album: Album,
        provider: MusicProvider,
        strict: bool,
        base_tracks_memo: _BaseTracksMemo,
    ) -> AlbumMatchEvidence:
        """
        Return the match evidence for a fully-fetched provider album.

        An ambiguous album is escalated to ordered track fingerprints and, only if those
        stay inconclusive, to MusicBrainz; a mapping is accepted only on a MATCH.

        :param provider: The exact provider instance the candidate album was matched on;
            its tracklist is fetched directly so a same-domain fallback can never
            fingerprint the candidate against a different account/server.
        """
        evidence = compare_album_evidence(db_album, prov_album, strict=strict)
        if evidence != AlbumMatchEvidence.INSUFFICIENT:
            return evidence
        # ambiguous metadata: resolve conservatively with ordered track fingerprints
        base_tracks = await self._resolve_base_album_tracks(db_album, base_tracks_memo)
        try:
            compare_tracks = await provider.get_album_tracks(prov_album.item_id)
        except _ALBUM_TRACK_LOOKUP_ERRORS as err:
            # the candidate tracklist is unavailable: treat it as absent and let MusicBrainz decide
            self.logger.debug(
                "Album tracks unavailable for %s on %s: %s",
                prov_album.item_id,
                provider.instance_id,
                err,
            )
            compare_tracks = []
        evidence = compare_album_evidence(
            db_album,
            prov_album,
            strict=strict,
            base_tracks=base_tracks,
            compare_tracks=compare_tracks,
        )
        if evidence != AlbumMatchEvidence.INSUFFICIENT:
            return evidence
        # tracklists could not resolve it either: consult MusicBrainz as a last resort
        return await self._musicbrainz_album_evidence(db_album, prov_album)

    async def _resolve_base_album_tracks(
        self, db_album: Album, base_tracks_memo: _BaseTracksMemo
    ) -> list[Track] | None:
        """Return the memoized base tracklist, resolving it once on first use."""
        if not base_tracks_memo.resolved:
            base_tracks_memo.tracks = await self._load_base_album_tracks(db_album)
            base_tracks_memo.resolved = True
        return base_tracks_memo.tracks

    async def _load_base_album_tracks(self, db_album: Album) -> list[Track] | None:
        """
        Return a complete, ordered base tracklist to fingerprint against.

        Iterates the album's existing provider mappings in a deterministic order and
        returns the first loaded provider's full tracklist whose disc/track positions can
        be trusted. A provider-sourced tracklist is used rather than the stored library
        tracks because those can be an incomplete subset (individually added tracks), and
        an incomplete base would make a track-count difference look like a real conflict.
        """
        for mapping in sorted(
            db_album.provider_mappings,
            key=lambda mapping: (
                mapping.provider_domain,
                mapping.provider_instance,
                mapping.item_id,
            ),
        ):
            if not mapping.available:
                continue
            provider = self.mass.get_provider(mapping.provider_instance, return_unavailable=True)
            if (
                provider is None
                or provider.instance_id != mapping.provider_instance
                or not provider.available
            ):
                # only trust the exact, currently-available provider instance and never a
                # same-domain fallback pointing at a different account/server
                continue
            try:
                provider_tracks = await self._get_provider_album_tracks(
                    mapping.item_id, mapping.provider_instance
                )
            except _ALBUM_TRACK_LOOKUP_ERRORS as err:
                # this mapping's tracklist is unavailable: try the next existing mapping
                self.logger.debug(
                    "Base album tracks unavailable for %s on %s: %s",
                    mapping.item_id,
                    mapping.provider_instance,
                    err,
                )
                continue
            if album_tracks_have_positions(provider_tracks):
                return provider_tracks
        return None

    async def _musicbrainz_album_evidence(
        self, base_album: Album, compare_album: Album
    ) -> AlbumMatchEvidence:
        """
        Return album match evidence from MusicBrainz release identity, or abstain.

        A barcode that resolves unambiguously to a single specific MusicBrainz release on
        both albums is strong positive evidence; barcodes belonging to entirely different
        release groups are negative. A barcode resolving to several releases, a shared
        release group alone, an unresolved barcode or a lookup failure abstains
        (INSUFFICIENT) rather than guessing.
        """
        base_barcodes = _canonical_album_barcodes(base_album)
        compare_barcodes = _canonical_album_barcodes(compare_album)
        if not base_barcodes or not compare_barcodes:
            return AlbumMatchEvidence.INSUFFICIENT
        musicbrainz = self.mass.get_provider("musicbrainz")
        if musicbrainz is None:
            return AlbumMatchEvidence.INSUFFICIENT
        musicbrainz = cast("MusicbrainzProvider", musicbrainz)
        releases_by_barcode: dict[str, list[MusicBrainzBarcodeRelease]] = {}
        try:
            for barcode in sorted(base_barcodes | compare_barcodes):
                releases_by_barcode[barcode] = await musicbrainz.get_releases_by_barcode(barcode)
        except (RetriesExhausted, InvalidDataError, TimeoutError, aiohttp.ClientError) as err:
            self.logger.debug(
                "MusicBrainz barcode lookup failed while matching album %s: %s",
                base_album.name,
                err,
            )
            return AlbumMatchEvidence.INSUFFICIENT
        base_release_ids = _unambiguous_release_ids(base_barcodes, releases_by_barcode)
        compare_release_ids = _unambiguous_release_ids(compare_barcodes, releases_by_barcode)
        if base_release_ids & compare_release_ids:
            # both albums carry a barcode that names the same single specific release
            return AlbumMatchEvidence.MATCH
        if not all(releases_by_barcode[barcode] for barcode in base_barcodes | compare_barcodes):
            # an unresolved barcode leaves the release-group sets incomplete, so a disjoint
            # comparison could wrongly reject regional equivalents: abstain instead
            return AlbumMatchEvidence.INSUFFICIENT
        base_group_ids = _release_group_ids(base_barcodes, releases_by_barcode)
        compare_group_ids = _release_group_ids(compare_barcodes, releases_by_barcode)
        if base_group_ids.isdisjoint(compare_group_ids):
            # the barcodes belong to entirely different release groups: different albums
            return AlbumMatchEvidence.NO_MATCH
        # a shared release group alone (or an ambiguous barcode) never identifies an edition
        return AlbumMatchEvidence.INSUFFICIENT

    async def _set_album_artists(
        self,
        db_id: int,
        artists: Iterable[Artist | ItemMapping],
        overwrite: bool = False,
    ) -> None:
        """
        Store Album Artists.

        An empty set of artists never clears the stored rows: an album that lost its
        artists disappears from their discography and is skipped by provider matching.
        """
        all_artists = list(artists)
        if not all_artists:
            if overwrite:
                # a caller asking to replace all artists with none is a bug,
                # so keep the stored rows and make the attempt visible
                self.logger.warning("Ignoring request to clear all artists of album id %s", db_id)
            return
        if overwrite:
            # on overwrite, clear the album_artists table first
            await self.mass.music.database.delete(
                DB_TABLE_ALBUM_ARTISTS,
                {
                    "album_id": db_id,
                },
            )
        for artist in all_artists:
            await self._set_album_artist(db_id, artist=artist, overwrite=overwrite)

    async def _set_album_artist(
        self, db_id: int, artist: Artist | ItemMapping, overwrite: bool = False
    ) -> ItemMapping:
        """Store Album Artist info."""
        db_artist: Artist | ItemMapping | None = None
        if artist.provider == "library":
            db_artist = artist
        elif existing := await self.mass.music.artists.get_library_item_by_prov_id(
            artist.item_id, artist.provider
        ):
            db_artist = existing

        if not db_artist or overwrite:
            # Convert ItemMapping to Artist if needed
            artist_to_add = (
                self.mass.music.artists.artist_from_item_mapping(artist)
                if isinstance(artist, ItemMapping)
                else artist
            )
            db_artist = await self.mass.music.artists.add_item_to_library(
                artist_to_add, overwrite_existing=overwrite
            )
        # write (or update) record in album_artists table
        await self.mass.music.database.insert_or_replace(
            DB_TABLE_ALBUM_ARTISTS,
            {
                "album_id": db_id,
                "artist_id": int(db_artist.item_id),
            },
        )
        return ItemMapping.from_item(db_artist)

    async def _signal_listen_later_change(self, db_id: int) -> None:
        """
        Re-read the album and announce the change, mirroring _set_flag_columns.

        The shelf write lands in an association table rather than on the item row, so
        it cannot go through _set_flag_columns — but the same re-read / signal_event
        contract has to hold, or the Discover row never learns the shelf changed.

        :param db_id: Library album item_id (database id) that was just written.
        """
        library_item = await self.get_library_item(db_id)
        self.mass.signal_event(EventType.MEDIA_ITEM_UPDATED, library_item.uri, library_item)

    async def _merge_library_item_references(self, target_id: int, source_id: int) -> None:
        """Move every user's listen-later entry for the merged album onto the target."""
        await self.mass.music.database.execute_write(
            f"INSERT OR IGNORE INTO {DB_TABLE_ALBUM_LISTEN_LATER}(item_id, userid, added_at) "
            f"SELECT :target_id, userid, added_at FROM {DB_TABLE_ALBUM_LISTEN_LATER} "
            "WHERE item_id = :source_id",
            {"target_id": target_id, "source_id": source_id},
        )
        await self.mass.music.database.delete(DB_TABLE_ALBUM_LISTEN_LATER, {"item_id": source_id})

    async def _set_album_track(self, db_id: int, db_track_id: int, track: Track) -> None:
        """Store Album Track info."""
        # write (or update) record in album_tracks table
        await self.mass.music.database.insert_or_replace(
            DB_TABLE_ALBUM_TRACKS,
            {
                "album_id": db_id,
                "track_id": db_track_id,
                "track_number": track.track_number,
                "disc_number": track.disc_number,
            },
        )

    def _sync_details_query_parts(self) -> tuple[str, str, dict[str, Any]]:
        """Return extra (columns, joins, params) for the albums sync-details query."""
        # the sync loop needs the listen-later flag plus the review/DR metadata to
        # decide whether anything actually changed, without hydrating a full Album
        # household-wide EXISTS, not the calling user's shelf: the sync loop runs as a
        # background task with no calling user, and the demotion it drives takes the
        # album off *everyone's* shelf. A per-user read here would always be False and
        # the demotion would never fire.
        extra_columns = f"""
            , EXISTS(SELECT 1 FROM {DB_TABLE_ALBUM_LISTEN_LATER}
                WHERE item_id = {DB_TABLE_ALBUMS}.item_id) AS listen_later
            , json_extract({DB_TABLE_ALBUMS}.metadata, '$.critical_reception')
                AS critical_reception
            , {_DR_JSON} AS dynamic_range
        """
        return extra_columns, "", {}

    def _parse_sync_details_row(self, db_row: Mapping[str, Any]) -> AlbumSyncDetails:
        """Parse a raw sync-details db row into an AlbumSyncDetails object."""
        raw_cr = db_row["critical_reception"]
        return AlbumSyncDetails(
            item_id=db_row["item_id"],
            date_added=datetime.fromtimestamp(db_row["timestamp_added"], tz=UTC),
            provider_mappings=self._parse_sync_details_mappings(db_row),
            listen_later=bool(db_row["listen_later"]),
            critical_reception=CriticalReception.from_dict(json_loads(raw_cr)) if raw_cr else None,
            dynamic_range=db_row["dynamic_range"],
        )

    def _parse_summary_row(self, db_row: Mapping[str, Any]) -> AlbumSummary:
        """Parse a raw summary db row into an AlbumSummary object."""
        item = cast("AlbumSummary", super()._parse_summary_row(db_row))
        item.version = db_row["version"] or ""
        item.year = db_row["year"]
        item.album_type = AlbumType(db_row["album_type"])
        item.artists = self._parse_summary_artist_mappings(db_row)
        # listen_later is an enhanced-branch column and summary rows are built
        # field-by-field rather than via _parse_db_row, so it has to be carried
        # across explicitly - otherwise every list view (summary=True is the
        # default) would report the flag as unset.
        item.listen_later = bool(db_row["listen_later"])
        item.listen_later_added_at = db_row["listen_later_added_at"]
        # The slim metadata built by the (final) base helper carries only the thumb,
        # so the DR value and review data the list view renders have to be attached
        # here or every summary row would come back without them.
        if raw_cr := db_row["critical_reception"]:
            item.metadata.critical_reception = CriticalReception.from_dict(json_loads(raw_cr))
        item.metadata.dynamic_range = db_row["dynamic_range"]
        return item


def _canonical_album_barcodes(album: Album) -> set[str]:
    """Return an album's valid barcodes in canonical UPC form."""
    return {
        barcode_to_upc(value)
        for external_id_type, value in album.external_ids
        if external_id_type == ExternalID.BARCODE and is_valid_barcode(value)
    }


def _streaming_edition_rank(
    release: MusicBrainzBarcodeRelease, services: set[str]
) -> tuple[bool, bool, bool, bool, str]:
    """
    Return the sort key ranking a group's official editions, the one the user's services carry first.

    :param release: The edition as the release group browse lists it.
    :param services: The domains of the music services the user may see.
    """
    linked = {
        service
        for url in relation_urls(release.relations)
        if (service := share_url_provider(url, MediaType.ALBUM))
    }
    return (
        not linked & services,
        not is_digital_release(release),
        not linked,
        release.country not in ("XW", "XE"),
        release.date or "9999",
    )


def _within_sources(
    candidates: Iterable[ProviderMapping], sources: Sequence[MusicProvider]
) -> list[ProviderMapping]:
    """
    Return the candidates that name one of the given music sources.

    A candidate stays on its own instance when that is an available source; otherwise it
    moves to the first available instance the sources have of its service, failing that to
    the first there is. One on a service they do not include is left out.
    """
    available = {source.instance_id for source in sources if source.available}
    first_by_domain: dict[str, str] = {}
    for source in sorted(sources, key=lambda source: not source.available):
        first_by_domain.setdefault(source.domain, source.instance_id)
    within: list[ProviderMapping] = []
    for candidate in candidates:
        if candidate.provider_instance in available:
            within.append(candidate)
        elif instance := first_by_domain.get(candidate.provider_domain):
            within.append(replace(candidate, provider_instance=instance))
    return within


def _unambiguous_release_ids(
    barcodes: set[str], releases_by_barcode: dict[str, list[MusicBrainzBarcodeRelease]]
) -> set[str]:
    """Return release ids that at least one of the barcodes resolves to unambiguously."""
    release_ids: set[str] = set()
    for barcode in barcodes:
        resolved = {release.id for release in releases_by_barcode.get(barcode, [])}
        # only a barcode that maps to exactly one specific release is trustworthy evidence
        if len(resolved) == 1:
            release_ids |= resolved
    return release_ids


def _release_group_ids(
    barcodes: set[str], releases_by_barcode: dict[str, list[MusicBrainzBarcodeRelease]]
) -> set[str]:
    """Return every release-group id the barcodes resolve to."""
    return {
        release.release_group.id
        for barcode in barcodes
        for release in releases_by_barcode.get(barcode, [])
    }


def _recording_matches_track(recording: MusicBrainzRecording, track: Track) -> bool:
    """Return whether a release's recording is the given library track, by title and length."""
    if not compare_strings(recording.title, track.name, strict=False):
        return False
    if recording.length is None or not track.duration:
        return True
    return abs(recording.length / 1000 - track.duration) <= _TRACK_DURATION_TOLERANCE


def _matching_provider_track(db_track: Track, provider_tracks: Sequence[Track]) -> Track | None:
    """Return the provider track that is the given library track: by ISRC, else by position."""
    isrcs = _isrcs(db_track)
    if isrcs:
        # an ISRC is occasionally reused, so the durations must agree as well
        for provider_track in provider_tracks:
            if isrcs & _isrcs(provider_track) and _durations_agree(db_track, provider_track):
                return provider_track
    if not db_track.track_number:
        return None
    # a digital release stores its single disc as disc 0 or 1
    position = (db_track.disc_number or 1, db_track.track_number)
    for provider_track in provider_tracks:
        if (
            (provider_track.disc_number or 1, provider_track.track_number) == position
            and compare_strings(provider_track.name, db_track.name, strict=False)
            and _durations_agree(db_track, provider_track)
        ):
            return provider_track
    return None


def _durations_agree(track: Track, other: Track) -> bool:
    """Return whether two tracks' durations are within tolerance, an unknown duration passing."""
    if not track.duration or not other.duration:
        return True
    return abs(track.duration - other.duration) <= _TRACK_DURATION_TOLERANCE


def _isrcs(track: Track) -> set[str]:
    """Return a track's valid ISRCs in canonical form."""
    return {
        normalize_external_id(ExternalID.ISRC, value)
        for id_type, value in track.external_ids
        if id_type == ExternalID.ISRC and is_valid_isrc(value)
    }
