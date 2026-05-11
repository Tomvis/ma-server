"""Manage MediaItems of type Album."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

from music_assistant_models.enums import AlbumType, EventType, MediaType, ProviderFeature
from music_assistant_models.errors import InvalidDataError, MediaNotFoundError, MusicAssistantError
from music_assistant_models.media_items import (
    Album,
    Artist,
    ItemMapping,
    MediaItemImage,
    ProviderMapping,
    Track,
    UniqueList,
)

from music_assistant.constants import DB_TABLE_ALBUM_ARTISTS, DB_TABLE_ALBUM_TRACKS, DB_TABLE_ALBUMS
from music_assistant.controllers.media.base import MediaControllerBase
from music_assistant.controllers.webserver.helpers.auth_middleware import get_current_user
from music_assistant.helpers.compare import (
    compare_album,
    compare_artists,
    compare_media_item,
    create_safe_string,
    loose_compare_strings,
)
from music_assistant.helpers.database import UNSET
from music_assistant.helpers.json import serialize_to_json
from music_assistant.models.music_provider import MusicProvider

if TYPE_CHECKING:
    from music_assistant import MusicAssistant


def _critical_reception_is_richer(new: object, existing: object) -> bool:
    """True if `new` carries strictly more critical_reception data than `existing`.

    Coarse "more fields filled" heuristic: a forward improvement (AMG DR landing,
    additional sources) wins; a regression (transient probe failure that drops
    sources) is rejected. The canonical (measured) DR has moved off CriticalReception
    onto MediaItemMetadata.dynamic_range, so it isn't part of this comparison.
    """
    if new is None:
        return False
    if existing is None:
        return True
    new_amg_dr = getattr(new, "amg_dr", None)
    cur_amg_dr = getattr(existing, "amg_dr", None)
    if new_amg_dr is not None and cur_amg_dr is None:
        return True
    new_sources = getattr(new, "sources", None) or []
    cur_sources = getattr(existing, "sources", None) or []
    return len(new_sources) > len(cur_sources)


# DR quality thresholds (mirrors src/helpers/album_tags.ts on the frontend).
_DR_BUCKET_RANGES: dict[str, tuple[float, float | None]] = {
    "excellent": (14, None),
    "good": (10, 14),
    "fair": (7, 10),
    "poor": (0, 7),
}

# Map normalized label kind -> SQL pattern matched against each label string in the
# JSON labels[] array. Year/month suffixes on AOTY/AOTM/HONORABLE_MENTION are matched
# with LIKE so the same kind covers every annual variant.
_LABEL_KIND_PATTERNS: dict[str, str] = {
    "aoty": "AOTY-%",
    "aotm": "AOTM-%",
    "honorable_mention": "HONORABLE_MENTION-%",
    "record_of_the_month": "RECORD_OF_THE_MONTH",
}


def _amg_rating_bucket_clause(values: list[int], param_prefix: str) -> tuple[str, dict[str, Any]]:
    """Build the WHERE fragment for AMG rating buckets (1..5; floor(rating) == N)."""
    int_values = sorted({int(v) for v in values if 1 <= int(v) <= 5})
    if not int_values:
        return "", {}
    params = {f"{param_prefix}_{i}": v for i, v in enumerate(int_values)}
    bind_list = ", ".join(f":{k}" for k in params)
    sub = (
        "EXISTS(SELECT 1 FROM json_each(albums.metadata, '$.critical_reception.sources') "
        "WHERE json_extract(value, '$.source') = 'AMG' "
        f"AND CAST(json_extract(value, '$.rating') AS INTEGER) IN ({bind_list}))"
    )
    return sub, params


def _tps_rating_bucket_clause(values: list[int], param_prefix: str) -> tuple[str, dict[str, Any]]:
    """Build WHERE fragment for TPS rating buckets (selectors 1,3,5,7,9 → [N, N+2))."""
    valid = sorted({int(v) for v in values if int(v) in (1, 3, 5, 7, 9)})
    if not valid:
        return "", {}
    bucket_clauses: list[str] = []
    params: dict[str, Any] = {}
    for i, lo in enumerate(valid):
        lo_key, hi_key = f"{param_prefix}_lo_{i}", f"{param_prefix}_hi_{i}"
        params[lo_key] = lo
        params[hi_key] = lo + 2
        bucket_clauses.append(
            f"(json_extract(value, '$.rating') >= :{lo_key} "
            f"AND json_extract(value, '$.rating') < :{hi_key})"
        )
    inner = " OR ".join(bucket_clauses)
    sub = (
        "EXISTS(SELECT 1 FROM json_each(albums.metadata, '$.critical_reception.sources') "
        f"WHERE json_extract(value, '$.source') = 'TPS' AND ({inner}))"
    )
    return sub, params


def _source_favorite_clause(source: str, param_prefix: str) -> tuple[str, dict[str, Any]]:
    """Match albums where the given source has favorite=true."""
    src_key = f"{param_prefix}_src"
    sub = (
        "EXISTS(SELECT 1 FROM json_each(albums.metadata, '$.critical_reception.sources') "
        f"WHERE json_extract(value, '$.source') = :{src_key} "
        "AND json_extract(value, '$.favorite') = 1)"
    )
    return sub, {src_key: source}


def _source_labels_clause(
    source: str, kinds: list[str], param_prefix: str
) -> tuple[str, dict[str, Any]]:
    """Match albums where the given source carries one of the requested label kinds."""
    patterns = [_LABEL_KIND_PATTERNS[k] for k in kinds if k in _LABEL_KIND_PATTERNS]
    if not patterns:
        return "", {}
    src_key = f"{param_prefix}_src"
    params: dict[str, Any] = {src_key: source}
    label_keys: list[str] = []
    for i, p in enumerate(patterns):
        k = f"{param_prefix}_{i}"
        params[k] = p
        label_keys.append(k)
    label_or = " OR ".join(f"label_each.value LIKE :{k}" for k in label_keys)
    sub = (
        "EXISTS(SELECT 1 FROM json_each(albums.metadata, '$.critical_reception.sources') src "
        f"WHERE json_extract(src.value, '$.source') = :{src_key} "
        "AND EXISTS(SELECT 1 FROM json_each(json_extract(src.value, '$.labels')) label_each "
        f"WHERE {label_or}))"
    )
    return sub, params


def _source_untagged_clause(source: str, param_prefix: str) -> tuple[str, dict[str, Any]]:
    """Match albums that do not carry an entry for the given source."""
    src_key = f"{param_prefix}_src"
    sub = (
        "NOT EXISTS(SELECT 1 FROM json_each(albums.metadata, '$.critical_reception.sources') "
        f"WHERE json_extract(value, '$.source') = :{src_key})"
    )
    return sub, {src_key: source}


def _apply_critical_reception_filters(
    *,
    query_parts: list[str],
    query_params: dict[str, Any],
    dr_buckets: list[str] | None,
    amg_ratings: list[int] | None,
    amg_favorite: bool | None,
    amg_labels: list[str] | None,
    amg_untagged: bool | None,
    tps_ratings: list[int] | None,
    tps_favorite: bool | None,
    tps_labels: list[str] | None,
    tps_untagged: bool | None,
) -> None:
    """Append SQL clauses for critical_reception filters into the supplied lists.

    All filters are independent; multiple filters AND together. Each list-shaped filter
    OR-combines its values internally (a DR bucket selector matches if the album falls
    in any of the chosen buckets).
    """
    # DR buckets — combine into a single OR clause referencing one extracted value.
    # Filters on the canonical (measured) album dynamic range; the AMG-review-reported
    # value at $.critical_reception.amg_dr is intentionally not part of this filter.
    if dr_buckets:
        kinds = [b for b in dr_buckets if b in _DR_BUCKET_RANGES or b == "untagged"]
        if kinds:
            dr_path = "json_extract(albums.metadata, '$.dynamic_range')"
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
            query_parts.append("(" + " OR ".join(or_parts) + ")")

    # AMG / TPS rating buckets
    if amg_ratings:
        clause, params = _amg_rating_bucket_clause(list(amg_ratings), "amg_rb")
        if clause:
            query_parts.append(clause)
            query_params.update(params)
    if tps_ratings:
        clause, params = _tps_rating_bucket_clause(list(tps_ratings), "tps_rb")
        if clause:
            query_parts.append(clause)
            query_params.update(params)

    # AMG / TPS favorite flag
    if amg_favorite:
        clause, params = _source_favorite_clause("AMG", "amg_fav")
        query_parts.append(clause)
        query_params.update(params)
    if tps_favorite:
        clause, params = _source_favorite_clause("TPS", "tps_fav")
        query_parts.append(clause)
        query_params.update(params)

    # AMG / TPS label-kind filters
    for source, labels_list, prefix in (
        ("AMG", amg_labels, "amg_lbl"),
        ("TPS", tps_labels, "tps_lbl"),
    ):
        if labels_list:
            clause, params = _source_labels_clause(source, list(labels_list), prefix)
            if clause:
                query_parts.append(clause)
                query_params.update(params)

    # untagged-source flags
    if amg_untagged:
        clause, params = _source_untagged_clause("AMG", "amg_unt")
        query_parts.append(clause)
        query_params.update(params)
    if tps_untagged:
        clause, params = _source_untagged_clause("TPS", "tps_unt")
        query_parts.append(clause)
        query_params.update(params)


class AlbumsController(MediaControllerBase[Album]):
    """Controller managing MediaItems of type Album."""

    db_table = DB_TABLE_ALBUMS
    media_type = MediaType.ALBUM
    item_cls = Album

    def __init__(self, mass: MusicAssistant) -> None:
        """Initialize class."""
        super().__init__(mass)
        self.base_query = """
        SELECT
            albums.*,
            (SELECT JSON_GROUP_ARRAY(
                json_object(
                'item_id', album_pm.provider_item_id,
                    'provider_domain', album_pm.provider_domain,
                        'provider_instance', album_pm.provider_instance,
                        'available', album_pm.available,
                        'audio_format', json(album_pm.audio_format),
                        'url', album_pm.url,
                        'details', album_pm.details,
                        'in_library', album_pm.in_library,
                        'is_unique', album_pm.is_unique
                )) FROM provider_mappings album_pm WHERE album_pm.item_id = albums.item_id AND album_pm.media_type = 'album') AS provider_mappings,
            (SELECT JSON_GROUP_ARRAY(
                json_object(
                'item_id', artists.item_id,
                'provider', 'library',
                    'name', artists.name,
                    'sort_name', artists.sort_name,
                    'media_type', 'artist'
                )) FROM artists JOIN album_artists on album_artists.album_id = albums.item_id  WHERE artists.item_id = album_artists.artist_id) AS artists
            FROM albums"""
        # register (extra) api handlers
        api_base = self.api_base
        self.mass.register_api_command(f"music/{api_base}/album_tracks", self.tracks)
        self.mass.register_api_command(f"music/{api_base}/album_versions", self.versions)

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

    async def library_items(
        self,
        favorite: bool | None = None,
        search: str | None = None,
        limit: int = 500,
        offset: int = 0,
        order_by: str = "sort_name",
        provider: str | list[str] | None = None,
        genre: int | list[int] | None = None,
        album_types: list[AlbumType] | None = None,
        listen_later: bool | None = None,
        dr_buckets: list[str] | None = None,
        amg_ratings: list[int] | None = None,
        amg_favorite: bool | None = None,
        amg_labels: list[str] | None = None,
        amg_untagged: bool | None = None,
        tps_ratings: list[int] | None = None,
        tps_favorite: bool | None = None,
        tps_labels: list[str] | None = None,
        tps_untagged: bool | None = None,
        **kwargs: Any,
    ) -> list[Album]:
        """Get in-database albums.

        :param favorite: Filter by favorite status.
        :param search: Filter by search query.
        :param limit: Maximum number of items to return.
        :param offset: Number of items to skip.
        :param order_by: Order by field (e.g. 'sort_name', 'timestamp_added').
        :param provider: Filter by provider instance ID (single string or list).
        :param album_types: Filter by album types.
        :param genre: Filter by genre id(s).
        :param dr_buckets: Filter by DR quality bucket (excellent/good/fair/poor/untagged).
        :param amg_ratings / tps_ratings: Filter by review-source rating buckets.
        :param amg_favorite / tps_favorite: Keep only entries flagged as favourite.
        :param amg_labels / tps_labels: Filter by accolade label kind.
        :param amg_untagged / tps_untagged: Keep only albums missing that source.
        """
        extra_query_params: dict[str, Any] = {}
        extra_query_parts: list[str] = []
        extra_join_parts: list[str] = []
        artist_table_joined = False
        # optional album type filter
        if album_types:
            extra_query_parts.append("albums.album_type IN :album_types")
            extra_query_params["album_types"] = [x.value for x in album_types]
        # listen_later filter — bool, applies to albums.listen_later directly
        if listen_later is not None:
            extra_query_parts.append("albums.listen_later = :listen_later_flag")
            extra_query_params["listen_later_flag"] = listen_later
        # optional critical_reception filters (DR + AMG/TPS source entries)
        _apply_critical_reception_filters(
            query_parts=extra_query_parts,
            query_params=extra_query_params,
            dr_buckets=dr_buckets,
            amg_ratings=amg_ratings,
            amg_favorite=amg_favorite,
            amg_labels=amg_labels,
            amg_untagged=amg_untagged,
            tps_ratings=tps_ratings,
            tps_favorite=tps_favorite,
            tps_labels=tps_labels,
            tps_untagged=tps_untagged,
        )
        if order_by and "artist_name" in order_by:
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
            extra_query_parts.append("albums.search_name LIKE :search_title")
            extra_query_params["search_title"] = f"%{title_str}%"
            # use join with artists table to filter on artist name
            extra_join_parts.append(
                "JOIN album_artists ON album_artists.album_id = albums.item_id "
                "JOIN artists ON artists.item_id = album_artists.artist_id "
                "AND artists.search_name LIKE :search_artist"
                if not artist_table_joined
                else "AND artists.search_name LIKE :search_artist"
            )
            artist_table_joined = True
            extra_query_params["search_artist"] = f"%{artist_str}%"
        result = await self.get_library_items_by_query(
            favorite=favorite,
            search=search,
            genre_ids=genre,
            limit=limit,
            offset=offset,
            order_by=order_by,
            provider_filter=self._ensure_provider_filter(provider),
            extra_query_parts=extra_query_parts,
            extra_query_params=extra_query_params,
            extra_join_parts=extra_join_parts,
            # listen-later items intentionally don't flip in_library, so they'd
            # be filtered out by the standard library JOIN. Relax the filter
            # when this query is scoped to the listen-later list.
            in_library_only=listen_later is not True,
        )

        # Calculate how many more items we need to reach the original limit
        remaining_limit = limit - len(result)

        if search and len(result) < 25 and not offset and remaining_limit > 0:
            # append artist items to result
            search = create_safe_string(search, True, True)
            extra_join_parts.append(
                "JOIN album_artists ON album_artists.album_id = albums.item_id "
                "JOIN artists ON artists.item_id = album_artists.artist_id "
                "AND artists.search_name LIKE :search_artist"
                if not artist_table_joined
                else "AND artists.search_name LIKE :search_artist"
            )
            extra_query_params["search_artist"] = f"%{search}%"
            existing_uris = {item.uri for item in result}

            for album in await self.get_library_items_by_query(
                favorite=favorite,
                search=None,
                limit=remaining_limit,
                order_by=order_by,
                provider_filter=self._ensure_provider_filter(provider),
                extra_query_parts=extra_query_parts,
                extra_query_params=extra_query_params,
                extra_join_parts=extra_join_parts,
                in_library_only=listen_later is not True,
            ):
                # prevent duplicates (when artist is also in the title)
                if album.uri not in existing_uris:
                    result.append(album)
                    # Stop if we've reached the original limit
                    if len(result) >= limit:
                        break
        return result

    async def library_count(
        self,
        favorite_only: bool = False,
        album_types: list[AlbumType] | None = None,
        listen_later_only: bool = False,
        dr_buckets: list[str] | None = None,
        amg_ratings: list[int] | None = None,
        amg_favorite: bool | None = None,
        amg_labels: list[str] | None = None,
        amg_untagged: bool | None = None,
        tps_ratings: list[int] | None = None,
        tps_favorite: bool | None = None,
        tps_labels: list[str] | None = None,
        tps_untagged: bool | None = None,
        **kwargs: Any,
    ) -> int:
        """Return the total number of items in the library."""
        sql_query = f"SELECT item_id FROM {self.db_table}"
        query_parts: list[str] = []
        query_params: dict[str, Any] = {}
        if favorite_only:
            query_parts.append("favorite = 1")
        if listen_later_only:
            query_parts.append("listen_later = 1")
        if album_types:
            query_parts.append("albums.album_type IN :album_types")
            query_params["album_types"] = [x.value for x in album_types]
        _apply_critical_reception_filters(
            query_parts=query_parts,
            query_params=query_params,
            dr_buckets=dr_buckets,
            amg_ratings=amg_ratings,
            amg_favorite=amg_favorite,
            amg_labels=amg_labels,
            amg_untagged=amg_untagged,
            tps_ratings=tps_ratings,
            tps_favorite=tps_favorite,
            tps_labels=tps_labels,
            tps_untagged=tps_untagged,
        )
        if query_parts:
            sql_query += f" WHERE {' AND '.join(query_parts)}"
        return await self.mass.music.database.get_count_from_query(sql_query, query_params)

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
        # delete the album itself from db
        # this will raise if the item still has references and recursive is false
        await super().remove_item_from_library(item_id)

    async def set_listen_later(self, item_id: str | int, listen_later: bool) -> None:
        """Set the listen_later flag on a library album.

        Independent of `favorite` — this is the Roon-style "save for later" pile,
        not a library/favorites add. Stamps `listen_later_added_at` with the
        current epoch on flip-to-true so the dedicated view can sort newest-first.
        """
        db_id = int(item_id)
        library_item = await self.get_library_item(db_id)
        if library_item.listen_later == listen_later:
            return
        update: dict[str, Any] = {"listen_later": listen_later}
        if listen_later:
            update["listen_later_added_at"] = int(time.time())
        else:
            update["listen_later_added_at"] = None
        await self.mass.music.database.update(self.db_table, {"item_id": db_id}, update)
        library_item = await self.get_library_item(db_id)
        self.mass.signal_event(EventType.MEDIA_ITEM_UPDATED, library_item.uri, library_item)

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
            if album_tracks and not album_tracks[0].image:
                # set album image from provider album if not present on tracks
                prov_album = await self.get_provider_item(item_id, provider_instance_id_or_domain)
                if prov_album.image:
                    for track in album_tracks:
                        if not track.image:
                            track.metadata.add_image(prov_album.image)
            return album_tracks

        db_items = await self.get_library_album_tracks(library_album.item_id)
        result: list[Track] = list(db_items)
        if in_library_only:
            # return in-library items only
            return sorted(db_items, key=lambda x: (x.disc_number, x.track_number))

        # return all (unique) items from all providers
        # because we are returning the items from all providers combined,
        # we need to make sure that we don't return duplicates
        unique_ids: set[str] = {f"{x.disc_number}.{x.track_number}" for x in db_items}
        unique_ids.update({f"{x.name.lower()}.{x.version.lower()}" for x in db_items})
        for db_item in db_items:
            unique_ids.update(x.item_id for x in db_item.provider_mappings)
        user = get_current_user()
        user_provider_filter = user.provider_filter if user and user.provider_filter else None
        for provider_mapping in library_album.provider_mappings:
            if (
                user_provider_filter
                and provider_mapping.provider_instance not in user_provider_filter
            ):
                continue
            provider_tracks = await self._get_provider_album_tracks(
                provider_mapping.item_id, provider_mapping.provider_instance
            )
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
                unique_id = f"{provider_track.disc_number}.{provider_track.track_number}"
                if unique_id in unique_ids:
                    continue
                unique_id = f"{provider_track.name.lower()}.{provider_track.version.lower()}"
                if unique_id in unique_ids:
                    continue
                unique_ids.add(unique_id)
                provider_track.album = library_album
                # always prefer album image
                album_images = [library_album.image] if library_album.image else []
                track_images: list[MediaItemImage] = provider_track.metadata.images or []
                provider_track.metadata.images = UniqueList(album_images + track_images)
                result.append(provider_track)
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
        search_query = f"{album.artists[0].name} - {album.name}" if album.artists else album.name
        result: UniqueList[Album] = UniqueList()
        for provider_id in self.mass.music.get_unique_providers():
            provider = self.mass.get_provider(provider_id)
            if not provider or not isinstance(provider, MusicProvider):
                continue
            if not provider.library_supported(MediaType.ALBUM):
                continue
            result.extend(
                prov_item
                for prov_item in await self.search(search_query, provider_id)
                if loose_compare_strings(album.name, prov_item.name)
                and compare_artists(prov_item.artists, album.artists, any_match=True)
                # make sure that the 'base' version is NOT included
                and not album.provider_mappings.intersection(prov_item.provider_mappings)
            )
        return result

    async def get_library_album_tracks(
        self,
        item_id: str | int,
    ) -> list[Track]:
        """Return in-database album tracks for the given database album."""
        db_id = int(item_id)  # ensure integer
        return await self.mass.music.tracks.get_library_items_by_query(
            extra_query_parts=["WHERE album_tracks.album_id = :album_id"],
            extra_query_params={"album_id": db_id},
        )

    async def add_item_mapping_as_album_to_library(self, item: ItemMapping) -> Album:
        """
        Add an ItemMapping as an Album to the library.

        This is only used in special occasions as is basically adds an album
        to the db without a lot of mandatory data, such as artists.
        """
        album = self.album_from_item_mapping(item)
        return await self.add_item_to_library(album)

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
                "favorite": item.favorite,
                "album_type": item.album_type,
                "year": item.year,
                "metadata": serialize_to_json(item.metadata),
                "external_ids": serialize_to_json(item.external_ids),
                "search_name": create_safe_string(item.name, True, True),
                "search_sort_name": create_safe_string(item.sort_name or "", True, True),
                "timestamp_added": int(item.date_added.timestamp()) if item.date_added else UNSET,
            },
        )
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
        metadata = update.metadata if overwrite else cur_item.metadata.update(update.metadata)
        # MediaItemMetadata.update() only fills None-valued fields and does not
        # deep-merge structured sub-shapes like critical_reception — so a fresher
        # CR (e.g. one that now carries a DR value) gets dropped on the floor when
        # the existing field is already non-None. Apply a targeted replacement
        # whenever the incoming CR strictly extends what's already stored.
        if (
            not overwrite
            and update.metadata is not None
            and _critical_reception_is_richer(
                update.metadata.critical_reception, metadata.critical_reception
            )
        ):
            metadata.critical_reception = update.metadata.critical_reception
        if getattr(update, "album_type", AlbumType.UNKNOWN) != AlbumType.UNKNOWN:
            album_type = update.album_type
        else:
            album_type = cur_item.album_type
        cur_item.external_ids.update(update.external_ids)
        name = update.name if overwrite else cur_item.name
        sort_name = update.sort_name if overwrite else cur_item.sort_name or update.sort_name
        await self.mass.music.database.update(
            self.db_table,
            {"item_id": db_id},
            {
                "name": name,
                "sort_name": sort_name,
                "version": update.version if overwrite else cur_item.version or update.version,
                "year": update.year if overwrite else cur_item.year or update.year,
                "album_type": album_type.value,
                "metadata": serialize_to_json(metadata),
                "external_ids": serialize_to_json(
                    update.external_ids if overwrite else cur_item.external_ids
                ),
                "search_name": create_safe_string(name, True, True),
                "search_sort_name": create_safe_string(sort_name or "", True, True),
                "timestamp_added": int(update.date_added.timestamp())
                if update.date_added
                else UNSET,
            },
        )
        # update/set provider_mappings table
        provider_mappings = (
            update.provider_mappings
            if overwrite
            else {*update.provider_mappings, *cur_item.provider_mappings}
        )
        await self.set_provider_mappings(db_id, provider_mappings, overwrite)
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

    async def radio_mode_base_tracks(
        self,
        item: Album,
        preferred_provider_instances: list[str] | None = None,
    ) -> list[Track]:
        """
        Get the list of base tracks from the controller used to calculate the dynamic radio.

        :param item: The Album to get base tracks for.
        :param preferred_provider_instances: List of preferred provider instance IDs to use.
        """
        return await self.tracks(item.item_id, item.provider, in_library_only=False)

    async def _set_album_artists(
        self,
        db_id: int,
        artists: Iterable[Artist | ItemMapping],
        overwrite: bool = False,
    ) -> None:
        """Store Album Artists."""
        if overwrite:
            # on overwrite, clear the album_artists table first
            await self.mass.music.database.delete(
                DB_TABLE_ALBUM_ARTISTS,
                {
                    "album_id": db_id,
                },
            )
        for artist in artists:
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

    async def match_provider(
        self, db_album: Album, provider: MusicProvider, strict: bool = True
    ) -> list[ProviderMapping]:
        """
        Try to find match on (streaming) provider for the provided (database) album.

        This is used to link objects of different providers/qualities together.
        """
        self.logger.debug("Trying to match album %s on provider %s", db_album.name, provider.name)
        matches: list[ProviderMapping] = []
        artist_name = db_album.artists[0].name
        search_str = f"{artist_name} - {db_album.name}"
        search_result = await self.search(search_str, provider.instance_id)
        for search_result_item in search_result:
            if not search_result_item.available:
                continue
            if not compare_media_item(db_album, search_result_item, strict=strict):
                continue
            # we must fetch the full album version, search results can be simplified objects
            prov_album = await self.get_provider_item(
                search_result_item.item_id,
                search_result_item.provider,
                fallback=search_result_item,
            )
            if compare_album(db_album, prov_album, strict=strict):
                # 100% match
                matches.extend(prov_album.provider_mappings)
        if not matches:
            self.logger.debug(
                "Could not find match for Album %s on provider %s",
                db_album.name,
                provider.name,
            )
        return matches

    async def match_providers(self, db_album: Album) -> None:
        """Try to find match on all (streaming) providers for the provided (database) album.

        This is used to link objects of different providers/qualities together.
        """
        if db_album.provider != "library":
            return  # Matching only supported for database items
        if not db_album.artists:
            return  # guard

        # try to find match on all providers
        processed_domains = set()
        for provider in self.mass.music.providers:
            if provider.domain in processed_domains:
                continue
            if ProviderFeature.SEARCH not in provider.supported_features:
                continue
            if not provider.library_supported(MediaType.ALBUM):
                continue
            if not provider.is_streaming_provider:
                # matching on unique providers is pointless as they push (all) their content to MA
                continue
            if match := await self.match_provider(db_album, provider):
                # 100% match, we update the db with the additional provider mapping(s)
                await self.add_provider_mappings(db_album.item_id, match)
                processed_domains.add(provider.domain)

    def album_from_item_mapping(self, item: ItemMapping) -> Album:
        """Create an Album object from an ItemMapping object."""
        domain, instance_id = None, None
        if prov := self.mass.get_provider(item.provider):
            domain = prov.domain
            instance_id = prov.instance_id
        return Album.from_dict(
            {
                **item.to_dict(),
                "provider_mappings": [
                    {
                        "item_id": item.item_id,
                        "provider_domain": domain,
                        "provider_instance": instance_id,
                        "available": item.available,
                    }
                ],
            }
        )
