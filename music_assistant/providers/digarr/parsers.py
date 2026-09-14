"""
Resolve digarr recommendations to playable items.

This is a deliberate adaptation of lastfm_recommendations/parsers.py rather than
a shared import. That module is UPSTREAM code (confirmed via
`git ls-tree origin/dev`); extracting a common helper would place fork edits in
an upstream file and cost a merge conflict on every rebase. The duplication is
bounded and lives in a file no upstream commit will ever touch.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

from music_assistant_models.enums import ExternalID, MediaType, ProviderFeature
from music_assistant_models.media_items import Album, Artist, ItemMapping, Track

from music_assistant.constants import MASS_LOGGER_NAME
from music_assistant.helpers.compare import compare_strings
from music_assistant.providers.digarr.constants import (
    DOMAIN,
    PROVIDER_SEARCH_LIMIT,
    SEARCH_CONCURRENCY_LIMIT,
)

if TYPE_CHECKING:
    from music_assistant import MusicAssistant
    from music_assistant.controllers.music.media.albums import AlbumsController
    from music_assistant.controllers.music.media.artists import ArtistsController
    from music_assistant.controllers.music.media.tracks import TracksController
    from music_assistant.providers.digarr.client import DigarrRecommendation

LOGGER = logging.getLogger(f"{MASS_LOGGER_NAME}.digarr")

# Limit concurrent provider searches to avoid overwhelming their APIs.
_SEARCH_SEMAPHORE = asyncio.Semaphore(SEARCH_CONCURRENCY_LIMIT)


async def resolve_artist(
    rec: DigarrRecommendation,
    mass: MusicAssistant,
    provider_instance_to_skip: str,
) -> Artist | None:
    """
    Resolve a digarr recommendation to a library or streaming-provider Artist.

    A plugin provider cannot serve artist detail pages, so an unresolved artist
    must never be emitted -- its card would render and then fail on click.

    :param rec: The digarr recommendation to resolve.
    :param mass: MusicAssistant instance for accessing library and providers.
    :param provider_instance_to_skip: Provider instance to skip (ourselves).
    """
    item_mapping = ItemMapping(
        media_type=MediaType.ARTIST,
        item_id="temp",
        provider=DOMAIN,
        name=rec.artist_name,
        # digarr guarantees artists.mbid (uuid, NOT NULL, UNIQUE), so unlike the
        # Last.fm original this is the primary route, not a lucky extra.
        external_ids={(ExternalID.MB_ARTIST, rec.artist_mbid)},
    )
    return cast(
        "Artist | None",
        await _resolve_item(item_mapping, mass, provider_instance_to_skip),
    )


def _is_matching_result(
    item_mapping: ItemMapping, result: Artist | Album | Track, artist_name: str | None
) -> bool:
    """
    Return True if a search result matches the searched item by name (and artist, if known).

    :param item_mapping: ItemMapping that was searched for.
    :param result: Search result to verify.
    :param artist_name: Artist name to verify against the result's artists, if known.
    """
    # Album/track search names are "Artist - Title" while results usually expose only the title.
    searched_name = item_mapping.name
    title_part = (
        searched_name[len(artist_name) + 3 :]
        if artist_name and searched_name.startswith(f"{artist_name} - ")
        else searched_name
    )
    if not compare_strings(title_part, result.name, strict=False) and not compare_strings(
        searched_name, result.name, strict=False
    ):
        return False

    # Verify the artist too: a matching title from a different artist (cover or karaoke
    # version) must not pass. Results without artist info can't contradict, accept those.
    result_artists = getattr(result, "artists", None)
    if artist_name and result_artists:
        return any(
            compare_strings(artist_name, result_artist.name, strict=False)
            for result_artist in result_artists
        )
    return True


def _get_streaming_providers(
    mass: MusicAssistant, item_mapping: ItemMapping, provider_instance_to_skip: str
) -> list[Any]:
    """
    Return streaming providers that support the ItemMapping's media type.

    Sourced from `mass.get_providers_supporting_feature` (rather than walking
    `mass.music.providers` and checking `supported_features` by hand, as the
    Last.fm original does) since that helper already does exactly this lookup.

    :param mass: MusicAssistant instance.
    :param item_mapping: ItemMapping with the media type to search for.
    :param provider_instance_to_skip: Provider instance to skip (ourselves).
    """
    if item_mapping.media_type == MediaType.ARTIST:
        feature = ProviderFeature.LIBRARY_ARTISTS
    elif item_mapping.media_type == MediaType.ALBUM:
        feature = ProviderFeature.LIBRARY_ALBUMS
    elif item_mapping.media_type == MediaType.TRACK:
        feature = ProviderFeature.LIBRARY_TRACKS
    else:
        return []

    return [
        p
        for p in mass.get_providers_supporting_feature(feature)
        if p.instance_id != provider_instance_to_skip and p.is_streaming_provider
    ]


async def _search_provider(
    item_mapping: ItemMapping,
    provider: Any,
) -> Artist | Album | Track | None:
    """
    Search a single provider for a matching item.

    Calls the provider's own `search()` directly rather than routing through
    the media controller's `search()` (as the Last.fm original does): the
    provider object is already in hand from `_get_streaming_providers`, and
    catching broadly here (not just `MusicAssistantError`) means one
    misbehaving provider -- a timeout, a malformed response, anything -- can
    never sink resolution against the others.

    :param item_mapping: ItemMapping to search for.
    :param provider: Provider instance to search.
    """
    async with _SEARCH_SEMAPHORE:
        try:
            LOGGER.debug(
                "Searching %s on %s for: %s",
                item_mapping.media_type.value,
                provider.name,
                item_mapping.name,
            )
            # Use a higher limit to work around provider bugs (e.g. Spotify misbehaves at limit=1).
            search_results = await provider.search(
                item_mapping.name, [item_mapping.media_type], PROVIDER_SEARCH_LIMIT
            )
            if item_mapping.media_type == MediaType.ARTIST:
                candidates = search_results.artists
            elif item_mapping.media_type == MediaType.ALBUM:
                candidates = search_results.albums
            else:
                candidates = search_results.tracks

            if not candidates:
                return None

            return cast("Artist | Album | Track", candidates[0])
        except Exception as err:
            LOGGER.debug("Provider %s search failed: %s", provider.name, type(err).__name__)
            return None


async def _search_providers_concurrent(
    item_mapping: ItemMapping,
    providers: list[Any],
    artist_name: str | None,
) -> Artist | Album | Track | None:
    """
    Search multiple providers concurrently and return the first verified match.

    :param item_mapping: ItemMapping to search for.
    :param providers: List of providers to search.
    :param artist_name: Artist name to verify candidate matches against, if known.
    """
    tasks = [
        asyncio.create_task(_search_provider(item_mapping, provider)) for provider in providers
    ]

    for task in asyncio.as_completed(tasks):
        result = await task
        if result is None:
            continue

        if _is_matching_result(item_mapping, result, artist_name):
            LOGGER.debug(
                "Match on %s: %s (searched: %s)",
                result.provider,
                result.name,
                item_mapping.name,
            )
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return result

        LOGGER.debug(
            "Rejecting %s from %s: name mismatch (searched: %s)",
            result.name,
            result.provider,
            item_mapping.name,
        )

    return None


async def _resolve_item(
    item_mapping: ItemMapping,
    mass: MusicAssistant,
    provider_instance_to_skip: str,
    artist_name: str | None = None,
) -> Artist | Album | Track | None:
    """
    Resolve an ItemMapping to a library or provider item.

    :param item_mapping: ItemMapping with metadata and external IDs.
    :param mass: MusicAssistant instance.
    :param provider_instance_to_skip: Provider instance to skip (ourselves).
    :param artist_name: Artist name to verify candidate matches against, if known.
    """
    ctrl: ArtistsController | AlbumsController | TracksController
    if item_mapping.media_type == MediaType.ARTIST:
        ctrl = mass.music.artists
    elif item_mapping.media_type == MediaType.ALBUM:
        ctrl = mass.music.albums
    elif item_mapping.media_type == MediaType.TRACK:
        ctrl = mass.music.tracks
    else:
        return None

    LOGGER.debug(
        "Resolving %s: %s (external IDs: %s)",
        item_mapping.media_type.value,
        item_mapping.name,
        item_mapping.external_ids or "none",
    )

    if library_item := await ctrl.get_library_item_by_external_ids(item_mapping.external_ids):
        LOGGER.debug("Found %s in library: %s", item_mapping.media_type.value, library_item.name)
        return library_item

    streaming_providers = _get_streaming_providers(mass, item_mapping, provider_instance_to_skip)
    if not streaming_providers:
        LOGGER.debug("No streaming providers available for resolution")
        return None

    result = await _search_providers_concurrent(item_mapping, streaming_providers, artist_name)
    if result is None:
        LOGGER.debug("Could not resolve %s: %s", item_mapping.media_type.value, item_mapping.name)
        return None

    # Streaming providers expose ISRCs (and sometimes MBIDs) that digarr doesn't always
    # provide; re-check the library against the resolved item's external IDs so we prefer
    # the user's own copy when it exists.
    if result.external_ids:
        if library_item := await ctrl.get_library_item_by_external_ids(result.external_ids):
            LOGGER.debug(
                "Found %s in library via resolved external IDs: %s",
                item_mapping.media_type.value,
                library_item.name,
            )
            return library_item

    return result
