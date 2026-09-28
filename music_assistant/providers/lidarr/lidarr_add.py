"""
The "Add to Lidarr" flow against Lidarr's own API.

Given an artist MBID, the album's identifiers and the acting user's root folder:

1. Use the artist if Lidarr already has it -- wherever it lives, even under another
   person's root (moving files between libraries is not this action's call). Otherwise
   add it to ``root_folder`` with that root folder's default profiles, monitoring no
   existing albums and no future releases.
2. Make sure the artist is monitored: Lidarr never searches an unmonitored artist's
   albums, and it zeroes the flag itself on an add with ``addOptions.monitor=none``.
3. Find the album in the artist's discography, refreshing the artist first when it is
   new or the album isn't loaded yet (a release newer than the last refresh).
4. Monitor that one album and queue an AlbumSearch for it.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from music_assistant.providers.lidarr.client import LidarrError

if TYPE_CHECKING:
    import logging

    from music_assistant.providers.lidarr.client import LidarrClient

POLL_INTERVAL = 1.5
# Bounds the wait for a queued RefreshArtist command to finish.
COMMAND_TIMEOUT = 60.0
# After the command completes Lidarr can still be committing the discography, so the
# album list is polled until the album appears, bounded by this.
DISCOGRAPHY_TIMEOUT = 60.0


@dataclass(frozen=True)
class AlbumRequest:
    """What MA knows about the album being sent to Lidarr."""

    artist_mbid: str
    artist_name: str
    album_name: str
    release_group_mbid: str | None
    release_mbid: str | None


async def add_album(
    client: LidarrClient,
    request: AlbumRequest,
    *,
    root_folder: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    """
    Get ``request``'s album monitored and searched in Lidarr.

    Returns the frontend's LidarrAddAlbumResult fields, minus ``lidarr_instance``.

    :param client: The Lidarr client.
    :param request: The album to add.
    :param root_folder: Root folder path for a newly added artist (the acting user's).
    :param logger: Logger for progress and fallbacks.
    """
    artist, added = await _find_or_add_artist(client, request, root_folder)
    if added:
        await _refresh_artist(client, int(artist["id"]), logger)
        artist = await client.get_artist(int(artist["id"]))
    if not artist.get("monitored"):
        artist = await client.update_artist({**artist, "monitored": True})
        logger.info("Lidarr: set artist %r monitored", artist.get("artistName"))

    album = await _find_album(client, artist, request, refreshed=added, logger=logger)
    album_id = int(album["id"])
    already_monitored = bool(album.get("monitored"))
    if not already_monitored:
        await _monitor_album(client, album_id, logger)
        await client.queue_command("AlbumSearch", albumIds=[album_id])
    logger.info(
        "Lidarr: %r - %r (album id=%d) monitored (was already: %s, artist added: %s)",
        artist.get("artistName"),
        album.get("title"),
        album_id,
        already_monitored,
        added,
    )
    return {
        "artist_name": artist.get("artistName") or request.artist_name,
        "album_name": album.get("title") or request.album_name,
        "artist_added": added,
        "album_monitored": True,
        "already_monitored": already_monitored,
    }


def normalize_title(value: str) -> str:
    """
    Casefold a title and reduce it to alphanumerics, dropping a trailing parenthetical.

    Lidarr's stored title can carry or lack an edition tag and differ in punctuation
    from the streaming provider's title.
    """
    no_paren = re.sub(r"\s*\(.*?\)\s*$", "", value)
    return re.sub(r"[^0-9a-z]+", "", no_paren.casefold())


async def _find_or_add_artist(
    client: LidarrClient, request: AlbumRequest, root_folder: str
) -> tuple[dict[str, Any], bool]:
    """Return (Lidarr artist, whether it was just added)."""
    for existing in await client.list_artists():
        if existing.get("foreignArtistId") == request.artist_mbid:
            return existing, False

    root = await _root(client, root_folder)
    candidates = await client.lookup_artist(f"lidarr:{request.artist_mbid}")
    match = _with_mbid(candidates, request.artist_mbid)
    if match is None:
        match = _with_mbid(await client.lookup_artist(request.artist_name), request.artist_mbid)
    if match is None:
        raise LidarrError(
            f"Lidarr couldn't find artist {request.artist_name!r} (MusicBrainz "
            f"{request.artist_mbid}) on its metadata server -- it may be too new for "
            "Lidarr's MusicBrainz mirror."
        )
    body = {
        **match,
        "monitored": True,
        "monitorNewItems": "none",
        "rootFolderPath": root["path"],
        "qualityProfileId": root["defaultQualityProfileId"],
        "metadataProfileId": root["defaultMetadataProfileId"],
        "addOptions": {"monitor": "none", "searchForMissingAlbums": False},
    }
    return await client.add_artist(body), True


def _with_mbid(candidates: list[dict[str, Any]], mbid: str) -> dict[str, Any] | None:
    return next((c for c in candidates if c.get("foreignArtistId") == mbid), None)


async def _root(client: LidarrClient, path: str) -> dict[str, Any]:
    """Return Lidarr's root folder at ``path``, requiring its default profiles."""
    wanted = path.rstrip("/")
    for root in await client.list_root_folders():
        if str(root.get("path", "")).rstrip("/") == wanted:
            if not (root.get("defaultQualityProfileId") and root.get("defaultMetadataProfileId")):
                raise LidarrError(
                    f"Lidarr root folder {wanted!r} has no default quality/metadata profile; "
                    "set them in Lidarr (Settings > Media Management > Root Folders)."
                )
            return root
    raise LidarrError(f"Lidarr has no root folder {wanted!r}; fix the Lidarr provider's options.")


async def _find_album(
    client: LidarrClient,
    artist: dict[str, Any],
    request: AlbumRequest,
    *,
    refreshed: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Find the album in the artist's discography, refreshing the artist once if needed."""
    artist_id = int(artist["id"])
    if not refreshed:
        if (album := _match(await client.list_albums(artist_id), request)) is not None:
            return album
        await _refresh_artist(client, artist_id, logger)
    deadline = asyncio.get_running_loop().time() + DISCOGRAPHY_TIMEOUT
    while True:
        albums = await client.list_albums(artist_id)
        if (album := _match(albums, request)) is not None:
            return album
        if asyncio.get_running_loop().time() >= deadline:
            break
        await asyncio.sleep(POLL_INTERVAL)
    profile = await _metadata_profile_name(client, artist.get("metadataProfileId"))
    raise LidarrError(
        f"{request.album_name!r} isn't in Lidarr's catalog for {artist.get('artistName')!r} "
        f"({len(albums)} albums loaded). The artist's metadata profile {profile!r} may "
        "exclude its release type (e.g. Singles), or Lidarr's MusicBrainz mirror lags."
    )


def _match(albums: list[dict[str, Any]], request: AlbumRequest) -> dict[str, Any] | None:
    """Match by release group, then by one of its releases, then by normalized title."""
    if request.release_group_mbid:
        for album in albums:
            if album.get("foreignAlbumId") == request.release_group_mbid:
                return album
    if request.release_mbid:
        for album in albums:
            releases = album.get("releases") or []
            if any(r.get("foreignReleaseId") == request.release_mbid for r in releases):
                return album
    target = normalize_title(request.album_name)
    return next((a for a in albums if normalize_title(a.get("title", "")) == target), None)


async def _refresh_artist(client: LidarrClient, artist_id: int, logger: logging.Logger) -> None:
    """Queue RefreshArtist and wait (bounded) for the command to finish."""
    command = await client.queue_command("RefreshArtist", artistIds=[artist_id])
    deadline = asyncio.get_running_loop().time() + COMMAND_TIMEOUT
    while command.get("id"):
        status = (await client.get_command(int(command["id"]))).get("status")
        if status in ("completed", "failed", "aborted"):
            return
        if asyncio.get_running_loop().time() >= deadline:
            logger.warning(
                "Lidarr RefreshArtist %d still %s; polling albums anyway", artist_id, status
            )
            return
        await asyncio.sleep(POLL_INTERVAL)


async def _monitor_album(client: LidarrClient, album_id: int, logger: logging.Logger) -> None:
    """
    Monitor an album, verifying it stuck.

    The bulk /album/monitor route can 202 without applying in some states; the full
    PUT /album/{id} (the route Lidarr's UI uses) is the fallback.
    """
    await client.set_albums_monitored([album_id], monitored=True)
    album = await client.get_album(album_id)
    if not album.get("monitored"):
        await client.update_album({**album, "monitored": True})
        album = await client.get_album(album_id)
    if not album.get("monitored"):
        raise LidarrError(f"Lidarr refused to monitor album id={album_id}")
    logger.debug("Lidarr: album id=%d monitored", album_id)


async def _metadata_profile_name(client: LidarrClient, profile_id: Any) -> str:
    try:
        profiles = await client.list_metadata_profiles()
    except LidarrError:
        return f"id {profile_id}"
    return next((p["name"] for p in profiles if p.get("id") == profile_id), f"id {profile_id}")
