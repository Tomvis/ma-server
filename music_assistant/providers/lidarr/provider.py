"""Lidarr Plugin Provider implementation."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any, cast

from music_assistant_models.enums import ExternalID, MediaType
from music_assistant_models.errors import (
    InvalidDataError,
    LoginFailed,
    MediaNotFoundError,
    ProviderUnavailableError,
)

from music_assistant.models.plugin import PluginProvider
from music_assistant.providers.lidarr.client import LidarrClient
from music_assistant.providers.lidarr.constants import (
    CONF_API_KEY,
    CONF_METADATA_PROFILE_ID,
    CONF_MONITOR_MODE,
    CONF_QUALITY_PROFILE_ID,
    CONF_ROOT_FOLDER,
    CONF_SEARCH_ON_ADD,
    CONF_URL,
    CONF_VERIFY_SSL,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.media_items import Album
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant


# Lidarr's metadata refresh after artist creation is async. We trigger it
# explicitly via /command and poll the command resource until completion;
# these constants bound that wait.
COMMAND_TIMEOUT = 60.0
COMMAND_POLL_INTERVAL = 1.5

# Even after the RefreshArtist command reports completed, Lidarr may take
# additional time to commit the discography to its DB. We poll the album list
# directly until the requested album shows up, capped by this timeout.
DISCOGRAPHY_TIMEOUT = 60.0
DISCOGRAPHY_POLL_INTERVAL = 1.5


def _normalize_title(value: str) -> str:
    """Casefold + strip punctuation for fuzzy album-title matches.

    Lidarr's stored title sometimes carries an edition parenthetical or
    differs in punctuation (ellipsis, bang) from the streaming-provider title.
    We drop a trailing parenthetical and reduce the rest to alphanumerics.
    """
    no_paren = re.sub(r"\s*\(.*?\)\s*$", "", value)
    return re.sub(r"[^a-z0-9]+", "", no_paren.casefold())


class LidarrProvider(PluginProvider):
    """Plugin provider that bridges Music Assistant albums into Lidarr."""

    _client: LidarrClient
    _unregister_handles: list[Callable[[], None]]

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
    ) -> None:
        """Initialize the provider with a bound Lidarr client."""
        super().__init__(mass, manifest, config)
        self._unregister_handles = []
        self._client = LidarrClient(
            url=cast("str", config.get_value(CONF_URL)),
            api_key=cast("str", config.get_value(CONF_API_KEY)),
            session=mass.http_session,
            verify_ssl=bool(config.get_value(CONF_VERIFY_SSL) or False),
        )

    async def loaded_in_mass(self) -> None:
        """Probe Lidarr and register the WebSocket command."""
        try:
            await self._client.system_status()
        except LoginFailed as err:
            raise LoginFailed(f"Lidarr authentication failed: {err}") from err
        except Exception as err:
            raise ProviderUnavailableError(
                f"Could not reach Lidarr at {self.config.get_value(CONF_URL)}: {err}"
            ) from err

        self._unregister_handles.append(
            self.mass.register_api_command("lidarr/add_album", self.add_album)
        )

    async def unload(self, is_removed: bool = False) -> None:
        """Drop the registered command handler."""
        for unregister in self._unregister_handles:
            unregister()
        self._unregister_handles.clear()
        await super().unload(is_removed)

    # ----- public API command -----

    async def add_album(self, item: str) -> dict[str, Any]:
        """Send an album to Lidarr.

        - Resolves the album from MA (`item` is an MA URI).
        - Looks up the artist in Lidarr by MBID; adds it if missing.
        - Finds the matching album in Lidarr's discography (by MBID) and sets
          monitored=true so Lidarr will start fetching it.
        - Optionally kicks off an AlbumSearch when the user opted in.

        Returns a dict shaped like the frontend's LidarrAddAlbumResult.
        """
        if not isinstance(item, str):
            raise InvalidDataError("lidarr/add_album expects an MA URI string")

        media_item = await self.mass.music.get_item_by_uri(item)
        if media_item.media_type != MediaType.ALBUM:
            raise InvalidDataError(
                f"lidarr/add_album only accepts albums, got {media_item.media_type.value}"
            )
        album = cast("Album", media_item)

        if not album.artists:
            raise InvalidDataError(f"Album {album.name!r} has no artist information")
        artist_name = album.artists[0].name

        artist_mbid, album_mbid = await self._resolve_mbids(album)
        if not artist_mbid:
            # Final fallback: ask Lidarr itself to identify the artist by name.
            # Lidarr's /artist/lookup is backed by MusicBrainz with disambiguation,
            # so a clean exact-name match is typically reliable.
            artist_mbid = await self._lookup_artist_via_lidarr(artist_name)
        if not artist_mbid:
            raise InvalidDataError(
                f"Couldn't determine a MusicBrainz artist ID for {artist_name!r}. "
                "Lidarr requires an MBID to add the artist — try refreshing the "
                "album's metadata in Music Assistant first."
            )

        # Find or create the artist in Lidarr.
        lidarr_artist, just_added = await self._find_or_add_artist(artist_mbid, artist_name)
        artist_id = int(lidarr_artist["id"])

        # When freshly added, wait for Lidarr's RefreshArtist command, then
        # immediately flip the artist's monitored flag back on (Lidarr's
        # addOptions.monitor="none" zeroes it out). We do this BEFORE finding
        # the album so the artist is in the right state even if discography
        # polling times out.
        if just_added:
            await self._wait_for_refresh(artist_id)
            try:
                lidarr_artist = await self._client.get_artist(artist_id)
            except Exception as err:
                self.logger.warning("Could not re-fetch artist %d: %s", artist_id, err)
            if not lidarr_artist.get("monitored"):
                lidarr_artist["monitored"] = True
                try:
                    lidarr_artist = await self._client.update_artist(lidarr_artist)
                    self.logger.info(
                        "Flipped artist %r to monitored",
                        lidarr_artist.get("artistName"),
                    )
                except Exception as err:
                    self.logger.warning(
                        "Failed to flip %s back to monitored after add: %s",
                        lidarr_artist.get("artistName"),
                        err,
                    )

        lidarr_album = await self._poll_album(
            artist_id, album_mbid=album_mbid, album_name=album.name
        )
        if lidarr_album is None:
            raise MediaNotFoundError(
                f"Lidarr couldn't find {album.name!r} in {artist_name!r}'s discography"
            )

        album_id = int(lidarr_album["id"])
        already_monitored = bool(lidarr_album.get("monitored"))
        self.logger.info(
            "Lidarr album %r (id=%d) currently monitored=%s",
            lidarr_album.get("title"),
            album_id,
            already_monitored,
        )
        if not already_monitored:
            await self._monitor_album(album_id)
            if bool(self.config.get_value(CONF_SEARCH_ON_ADD) or False):
                # Best-effort fire-and-forget — don't block the toast on the search queue.
                try:
                    await self._client.queue_command("AlbumSearch", albumIds=[album_id])
                except Exception as err:
                    self.logger.debug("AlbumSearch trigger failed: %s", err)

        return {
            "artist_name": lidarr_artist.get("artistName") or artist_name,
            "album_name": lidarr_album.get("title") or album.name,
            "artist_added": just_added,
            "album_monitored": True,
            "already_monitored": already_monitored,
            "lidarr_instance": self.name,
        }

    # ----- helpers -----

    async def _resolve_mbids(self, album: Album) -> tuple[str | None, str | None]:
        """Best-effort resolve (artist_mbid, album_mbid) for the given album.

        Strategy:
        1. Use MBIDs already on the album/artist mapping.
        2. Fetch the full Artist via the music controller (ItemMapping may be sparse).
        3. Ask the MusicBrainz provider to look up by release-group/release MBID.
        4. As a last resort, search MusicBrainz with one of the album's tracks.
        """
        album_mbid = album.get_external_id(ExternalID.MB_ALBUM) or album.get_external_id(
            ExternalID.MB_RELEASEGROUP
        )
        artist_mapping = album.artists[0]
        artist_mbid = artist_mapping.get_external_id(ExternalID.MB_ARTIST)

        if artist_mbid:
            return artist_mbid, album_mbid

        # Step 2: maybe the full Artist record carries it.
        try:
            full_artist = await self.mass.music.artists.get(
                artist_mapping.item_id, artist_mapping.provider
            )
        except Exception as err:
            self.logger.debug("Artist fetch failed during MBID resolve: %s", err)
        else:
            artist_mbid = full_artist.get_external_id(ExternalID.MB_ARTIST)
            if artist_mbid:
                return artist_mbid, album_mbid

        mb_provider: Any = self.mass.get_provider("musicbrainz")
        if mb_provider is None:
            return None, album_mbid

        # Step 3: ReleaseGroup-anchored lookup. Requires *some* MBID on the album.
        if album_mbid:
            try:
                mb_artist = await mb_provider.get_artist_details_by_album(
                    artist_mapping.name, album
                )
            except Exception as err:
                self.logger.debug("MB lookup-by-album failed: %s", err)
            else:
                if mb_artist and mb_artist.id:
                    return mb_artist.id, album_mbid

        # Step 4: track-anchored search as last resort. We grab the first track
        # from the album to get a recording name MusicBrainz can match on.
        try:
            tracks = await self.mass.music.albums.tracks(
                album.item_id, album.provider, in_library_only=False
            )
        except Exception as err:
            self.logger.debug("Could not fetch album tracks for MB search: %s", err)
            tracks = []
        for track in tracks[:3]:
            try:
                hit = await mb_provider.search(
                    artist_mapping.name, album.name, track.name, track.version or None
                )
            except Exception as err:
                self.logger.debug("MB track-search failed: %s", err)
                continue
            if hit:
                mb_artist, mb_release_group, _ = hit
                # If we picked up a release-group MBID along the way, prefer it.
                resolved_album_mbid = album_mbid or (
                    mb_release_group.id if mb_release_group else None
                )
                return mb_artist.id, resolved_album_mbid

        return None, album_mbid

    async def _lookup_artist_via_lidarr(self, artist_name: str) -> str | None:
        """Resolve an MBID via Lidarr's /artist/lookup as a last-ditch fallback.

        Lidarr's lookup endpoint queries MusicBrainz with their own ranking, so
        the top exact-name match is normally trustworthy. We accept the first
        case-insensitive name match that has a foreignArtistId; if no exact
        match exists we fall back to the first candidate Lidarr returned.
        """
        try:
            candidates = await self._client.lookup_artist(artist_name)
        except Exception as err:
            self.logger.warning("Lidarr lookup-by-name failed for %r: %s", artist_name, err)
            return None
        self.logger.info(
            "Lidarr lookup for %r returned %d candidate(s)",
            artist_name,
            len(candidates),
        )
        if not candidates:
            return None
        target = artist_name.casefold().strip()
        for c in candidates:
            mbid = c.get("foreignArtistId")
            name = c.get("artistName", "")
            if mbid and name.casefold().strip() == target:
                self.logger.info("Lidarr lookup matched %r -> %s", name, mbid)
                return cast("str", mbid)
        # No exact match — Lidarr's first result is usually what the user means
        # (their ranking already prefers the canonical artist), so accept it.
        first = candidates[0]
        if first.get("foreignArtistId"):
            self.logger.info(
                "Lidarr lookup: no exact match for %r, accepting top result %r -> %s",
                artist_name,
                first.get("artistName"),
                first["foreignArtistId"],
            )
            return cast("str", first["foreignArtistId"])
        return None

    async def _find_or_add_artist(
        self, artist_mbid: str, artist_name: str
    ) -> tuple[dict[str, Any], bool]:
        """Return (Lidarr artist record, just_added flag), creating it if needed."""
        for existing in await self._client.list_artists():
            if existing.get("foreignArtistId") == artist_mbid:
                return existing, False

        # Not in Lidarr yet — look it up by MBID and POST to create.
        candidates = await self._client.lookup_artist(f"lidarr:{artist_mbid}")
        match = next(
            (c for c in candidates if c.get("foreignArtistId") == artist_mbid),
            None,
        )
        if match is None:
            # Fall back to a name search; pick the candidate whose MBID matches.
            candidates = await self._client.lookup_artist(artist_name)
            match = next(
                (c for c in candidates if c.get("foreignArtistId") == artist_mbid),
                None,
            )
        if match is None:
            raise MediaNotFoundError(
                f"Lidarr couldn't resolve artist MBID {artist_mbid} ({artist_name!r})"
            )

        # addOptions.monitor controls which existing albums Lidarr flips on at
        # add time. We want NONE — the user pushed one specific album and we
        # monitor only that one explicitly afterward. monitorNewItems on the
        # artist resource is a separate field that controls Lidarr's behavior
        # for *future* releases, and that's where the user's preference applies.
        body = {
            **match,
            "monitored": True,
            "monitorNewItems": str(self.config.get_value(CONF_MONITOR_MODE) or "all"),
            "rootFolderPath": str(self.config.get_value(CONF_ROOT_FOLDER)),
            "qualityProfileId": int(cast("int", self.config.get_value(CONF_QUALITY_PROFILE_ID))),
            "metadataProfileId": int(cast("int", self.config.get_value(CONF_METADATA_PROFILE_ID))),
            "addOptions": {
                "monitor": "none",
                "searchForMissingAlbums": False,
            },
        }
        return await self._client.add_artist(body), True

    async def _monitor_album(self, album_id: int) -> None:
        """Flip an album's monitored flag, verifying the change actually stuck.

        First tries the bulk /album/monitor endpoint. If a fresh GET shows the
        change didn't take effect (Lidarr 202s but ignores the body in some
        states), falls back to PUT /album/{id} with the full AlbumResource —
        the same route Lidarr's UI uses for the toggle.
        """
        await self._client.set_albums_monitored([album_id], monitored=True)
        verified = await self._client.get_album(album_id)
        if not verified.get("monitored"):
            self.logger.info(
                "Bulk /album/monitor was a no-op for id=%d, retrying via PUT /album/{id}",
                album_id,
            )
            await self._client.update_album({**verified, "monitored": True})
            verified = await self._client.get_album(album_id)
        if not verified.get("monitored"):
            self.logger.warning(
                "Lidarr STILL has album id=%d as monitored=false after both PUT routes",
                album_id,
            )
        else:
            self.logger.info("Confirmed album id=%d is now monitored", album_id)

    async def _wait_for_refresh(self, lidarr_artist_id: int) -> None:
        """Trigger a RefreshArtist command and wait for it to finish.

        Lidarr typically queues this automatically after add_artist, but firing
        explicitly gives us a command id we can poll deterministically instead
        of looping over /album.
        """
        try:
            command = await self._client.queue_command(
                "RefreshArtist", artistIds=[int(lidarr_artist_id)]
            )
        except Exception as err:
            self.logger.debug("RefreshArtist enqueue failed: %s", err)
            await asyncio.sleep(COMMAND_POLL_INTERVAL)
            return
        command_id = command.get("id")
        if not command_id:
            await asyncio.sleep(COMMAND_POLL_INTERVAL)
            return
        deadline = asyncio.get_running_loop().time() + COMMAND_TIMEOUT
        while True:
            status = (await self._client.get_command(int(command_id))).get("status")
            if status in ("completed", "failed", "aborted"):
                return
            if asyncio.get_running_loop().time() > deadline:
                self.logger.warning(
                    "Lidarr RefreshArtist did not finish within %.0fs; proceeding anyway",
                    COMMAND_TIMEOUT,
                )
                return
            await asyncio.sleep(COMMAND_POLL_INTERVAL)

    async def _find_album(
        self,
        albums: list[dict[str, Any]],
        *,
        album_mbid: str | None,
        album_name: str,
    ) -> dict[str, Any] | None:
        """Locate the Lidarr album record matching the requested album."""
        if album_mbid:
            for la in albums:
                if la.get("foreignAlbumId") == album_mbid or la.get("releaseGroupId") == album_mbid:
                    return la
        # Punctuation-insensitive name match — Lidarr's title format may differ
        # slightly from the streaming provider's (ellipsis vs ellipses,
        # parenthetical edition tags, …).
        target = _normalize_title(album_name)
        for la in albums:
            if _normalize_title(la.get("title", "")) == target:
                return la
        return None

    async def _poll_album(
        self,
        lidarr_artist_id: int,
        *,
        album_mbid: str | None,
        album_name: str,
    ) -> dict[str, Any] | None:
        """Poll /album?artistId until the requested album appears.

        Lidarr finishes the RefreshArtist command before the album rows are
        readable, so list_albums right after the command can come back empty
        or stale. Retry on a 1.5s cadence up to DISCOGRAPHY_TIMEOUT.
        """
        deadline = asyncio.get_running_loop().time() + DISCOGRAPHY_TIMEOUT
        attempts = 0
        while True:
            attempts += 1
            albums = await self._client.list_albums(lidarr_artist_id)
            match = await self._find_album(albums, album_mbid=album_mbid, album_name=album_name)
            if match is not None:
                self.logger.info(
                    "Discography settled: found %r (id=%d) after %d poll(s)",
                    match.get("title"),
                    int(match.get("id", 0)),
                    attempts,
                )
                return match
            if asyncio.get_running_loop().time() > deadline:
                titles = [la.get("title") for la in albums[:25]]
                self.logger.warning(
                    "Discography poll for artist=%d timed out after %d attempts. "
                    "Looking for %r (mbid=%s). Lidarr has %d album(s); first %d titles: %s",
                    lidarr_artist_id,
                    attempts,
                    album_name,
                    album_mbid,
                    len(albums),
                    len(titles),
                    titles,
                )
                return None
            await asyncio.sleep(DISCOGRAPHY_POLL_INTERVAL)
