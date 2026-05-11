"""Lidarr Plugin Provider implementation (music-rater bridge).

The "Add to Lidarr" action no longer talks to Lidarr directly; it hands the
album off to music-rater (operator-run companion service), which orchestrates
the actual Lidarr sync. This keeps Music Assistant out of the artist /
metadata-profile / root-folder business and lets music-rater own the policy.

Two-step flow per music-rater's documented API:

1. Resolve  GET /api/v1/albums?music_assistant_uri=<uri>
            Music-rater stamps the MA URI on every album it syncs, so this
            normally resolves unambiguously. If the album isn't synced
            (user added it to MA by hand), fall back to ?search=<artist+album>.

2. Queue    POST /api/v1/albums/{album_id}/lidarr/queue
            Idempotent — sets lidarr_manual_add=True and runs an inline single-
            album sync. Returns a LidarrQueueResponse we map back to the
            frontend's LidarrAddAlbumResult shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from music_assistant_models.enums import MediaType
from music_assistant_models.errors import InvalidDataError, ProviderUnavailableError

from music_assistant.models.plugin import PluginProvider
from music_assistant.providers.lidarr.client import MusicRaterClient
from music_assistant.providers.lidarr.constants import CONF_URL, CONF_VERIFY_SSL

if TYPE_CHECKING:
    from collections.abc import Callable

    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.media_items import Album
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant


class LidarrProvider(PluginProvider):
    """Plugin provider that bridges Music Assistant albums into Lidarr via music-rater."""

    _client: MusicRaterClient
    _unregister_handles: list[Callable[[], None]]

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
    ) -> None:
        """Initialize the provider with a bound music-rater client."""
        super().__init__(mass, manifest, config)
        self._unregister_handles = []
        self._client = MusicRaterClient(
            url=cast("str", config.get_value(CONF_URL)),
            session=mass.http_session,
            verify_ssl=bool(config.get_value(CONF_VERIFY_SSL, True)),
        )

    async def loaded_in_mass(self) -> None:
        """Probe music-rater and register the WebSocket command."""
        try:
            await self._client.ping()
        except Exception as err:
            raise ProviderUnavailableError(
                f"Could not reach music-rater at {self.config.get_value(CONF_URL)}: {err}"
            ) from err

        self._unregister_handles.append(
            self.mass.register_api_command(
                "lidarr/add_album", self.add_album, required_role="admin"
            )
        )

    async def unload(self, is_removed: bool = False) -> None:
        """Drop the registered command handler."""
        for unregister in self._unregister_handles:
            unregister()
        self._unregister_handles.clear()
        await super().unload(is_removed)

    # ----- public API command -----

    async def add_album(self, item: str) -> dict[str, Any]:
        """Send an album to Lidarr via music-rater.

        - Resolves the album from MA (`item` is an MA URI) for the toast labels.
        - Looks up the music-rater album_id by URI; falls back to text search.
        - POSTs to music-rater's lidarr/queue endpoint and maps the response
          back to the frontend's LidarrAddAlbumResult shape.
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
        primary_artist = album.artists[0]
        artist_name = getattr(primary_artist, "name", None) if primary_artist else None
        if not artist_name:
            raise InvalidDataError(f"Album {album.name!r} has no usable artist name")

        album_id = await self._resolve_album_id(
            album_uri=item, artist_name=artist_name, album_name=album.name
        )
        response = await self._client.queue_lidarr(album_id)
        return self._build_result(response, artist_name=artist_name, album_name=album.name)

    # ----- helpers -----

    async def _resolve_album_id(self, *, album_uri: str, artist_name: str, album_name: str) -> int:
        """Resolve a music-rater album_id with URI-then-search fallback."""
        album_id = await self._client.resolve_by_uri(album_uri)
        if album_id is not None:
            self.logger.debug("music-rater resolved URI=%s -> album_id=%d", album_uri, album_id)
            return album_id

        # URI didn't hit. Music-rater hasn't synced this album to MA yet (likely
        # added by hand). Best-effort text search.
        query = f"{artist_name} {album_name}".strip()
        self.logger.info(
            "music-rater URI lookup empty for %r — falling back to search %r",
            album_uri,
            query,
        )
        album_id = await self._client.resolve_by_search(query)
        if album_id is None:
            raise InvalidDataError(
                f"music-rater doesn't know {artist_name!r} - {album_name!r}. "
                "Sync it from music-rater to MA first, or add it to music-rater."
            )
        return album_id

    @staticmethod
    def _as_count(value: Any) -> int:
        """Coerce a music-rater counter field to int; default to 0 on garbage."""
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _build_result(
        self,
        response: dict[str, Any],
        *,
        artist_name: str,
        album_name: str,
    ) -> dict[str, Any]:
        """Map music-rater's LidarrQueueResponse to the frontend's LidarrAddAlbumResult.

        music-rater fields we read:
          artists_added, albums_monitored, skipped, errors, error_log, lidarr_synced
        """
        artists_added = self._as_count(response.get("artists_added"))
        albums_monitored = self._as_count(response.get("albums_monitored"))
        skipped = self._as_count(response.get("skipped"))
        errors = self._as_count(response.get("errors"))
        lidarr_synced = bool(response.get("lidarr_synced"))

        if errors > 0:
            err_log = response.get("error_log") or "(no error log)"
            raise ProviderUnavailableError(
                f"music-rater reported {errors} Lidarr error(s): {err_log}"
            )
        if skipped > 0:
            base = str(self.config.get_value(CONF_URL) or "").rstrip("/")
            raise InvalidDataError(
                f"Lidarr couldn't match {artist_name!r} - {album_name!r}. "
                f"Resolve manually at {base}/lidarr/unmatched."
            )

        # albums_monitored == 1 → newly monitored now.
        # lidarr_synced=True with albums_monitored == 0 → idempotent re-call,
        # i.e. the album was already monitored before this request.
        already_monitored = lidarr_synced and albums_monitored == 0
        # No counters and no sync flag means music-rater accepted the POST but
        # neither monitored nor reported activity — treat that as an unexpected
        # backend response rather than a silent success.
        if not (albums_monitored > 0 or already_monitored or artists_added > 0):
            raise ProviderUnavailableError(
                f"music-rater returned no-op for {artist_name!r} - {album_name!r} "
                "(no errors, no monitors, no sync). Check music-rater logs."
            )
        return {
            "artist_name": artist_name,
            "album_name": album_name,
            "artist_added": artists_added > 0,
            "album_monitored": albums_monitored > 0 or already_monitored,
            "already_monitored": already_monitored,
            "lidarr_instance": self.name,
        }
