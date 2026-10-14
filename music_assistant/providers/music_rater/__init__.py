"""
music-rater plugin provider.

The only source of album critical reception (MUSIC-84). Every library album is
refreshed daily from music-rater's tag routes -- the field set reviewsync stamps
into the files, parsed by the same tag parser -- instead of reading it back out
of Navidrome's files (MUSIC-78..82) or keeping a one-off snapshot for albums
known only through a streaming catalog (MUSIC-79).
"""

from __future__ import annotations

from collections import Counter
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final, cast

from music_assistant_models.background_task import TaskSchedule
from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType, ExternalID, SortDirection, SortField

from music_assistant.helpers.critical_reception import critical_reception_for_update
from music_assistant.helpers.datetime import utc
from music_assistant.helpers.tags import critical_reception_from_tags
from music_assistant.models.plugin import PluginProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.enums import ProviderFeature
    from music_assistant_models.media_items import Album
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

CONF_URL: Final[str] = "url"
CONF_API_KEY: Final[str] = "api_key"
REFRESH_TASK_ID: Final[str] = "music_rater_review_refresh"
RECENT_TASK_ID: Final[str] = "music_rater_recent_review_refresh"
# Albums added this recently are re-read every hour, so a new album shows its review
# data soon after import instead of after the next daily run. A day covers any gap.
RECENT_WINDOW: Final[timedelta] = timedelta(days=1)
# Streaming catalogs often date a release a year off the review.
YEAR_OFFSETS: Final[tuple[int, ...]] = (0, -1, 1)


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return MusicRaterProvider(mass, manifest, config, set())


class MusicRaterProvider(PluginProvider):
    """Refreshes review data on library albums."""

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
        supported_features: set[ProviderFeature] | None = None,
    ) -> None:
        """Initialize the provider."""
        super().__init__(mass, manifest, config, supported_features)
        self._url = cast("str", config.get_value(CONF_URL, "")).rstrip("/")
        self._api_key = cast("str", config.get_value(CONF_API_KEY, ""))

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return the config entries for this provider."""
        return (
            # Not required: the instance is created with no values before the options
            # page can be filled, and load-time validation would reject it.
            ConfigEntry(key=CONF_URL, type=ConfigEntryType.STRING, required=False),
            ConfigEntry(key=CONF_API_KEY, type=ConfigEntryType.SECURE_STRING, required=False),
        )

    async def handle_async_init(self) -> None:
        """Schedule the daily refresh and the hourly one of newly added albums."""
        self.mass.tasks.register_scheduled_task(
            task_id=REFRESH_TASK_ID,
            name="Refresh music-rater review data",
            handler=self.sync_review_data,
            schedule=TaskSchedule.daily(hour=3),
            initial_delay=300,
        )
        self.mass.tasks.register_scheduled_task(
            task_id=RECENT_TASK_ID,
            name="Refresh music-rater review data of new albums",
            handler=self.sync_recent_review_data,
            schedule=TaskSchedule.hourly(),
            initial_delay=600,
        )

    async def unload(self, is_removed: bool = False) -> None:
        """Drop the scheduled refreshes."""
        for task_id in (REFRESH_TASK_ID, RECENT_TASK_ID):
            self.mass.tasks.unregister_scheduled_task(task_id, clear_persisted_state=is_removed)

    async def sync_review_data(self) -> None:
        """Write music-rater's current review data to every library album."""
        if not (self._url and self._api_key):
            self.logger.warning("music-rater URL or API key not set; skipping the refresh")
            return
        stats: Counter[str] = Counter()
        # Streaming mappings of these rows are mostly in_library=False (the old
        # household Listen Later shelf never added them to the Tidal/Bandcamp library),
        # so iterate every album row, not just the provider-library ones.
        async for album in self.mass.music.albums.iter_library_items(library_items_only=False):
            stats[await self._refresh_album(album)] += 1
        self.logger.info("Review data refresh: %s", dict(stats))

    async def sync_recent_review_data(self) -> None:
        """Write music-rater's current review data to the albums added within RECENT_WINDOW."""
        if not (self._url and self._api_key):
            return
        stats: Counter[str] = Counter()
        cutoff = utc() - RECENT_WINDOW
        async for album in self.mass.music.albums.iter_library_items(
            sort_field=SortField.TIMESTAMP_ADDED,
            sort_direction=SortDirection.DESC,
            library_items_only=False,
        ):
            if album.date_added is None or album.date_added < cutoff:
                break
            stats[await self._refresh_album(album)] += 1
        if stats["updated"]:
            self.logger.info("Review data refresh of new albums: %s", dict(stats))

    async def _refresh_album(self, album: Album) -> str:
        """
        Write music-rater's current review data to one album.

        :param album: The library album.
        :return: What happened, as a stats key.
        """
        rgids = sorted(v for t, v in album.external_ids if t == ExternalID.MB_RELEASEGROUP)
        identity = (
            (album.artists[0].name, album.name, album.year)
            if album.year and album.artists
            else None
        )
        if not rgids and identity is None:
            return "no-identity"
        fields = await self._fetch_fields(rgids, identity)
        if fields is None:
            return "no-match"
        stored = album.metadata.critical_reception
        new = critical_reception_for_update(critical_reception_from_tags(fields), stored)
        if new == stored:
            return "unchanged"
        album.metadata.critical_reception = new
        await self.mass.music.albums.update_item_in_library(album.item_id, album)
        return "updated"

    async def _fetch_fields(
        self, rgids: list[str], identity: tuple[str, str, int] | None
    ) -> dict[str, Any] | None:
        """
        Return music-rater's tag fields for an album, or None when it has none.

        The release-group ids are tried first, then the artist/album/year identity. A 404
        means no album (or no review data) for that key; anything else that is not a 200
        raises, ending the run rather than failing every album.

        :param rgids: The album's MusicBrainz release-group ids.
        :param identity: The album's (artist, album, year), if it has a year.
        """
        for rgid in rgids:
            if (fields := await self._get_fields(f"/tags/by-release-group/{rgid}")) is not None:
                return fields
        if identity is None:
            return None
        artist, album, year = identity
        for offset in YEAR_OFFSETS:
            params = {"artist": artist, "album": album, "year": year + offset}
            if (fields := await self._get_fields("/tags/for-album", params)) is not None:
                return fields
        return None

    async def _get_fields(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Return the tag fields of one music-rater lookup, or None on a 404."""
        async with self.mass.http_session.get(
            f"{self._url}{path}",
            params=params,
            headers={"Authorization": f"Bearer {self._api_key}"},
        ) as resp:
            if resp.status == 404:
                return None
            resp.raise_for_status()
            payload = await resp.json()
            return cast("dict[str, Any]", payload["fields"])
