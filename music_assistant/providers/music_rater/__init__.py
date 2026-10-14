"""
music-rater plugin provider.

Navidrome albums get their critical reception from the review tags music-rater
stamps into the files. Albums known only through a streaming catalog (Tidal,
Bandcamp) have no files, so their CR was a one-off snapshot and went stale
(MUSIC-79). This provider refreshes it daily from music-rater's
``GET /tags/for-album`` -- the same field set reviewsync writes into files --
parsed by the same tag parser, so both kinds of album read alike.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Final, cast

from music_assistant_models.background_task import TaskSchedule
from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType

from music_assistant.helpers.critical_reception import critical_reception_for_update
from music_assistant.helpers.tags import critical_reception_from_tags
from music_assistant.models.plugin import PluginProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.enums import ProviderFeature
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

CONF_URL: Final[str] = "url"
CONF_API_KEY: Final[str] = "api_key"
REFRESH_TASK_ID: Final[str] = "music_rater_review_refresh"
# Provider domains whose albums carry music-rater's tags in their files.
FILE_TAGGED_DOMAINS: Final[tuple[str, ...]] = ("opensubsonic", "filesystem")
# Streaming catalogs often date a release a year off the review.
YEAR_OFFSETS: Final[tuple[int, ...]] = (0, -1, 1)


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return MusicRaterProvider(mass, manifest, config, set())


class MusicRaterProvider(PluginProvider):
    """Refreshes review data on library albums that have no tagged files."""

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
            ConfigEntry(key=CONF_URL, type=ConfigEntryType.STRING, required=True),
            ConfigEntry(key=CONF_API_KEY, type=ConfigEntryType.SECURE_STRING, required=True),
        )

    async def handle_async_init(self) -> None:
        """Schedule the daily refresh, after reviewsync's nightly tag run."""
        self.mass.tasks.register_scheduled_task(
            task_id=REFRESH_TASK_ID,
            name="Refresh music-rater review data",
            handler=self.sync_review_data,
            schedule=TaskSchedule.daily(hour=3),
            initial_delay=300,
        )

    async def unload(self, is_removed: bool = False) -> None:
        """Drop the scheduled refresh."""
        self.mass.tasks.unregister_scheduled_task(REFRESH_TASK_ID, clear_persisted_state=is_removed)

    async def sync_review_data(self) -> None:
        """Write music-rater's current review data to every streaming-only album."""
        stats: Counter[str] = Counter()
        # Streaming mappings of these rows are mostly in_library=False (the old
        # household Listen Later shelf never added them to the Tidal/Bandcamp library),
        # so iterate every album row, not just the provider-library ones.
        async for album in self.mass.music.albums.iter_library_items(library_items_only=False):
            if any(
                m.provider_domain.startswith(FILE_TAGGED_DOMAINS) for m in album.provider_mappings
            ):
                continue
            if not album.year or not album.artists:
                stats["no-identity"] += 1
                continue
            fields = await self._fetch_fields(album.artists[0].name, album.name, album.year)
            if fields is None:
                stats["no-match"] += 1
                continue
            stored = album.metadata.critical_reception
            new = critical_reception_for_update(critical_reception_from_tags(fields), stored)
            if new == stored:
                stats["unchanged"] += 1
                continue
            album.metadata.critical_reception = new
            await self.mass.music.albums.update_item_in_library(album.item_id, album)
            stats["updated"] += 1
        self.logger.info("Review data refresh: %s", dict(stats))

    async def _fetch_fields(self, artist: str, album: str, year: int) -> dict[str, Any] | None:
        """
        Return music-rater's tag fields for an album, or None when it has none.

        A 404 means no album (or no review data) for that identity; anything else
        that is not a 200 raises, ending the run rather than failing every album.
        """
        for offset in YEAR_OFFSETS:
            async with self.mass.http_session.get(
                f"{self._url}/tags/for-album",
                params={"artist": artist, "album": album, "year": year + offset},
                headers={"Authorization": f"Bearer {self._api_key}"},
            ) as resp:
                if resp.status == 404:
                    continue
                resp.raise_for_status()
                payload = await resp.json()
                return cast("dict[str, Any]", payload["fields"])
        return None
