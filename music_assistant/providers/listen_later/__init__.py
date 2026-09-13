"""
Listen Later plugin provider.

Contributes a single Discover row listing the albums on the Listen Later shelf.
The shelf itself lives in the library core (the `albums.listen_later` column);
this provider only surfaces it, delegating every query to the albums controller
so the row can never drift from the dedicated Listen Later view.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from music_assistant_models.enums import EventType, ProviderFeature
from music_assistant_models.media_items import Album, RecommendationFolder, UniqueList

from music_assistant.models.plugin import PluginProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigEntry, ProviderConfig
    from music_assistant_models.event import MassEvent
    from music_assistant_models.media_items import (
        BrowseFolder,
        ItemMapping,
        MediaItemType,
    )
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

SUPPORTED_FEATURES: set[ProviderFeature] = {
    ProviderFeature.RECOMMENDATIONS,
}

# The row's stable identity. Per-user row preferences (order, hidden) are keyed on
# it, so it must not change across restarts or releases.
ROW_ID = "listen_later"

# How many albums the row carries. A Discover row is a glance, not a listing --
# the dedicated /listen-later view remains the full surface.
ROW_ITEM_LIMIT = 16

# The refresh signal the Discover page listens for. The frontend matches on this
# exact string (HomeWidgetRows.vue), so it is shared vocabulary, not ours to rename.
EVENT_RECOMMENDATIONS_UPDATED = "recommendations_updated"


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return ListenLaterProvider(mass, manifest, config, SUPPORTED_FEATURES)


class ListenLaterProvider(PluginProvider):
    """Builtin provider surfacing the Listen Later shelf on Discover."""

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return the (options) config entries for this provider instance."""
        return ()

    async def handle_async_init(self) -> None:
        """Initialise the shelf snapshot and the subscription handle list."""
        self._saved_uris: set[str] = set()
        self._unregister_handles: list[Callable[[], None]] = []

    async def loaded_in_mass(self) -> None:
        """Snapshot the shelf, then watch for changes to it."""
        # Seeding matters: without it the first update seen for each saved album
        # would look like a transition and signal a refresh, so a library sync
        # would storm the Discover page.
        saved = await self.mass.music.albums.library_items(
            listen_later=True, order_by="listen_later_added_at_desc", limit=0
        )
        self._saved_uris = {album.uri for album in saved if album.uri}
        self._unregister_handles.append(
            self.mass.subscribe(self._on_media_item_updated, EventType.MEDIA_ITEM_UPDATED)
        )

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload/close of the provider."""
        for unregister in self._unregister_handles:
            unregister()
        self._unregister_handles.clear()

    async def get_recommendations(self) -> list[RecommendationFolder]:
        """Get this plugin's available recommendation rows, without items."""
        return [
            RecommendationFolder(
                item_id=ROW_ID,
                provider=self.instance_id,
                name="Listen Later",
                translation_key=ROW_ID,
                icon="mdi-bookmark-music",
            )
        ]

    async def get_recommendation_items(
        self, item_id: str
    ) -> UniqueList[MediaItemType | ItemMapping | BrowseFolder]:
        """
        Get the items for a single recommendation row.

        :param item_id: The item_id of the row, as returned by get_recommendations.
        """
        if item_id != ROW_ID:
            return UniqueList()
        albums = await self.mass.music.albums.library_items(
            listen_later=True,
            order_by="listen_later_added_at_desc",
            limit=ROW_ITEM_LIMIT,
        )
        return UniqueList(albums)

    async def _on_media_item_updated(self, event: MassEvent) -> None:
        """Signal a Discover refresh when an album joins or leaves the shelf."""
        item = event.data
        if not isinstance(item, Album) or not item.uri:
            return
        saved = bool(item.listen_later)
        if saved == (item.uri in self._saved_uris):
            return
        if saved:
            self._saved_uris.add(item.uri)
        else:
            self._saved_uris.discard(item.uri)
        self.signal_provider_event({"event": EVENT_RECOMMENDATIONS_UPDATED})
