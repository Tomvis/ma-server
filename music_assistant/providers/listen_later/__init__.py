"""
Listen Later plugin provider.

Contributes a single Discover row listing the albums on the Listen Later shelf.
The shelf itself lives in the library core (the `albums.listen_later` column);
this provider only surfaces it, delegating every query to the albums controller
so the row can never drift from the dedicated Listen Later view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from music_assistant_models.enums import ProviderFeature
from music_assistant_models.media_items import RecommendationFolder, UniqueList

from music_assistant.models.plugin import PluginProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigEntry, ProviderConfig
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
