"""Recommendations subcontroller: aggregates library + provider recommendation rows."""

from __future__ import annotations

import asyncio
import logging
from itertools import zip_longest
from typing import TYPE_CHECKING, cast

from music_assistant_models.auth import Scope
from music_assistant_models.enums import MediaType, ProviderFeature
from music_assistant_models.media_items import Album, UniqueList

from music_assistant.constants import MASS_LOGGER_NAME
from music_assistant.controllers.music.constants import (
    RECOMMENDATIONS_ENRICH_TIMEOUT,
    RECOMMENDATIONS_ITEMS_TIMEOUT,
    RECOMMENDATIONS_ROWS_TIMEOUT,
)
from music_assistant.providers.recommendations import LibraryRecommendationsProvider

if TYPE_CHECKING:
    from music_assistant_models.media_items import (
        BrowseFolder,
        ItemMapping,
        MediaItemType,
        RecommendationFolder,
    )

    from music_assistant.mass import MusicAssistant
    from music_assistant.models.metadata_provider import MetadataProvider
    from music_assistant.models.music_provider import MusicProvider
    from music_assistant.models.plugin import PluginProvider


class RecommendationsController:
    """Serves the recommendations API: default library rows plus provider rows."""

    def __init__(self, mass: MusicAssistant) -> None:
        """Initialize the controller and register its api commands."""
        self.mass = mass
        self.logger = logging.getLogger(f"{MASS_LOGGER_NAME}.music.recommendations")
        self.mass.register_api_command(
            "music/recommendations",
            self.get_recommendations,
            required_scope=Scope.LIBRARY_READ,
        )
        self.mass.register_api_command(
            "music/recommendations/items",
            self.get_recommendation_items,
            required_scope=Scope.LIBRARY_READ,
        )

    async def get_recommendations(self) -> list[RecommendationFolder]:
        """Get all available recommendation rows (library + providers, interleaved), without items."""
        providers = self.mass.music._apply_user_provider_filter(
            self.mass.get_providers_supporting_feature(ProviderFeature.RECOMMENDATIONS)
        )
        rows_per_source: list[list[RecommendationFolder]] = [
            *await asyncio.gather(
                *[
                    self._provider_rows(
                        cast("MusicProvider | MetadataProvider | PluginProvider", provider)
                    )
                    for provider in providers
                ]
            ),
        ]
        # interleave: one folder per source per pass, preserving each source's ordering
        return [item for sublist in zip_longest(*rows_per_source) for item in sublist if item]

    async def get_recommendation_items(
        self, provider: str, item_id: str, providers: list[str] | None = None
    ) -> UniqueList[MediaItemType | ItemMapping | BrowseFolder]:
        """
        Get the items for a single recommendation row.

        :param provider: The provider instance id owning the row.
        :param item_id: The item_id of the row, as returned by the recommendations listing.
        :param providers: Restrict items to those reachable through one of these provider
            instance ids (OR semantics). Only honored on rows that advertise
            `supports_provider_filter`; ignored on other rows for backwards compatibility.
        """
        try:
            prov = self.mass.get_provider(provider)
            # re-apply the user provider filter the rows listing applies, so a user
            # can not fetch items from a music provider an admin has restricted them from
            if prov is None or not self.mass.music._apply_user_provider_filter([prov]):
                return UniqueList()
            if ProviderFeature.RECOMMENDATIONS not in prov.supported_features:
                # keep the base-model guarantee that this method is only called for
                # providers declaring the feature, matching the rows listing
                return UniqueList()
            async with asyncio.timeout(RECOMMENDATIONS_ITEMS_TIMEOUT):
                if isinstance(prov, LibraryRecommendationsProvider):
                    items = await prov.get_recommendation_items(item_id, providers=providers)
                else:
                    # external provider rows don't support provider filtering: their SPI
                    # signature is unchanged, so `providers` is silently ignored here
                    items = await cast(
                        "MusicProvider | MetadataProvider | PluginProvider", prov
                    ).get_recommendation_items(item_id)
            # deliberately outside the items timeout: enrichment is a nicety, and must
            # never be able to cost us items we have already fetched
            return await self._attach_library_album_badges(items)
        except TimeoutError:
            self.logger.warning(
                "Timeout while fetching recommendation items for %s/%s; skipping",
                provider,
                item_id,
            )
            return UniqueList()
        except Exception as err:
            self.logger.warning(
                "Error while fetching recommendation items for %s/%s: %s",
                provider,
                item_id,
                str(err),
                exc_info=err if self.logger.isEnabledFor(logging.DEBUG) else None,
            )
            return UniqueList()

    async def _attach_library_album_badges(
        self, items: UniqueList[MediaItemType | ItemMapping | BrowseFolder]
    ) -> UniqueList[MediaItemType | ItemMapping | BrowseFolder]:
        """
        Copy library critical-reception and dynamic-range onto provider album items.

        Rows served straight from a streaming or subsonic provider hand back items that
        were never matched to their library counterpart, so the album badges the clients
        render have nothing to read -- even when the very same album sits in the library
        carrying the data. Graft only those two metadata values across, leaving item_id,
        provider and uri untouched so navigation, playback and de-duplication behave
        exactly as before.

        Never raises and never drops items: on timeout or lookup failure the row is
        returned as it arrived.
        """
        targets = [
            item
            for item in items
            if item.media_type == MediaType.ALBUM
            and item.provider != "library"
            and getattr(item, "metadata", None) is not None
            and item.metadata.critical_reception is None
            and item.metadata.dynamic_range is None
        ]
        if not targets:
            return items
        try:
            async with asyncio.timeout(RECOMMENDATIONS_ENRICH_TIMEOUT):
                matches = await asyncio.gather(
                    *(
                        self.mass.music.albums.get_library_item_by_prov_id(
                            item.item_id, item.provider
                        )
                        for item in targets
                    ),
                    return_exceptions=True,
                )
        except TimeoutError:
            self.logger.debug("Timeout enriching album badges; serving the row unenriched")
            return items
        for item, match in zip(targets, matches, strict=True):
            if not isinstance(match, Album) or match.metadata is None:
                continue
            if match.metadata.critical_reception is not None:
                item.metadata.critical_reception = match.metadata.critical_reception
            if match.metadata.dynamic_range is not None:
                item.metadata.dynamic_range = match.metadata.dynamic_range
        return items

    async def _provider_rows(
        self, provider: MusicProvider | MetadataProvider | PluginProvider
    ) -> list[RecommendationFolder]:
        """Return a provider's recommendation rows, or an empty list if it times out or raises."""
        try:
            async with asyncio.timeout(RECOMMENDATIONS_ROWS_TIMEOUT):
                return await provider.get_recommendations()
        except TimeoutError:
            self.logger.warning(
                "Timeout while fetching recommendation rows from %s; skipping for this request",
                provider.name,
            )
            return []
        except Exception as err:
            self.logger.warning(
                "Error while fetching recommendation rows from %s: %s",
                provider.name,
                str(err),
                exc_info=err if self.logger.isEnabledFor(logging.DEBUG) else None,
            )
            return []
