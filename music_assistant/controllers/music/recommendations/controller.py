"""Recommendations subcontroller: aggregates library + provider recommendation rows."""

from __future__ import annotations

import asyncio
import logging
from itertools import zip_longest
from typing import TYPE_CHECKING, cast

from music_assistant_models.auth import Scope
from music_assistant_models.enums import MediaType, ProviderFeature
from music_assistant_models.media_items import Album, ItemMapping, UniqueList

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
        MediaItemType,
        RecommendationFolder,
    )

    from music_assistant.mass import MusicAssistant
    from music_assistant.models.media_capabilities import RecommendationsMixin


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
                    self._provider_rows(cast("RecommendationsMixin", provider))
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
                    items = await cast("RecommendationsMixin", prov).get_recommendation_items(
                        item_id
                    )
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
        Give album row items the critical-reception and dynamic-range the library holds.

        Two shapes arrive without badge data and need different treatment:

        - A provider album (subsonic, tidal) that was never matched to its library
          counterpart. Graft the two metadata values across and leave item_id, provider
          and uri alone, so navigation, playback and de-duplication are unaffected.
        - A minimized library row -- ``recently_played`` and friends return ItemMapping,
          which has no metadata field at all, so there is nowhere to graft. Swap in the
          full library album instead. It wears the identical ``library://album/<id>``
          uri, so this is the same item, only complete.

        Never raises and never drops items: on timeout or lookup failure the row is
        returned as it arrived.
        """
        upgrades: list[int] = []
        grafts: list[int] = []
        for index, item in enumerate(items):
            if item.media_type != MediaType.ALBUM:
                continue
            if isinstance(item, ItemMapping):
                # only library mappings may be swapped: replacing a provider mapping
                # would rewrite its uri and move the card to a different album page
                if item.provider == "library":
                    upgrades.append(index)
                continue
            if item.provider == "library":
                continue
            if (
                item.metadata.critical_reception is not None
                or item.metadata.dynamic_range is not None
            ):
                continue
            grafts.append(index)
        targets = upgrades + grafts
        if not targets:
            return items
        try:
            async with asyncio.timeout(RECOMMENDATIONS_ENRICH_TIMEOUT):
                matches = await asyncio.gather(
                    *(
                        self.mass.music.albums.get_library_item_by_prov_id(
                            items[index].item_id, items[index].provider
                        )
                        for index in targets
                    ),
                    return_exceptions=True,
                )
        except TimeoutError:
            self.logger.debug("Timeout enriching album badges; serving the row unenriched")
            return items
        enriched = list(items)
        for index, match in zip(targets, matches, strict=True):
            if not isinstance(match, Album) or match.metadata is None:
                continue
            if index in upgrades:
                enriched[index] = match
                continue
            item = enriched[index]
            if isinstance(item, ItemMapping):
                continue
            if match.metadata.critical_reception is not None:
                item.metadata.critical_reception = match.metadata.critical_reception
            if match.metadata.dynamic_range is not None:
                item.metadata.dynamic_range = match.metadata.dynamic_range
        return UniqueList(enriched)

    async def _provider_rows(self, provider: RecommendationsMixin) -> list[RecommendationFolder]:
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
