"""Recommendations subcontroller: aggregates library + provider recommendation rows."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import replace
from itertools import zip_longest
from typing import TYPE_CHECKING, cast

from music_assistant_models.auth import Scope
from music_assistant_models.enums import MediaType, ProviderFeature
from music_assistant_models.media_items import (
    Album,
    Audiobook,
    BrowseFolder,
    ItemMapping,
    PodcastEpisode,
    UniqueList,
)

from music_assistant.constants import MASS_LOGGER_NAME
from music_assistant.controllers.music.constants import (
    RECOMMENDATIONS_BROWSE_LIMIT,
    RECOMMENDATIONS_BROWSE_SCHEME,
    RECOMMENDATIONS_ENRICH_TIMEOUT,
    RECOMMENDATIONS_ITEMS_TIMEOUT,
    RECOMMENDATIONS_ROWS_TIMEOUT,
)
from music_assistant.controllers.webserver.helpers.auth_middleware import get_current_user
from music_assistant.helpers.provider_access import visible_items
from music_assistant.providers.recommendations import LibraryRecommendationsProvider

if TYPE_CHECKING:
    from music_assistant_models.media_items import (
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
            allow_impersonation=True,
        )
        self.mass.register_api_command(
            "music/recommendations/items",
            self.get_recommendation_items,
            required_scope=Scope.LIBRARY_READ,
            allow_impersonation=True,
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
        self,
        provider: str,
        item_id: str,
        providers: list[str] | None = None,
        hide_played: bool = False,
    ) -> UniqueList[MediaItemType | ItemMapping | BrowseFolder]:
        """
        Get the items for a single recommendation row.

        :param provider: The provider instance id owning the row.
        :param item_id: The item_id of the row, as returned by the recommendations listing.
        :param providers: Restrict items to those reachable through one of these provider
            instance ids (OR semantics). Only honored on rows that advertise
            `supports_provider_filter`; ignored on other rows for backwards compatibility.
        :param hide_played: Leave out the podcast episodes and audiobooks that are fully played.
            Rows that advertise `supports_hide_played` are the ones to offer this on.
        """
        return await self._get_row_items(provider, item_id, providers, hide_played=hide_played)

    async def browse(
        self, sub_path: str, random_seed: int | None = None
    ) -> list[MediaItemType | ItemMapping | BrowseFolder]:
        """
        Browse the recommendation rows under the discover:// path.

        :param sub_path: The path after discover://: empty for the providers owning rows,
            "<provider>" for the rows of that provider, "<provider>/<item_id>" for all
            items of that row.
        :param random_seed: Keep the order of a random row stable across pages.
        """
        provider, _, item_id = sub_path.partition("/")
        if item_id:
            # the browse view pages through the row, so a random row needs a seed even
            # when the caller does not ask for one: its own identity keeps it stable
            seed = random_seed if random_seed is not None else _row_seed(provider, item_id)
            return list(
                await self._get_row_items(
                    provider,
                    item_id,
                    limit=RECOMMENDATIONS_BROWSE_LIMIT,
                    random_seed=seed,
                )
            )
        rows = await self.get_recommendations()
        if provider:
            return [
                # a row stays a RecommendationFolder so its name translates as a row
                replace(
                    row,
                    path=f"{RECOMMENDATIONS_BROWSE_SCHEME}://{row.provider}/{row.item_id}",
                    items=UniqueList(),
                )
                for row in rows
                if row.provider == provider
            ]
        folders: list[MediaItemType | ItemMapping | BrowseFolder] = []
        for row_provider in dict.fromkeys(row.provider for row in rows):
            if not (prov := self.mass.get_provider(row_provider)):
                continue
            folders.append(
                BrowseFolder(
                    item_id=row_provider,
                    provider=prov.domain,
                    path=f"{RECOMMENDATIONS_BROWSE_SCHEME}://{row_provider}",
                    name=prov.name,
                )
            )
        return folders

    async def _get_row_items(
        self,
        provider: str,
        item_id: str,
        providers: list[str] | None = None,
        limit: int | None = None,
        random_seed: int | None = None,
        hide_played: bool = False,
    ) -> UniqueList[MediaItemType | ItemMapping | BrowseFolder]:
        """
        Get the items for a single recommendation row, empty when it can not be fetched.

        :param provider: The provider instance id owning the row.
        :param item_id: The item_id of the row.
        :param providers: Restrict items to those reachable through one of these provider
            instance ids, on rows that support it.
        :param limit: The maximum number of items a library row returns, None for its default.
        :param random_seed: Keep the order of a random row stable across pages.
        :param hide_played: Leave out the podcast episodes and audiobooks that are fully played.
        """
        # the rows listing leaves out the music sources the user may not see, and only lists
        # providers declaring the feature; a row of any other provider does not exist, nor
        # does another account of the same service stand in for the rows of a hidden one
        # (a row naming its service by domain is served by a visible account of it)
        prov = self.mass.music.get_visible_provider(provider)
        if (
            prov is None
            or provider not in (prov.instance_id, prov.domain)
            or ProviderFeature.RECOMMENDATIONS not in prov.supported_features
        ):
            return UniqueList()
        try:
            async with asyncio.timeout(RECOMMENDATIONS_ITEMS_TIMEOUT):
                if isinstance(prov, LibraryRecommendationsProvider):
                    if limit is None:
                        items = await prov.get_recommendation_items(item_id, providers=providers)
                    else:
                        items = await prov.get_recommendation_items(
                            item_id, providers=providers, limit=limit, random_seed=random_seed
                        )
                else:
                    # external provider rows don't support provider filtering, a limit or a
                    # seed: their SPI signature is unchanged, so those are ignored here
                    items = await cast("RecommendationsMixin", prov).get_recommendation_items(
                        item_id
                    )
                    # a plugin or metadata row can list items on any music source
                    items = UniqueList(visible_items(self.mass, items, get_current_user()))
            if hide_played:
                items = _without_played(items)
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


def _row_seed(provider: str, item_id: str) -> int:
    """
    Return the seed that keeps a row's random order stable across pages.

    :param provider: The provider instance id owning the row.
    :param item_id: The item_id of the row.
    """
    digest = hashlib.blake2s(f"{provider}/{item_id}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def _without_played(
    items: UniqueList[MediaItemType | ItemMapping | BrowseFolder],
) -> UniqueList[MediaItemType | ItemMapping | BrowseFolder]:
    """Return the items without the podcast episodes and audiobooks that are fully played."""
    return UniqueList(
        item
        for item in items
        if not (isinstance(item, PodcastEpisode | Audiobook) and item.fully_played)
    )
