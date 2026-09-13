"""Tests for grafting library album badges onto provider recommendation items."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from music_assistant_models.enums import AlbumType, MediaType
from music_assistant_models.media_items import Album, ItemMapping, Track, UniqueList
from music_assistant_models.media_items.metadata import (
    CriticalReception,
    MediaItemMetadata,
)

from music_assistant.controllers.music.recommendations.controller import (
    RecommendationsController,
)

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant


def _album(
    item_id: str,
    provider: str,
    *,
    cr: CriticalReception | None = None,
    dr: float | None = None,
) -> Album:
    """Build an album with optional badge metadata."""
    return Album(
        item_id=item_id,
        provider=provider,
        name=f"Album {item_id}",
        album_type=AlbumType.ALBUM,
        provider_mappings=set(),
        metadata=MediaItemMetadata(critical_reception=cr, dynamic_range=dr),
    )


@pytest.fixture
def controller(mass: MusicAssistant) -> RecommendationsController:
    """Return the live recommendations controller."""
    return mass.music.recommendations


async def test_badges_are_grafted_from_the_library_match(
    controller: RecommendationsController,
) -> None:
    """A provider album with no badge data inherits it from its library counterpart."""
    item = _album("abc", "opensubsonic--x")
    library = _album("358", "library", cr=CriticalReception(amg_dr=12.0), dr=12.0)
    controller.mass.music.albums.get_library_item_by_prov_id = AsyncMock(return_value=library)  # type: ignore[method-assign]

    out = await controller._attach_library_album_badges(UniqueList([item]))

    assert out[0].metadata.dynamic_range == 12.0
    assert out[0].metadata.critical_reception == CriticalReception(amg_dr=12.0)


async def test_item_identity_is_never_rewritten(
    controller: RecommendationsController,
) -> None:
    """Grafting must not touch item_id, provider or uri -- navigation depends on them."""
    item = _album("abc", "opensubsonic--x")
    before = (item.item_id, item.provider, item.uri)
    library = _album("358", "library", dr=12.0)
    controller.mass.music.albums.get_library_item_by_prov_id = AsyncMock(return_value=library)  # type: ignore[method-assign]

    out = await controller._attach_library_album_badges(UniqueList([item]))

    assert (out[0].item_id, out[0].provider, out[0].uri) == before


async def test_items_that_already_carry_badges_are_not_looked_up(
    controller: RecommendationsController,
) -> None:
    """An item with its own data needs no query -- this is the common library-row case."""
    item = _album("abc", "opensubsonic--x", dr=9.0)
    lookup = AsyncMock()
    controller.mass.music.albums.get_library_item_by_prov_id = lookup  # type: ignore[method-assign]

    out = await controller._attach_library_album_badges(UniqueList([item]))

    lookup.assert_not_awaited()
    assert out[0].metadata.dynamic_range == 9.0


async def test_library_items_are_skipped(controller: RecommendationsController) -> None:
    """Library-provider items are already the source of truth; no lookup is issued."""
    lookup = AsyncMock()
    controller.mass.music.albums.get_library_item_by_prov_id = lookup  # type: ignore[method-assign]

    await controller._attach_library_album_badges(UniqueList([_album("1", "library")]))

    lookup.assert_not_awaited()


async def test_non_albums_are_skipped(controller: RecommendationsController) -> None:
    """Only albums carry these badges; tracks and mappings must not be queried."""
    lookup = AsyncMock()
    controller.mass.music.albums.get_library_item_by_prov_id = lookup  # type: ignore[method-assign]
    track = Track(item_id="t1", provider="tidal--x", name="A Track", provider_mappings=set())
    mapping = ItemMapping(
        item_id="m1", provider="tidal--x", name="A Map", media_type=MediaType.ALBUM
    )

    out = await controller._attach_library_album_badges(UniqueList([track, mapping]))

    lookup.assert_not_awaited()
    assert len(out) == 2


async def test_no_library_match_leaves_the_item_untouched(
    controller: RecommendationsController,
) -> None:
    """An album that is not in the library keeps its empty metadata."""
    item = _album("abc", "tidal--x")
    controller.mass.music.albums.get_library_item_by_prov_id = AsyncMock(return_value=None)  # type: ignore[method-assign]

    out = await controller._attach_library_album_badges(UniqueList([item]))

    assert out[0].metadata.dynamic_range is None
    assert out[0].metadata.critical_reception is None


async def test_a_failing_lookup_never_drops_the_row(
    controller: RecommendationsController,
) -> None:
    """A lookup error degrades to no badges, never to a lost item."""
    item = _album("abc", "opensubsonic--x")
    controller.mass.music.albums.get_library_item_by_prov_id = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("db exploded")
    )

    out = await controller._attach_library_album_badges(UniqueList([item]))

    assert len(out) == 1
    assert out[0].metadata.dynamic_range is None


async def test_library_item_mapping_albums_are_upgraded_to_full_albums(
    controller: RecommendationsController,
) -> None:
    """recently_played returns minimized library rows; swap in the full album."""
    mapping = ItemMapping(
        item_id="358", provider="library", name="Album 358", media_type=MediaType.ALBUM
    )
    library = _album("358", "library", cr=CriticalReception(amg_dr=12.0), dr=12.0)
    controller.mass.music.albums.get_library_item_by_prov_id = AsyncMock(  # type: ignore[method-assign]
        return_value=library
    )

    out = await controller._attach_library_album_badges(UniqueList([mapping]))

    assert isinstance(out[0], Album)
    assert out[0].metadata.dynamic_range == 12.0
    # the swap is only safe because the uri is identical -- same item, just complete
    assert out[0].uri == mapping.uri


async def test_provider_item_mapping_albums_are_left_alone(
    controller: RecommendationsController,
) -> None:
    """Swapping a provider mapping would rewrite its uri and move the card's target."""
    mapping = ItemMapping(
        item_id="abc", provider="tidal--x", name="Album abc", media_type=MediaType.ALBUM
    )
    lookup = AsyncMock()
    controller.mass.music.albums.get_library_item_by_prov_id = lookup  # type: ignore[method-assign]

    out = await controller._attach_library_album_badges(UniqueList([mapping]))

    lookup.assert_not_awaited()
    assert out[0] is mapping
