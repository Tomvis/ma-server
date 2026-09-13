"""Tests for the Listen Later plugin's Discover row."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import ProviderFeature, ProviderType
from music_assistant_models.media_items import UniqueList

from music_assistant.providers.listen_later import (
    ROW_ID,
    ROW_ITEM_LIMIT,
    SUPPORTED_FEATURES,
    ListenLaterProvider,
)

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant


@pytest.fixture
def plugin() -> tuple[ListenLaterProvider, MagicMock]:
    """Construct the provider with a stubbed mass/manifest/config."""
    mass = MagicMock()
    manifest = MagicMock()
    manifest.type = ProviderType.PLUGIN
    manifest.domain = "listen_later"
    config = MagicMock()
    config.name = "Listen Later"
    config.instance_id = "listen_later"
    config.get_value = MagicMock(return_value="GLOBAL")
    mass.music.albums.library_items = AsyncMock(return_value=[])
    return ListenLaterProvider(mass, manifest, config, SUPPORTED_FEATURES), mass


async def test_get_recommendations_returns_exactly_one_row(
    plugin: tuple[ListenLaterProvider, MagicMock],
) -> None:
    """The provider contributes a single Discover row."""
    provider, _ = plugin
    rows = await provider.get_recommendations()
    assert len(rows) == 1


async def test_row_identity_is_stable_and_instance_scoped(
    plugin: tuple[ListenLaterProvider, MagicMock],
) -> None:
    """The row carries the stable item_id and the instance_id as its provider."""
    provider, _ = plugin
    row = (await provider.get_recommendations())[0]
    # User row preferences (order, hidden) are keyed on item_id, and the items API
    # resolves the provider from the `provider` field -- both must not drift.
    assert row.item_id == "listen_later"
    assert row.provider == "listen_later"
    assert row.translation_key == "listen_later"


async def test_get_recommendation_items_delegates_to_the_albums_controller(
    plugin: tuple[ListenLaterProvider, MagicMock],
) -> None:
    """Row items come from the existing library query, not from new SQL."""
    provider, mass = plugin
    items = await provider.get_recommendation_items(ROW_ID)
    assert isinstance(items, UniqueList)
    mass.music.albums.library_items.assert_awaited_once_with(
        listen_later=True,
        order_by="listen_later_added_at_desc",
        limit=ROW_ITEM_LIMIT,
    )


async def test_get_recommendation_items_ignores_an_unknown_row(
    plugin: tuple[ListenLaterProvider, MagicMock],
) -> None:
    """An unknown item_id returns empty rather than raising or querying."""
    provider, mass = plugin
    assert await provider.get_recommendation_items("not_a_row") == []
    mass.music.albums.library_items.assert_not_awaited()


async def test_provider_loads_with_the_recommendations_feature(
    mass: MusicAssistant,
) -> None:
    """The provider loads into a running server and declares RECOMMENDATIONS."""
    # The builtin provider may already be auto-loaded by the mass fixture.
    # If so, fetch it; otherwise create it.
    try:
        await mass.config._create_provider_instance("listen_later", {})
    except ValueError as e:
        if "does not support multiple instances" not in str(e):
            raise
    provider = mass.get_provider("listen_later", provider_type=ListenLaterProvider)
    assert provider is not None
    await provider.initialized.wait()
    assert ProviderFeature.RECOMMENDATIONS in provider.supported_features
