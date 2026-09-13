"""Tests for the Listen Later plugin's Discover-row refresh signalling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import AlbumType, EventType, ProviderType
from music_assistant_models.event import MassEvent
from music_assistant_models.media_items import Album, Track

from music_assistant.providers.listen_later import (
    EVENT_RECOMMENDATIONS_UPDATED,
    SUPPORTED_FEATURES,
    ListenLaterProvider,
)


def _album(uri: str, *, listen_later: bool) -> Album:
    """Build a minimal Album carrying a listen_later flag."""
    return Album(
        item_id=uri.rsplit("/", 1)[-1],
        provider="library",
        name="Kind of Blue",
        album_type=AlbumType.ALBUM,
        provider_mappings=set(),
        uri=uri,
        listen_later=listen_later,
    )


def _event(item: Album | Track) -> MassEvent:
    """Wrap a media item in a MEDIA_ITEM_UPDATED event."""
    return MassEvent(event=EventType.MEDIA_ITEM_UPDATED, object_id=item.uri, data=item)


@pytest.fixture
async def plugin() -> tuple[ListenLaterProvider, MagicMock, MagicMock, MagicMock]:
    """
    Construct the provider with a stubbed mass, already initialised and loaded.

    Returns the provider plus three standalone mocks -- the `signal_provider_event`
    replacement, the unregister callable `mass.subscribe()` hands back, and the
    stubbed `mass` itself -- so tests can assert on them directly. (Asserting through
    `provider.signal_provider_event` itself does not type-check: it is a real method
    on the class, and mypy does not carry the fixture's mock-attribute swap across
    into the separate test functions that receive `provider` as a parameter.)
    """
    mass = MagicMock()
    manifest = MagicMock()
    manifest.type = ProviderType.PLUGIN
    manifest.domain = "listen_later"
    config = MagicMock()
    config.name = "Listen Later"
    config.instance_id = "listen_later"
    config.get_value = MagicMock(return_value="GLOBAL")
    # Seeded shelf: one album already saved.
    mass.music.albums.library_items = AsyncMock(
        return_value=[_album("library://album/1", listen_later=True)]
    )
    unregister = MagicMock()
    mass.subscribe = MagicMock(return_value=unregister)
    provider = ListenLaterProvider(mass, manifest, config, SUPPORTED_FEATURES)
    await provider.handle_async_init()
    await provider.loaded_in_mass()
    signal = MagicMock()
    provider.signal_provider_event = signal  # type: ignore[method-assign, misc]
    return provider, signal, unregister, mass


async def test_saving_a_new_album_signals_a_refresh(
    plugin: tuple[ListenLaterProvider, MagicMock, MagicMock, MagicMock],
) -> None:
    """An album newly flipped to saved refreshes the Discover row."""
    provider, signal, _unregister, _mass = plugin
    await provider._on_media_item_updated(_event(_album("library://album/2", listen_later=True)))
    signal.assert_called_once_with({"event": EVENT_RECOMMENDATIONS_UPDATED})


async def test_unsaving_a_known_album_signals_a_refresh(
    plugin: tuple[ListenLaterProvider, MagicMock, MagicMock, MagicMock],
) -> None:
    """An album flipped from saved to unsaved refreshes the Discover row."""
    provider, signal, _unregister, _mass = plugin
    await provider._on_media_item_updated(_event(_album("library://album/1", listen_later=False)))
    signal.assert_called_once_with({"event": EVENT_RECOMMENDATIONS_UPDATED})


async def test_unchanged_saved_album_signals_nothing(
    plugin: tuple[ListenLaterProvider, MagicMock, MagicMock, MagicMock],
) -> None:
    """A metadata refresh of an already-saved album is not a shelf change."""
    provider, signal, _unregister, _mass = plugin
    await provider._on_media_item_updated(_event(_album("library://album/1", listen_later=True)))
    signal.assert_not_called()


async def test_unrelated_unsaved_album_signals_nothing(
    plugin: tuple[ListenLaterProvider, MagicMock, MagicMock, MagicMock],
) -> None:
    """
    An update to an album that was never saved is not a shelf change.

    A library sync itself stays silent (run_sync suppresses MEDIA_ITEM_UPDATED), but
    the first ordinary metadata update of an unsaved album afterwards must still be a
    no-op here, or it would misread as a spurious join.
    """
    provider, signal, _unregister, _mass = plugin
    await provider._on_media_item_updated(_event(_album("library://album/99", listen_later=False)))
    signal.assert_not_called()


async def test_non_album_items_are_ignored(
    plugin: tuple[ListenLaterProvider, MagicMock, MagicMock, MagicMock],
) -> None:
    """
    Only albums carry the shelf flag; other media types are skipped.

    listen_later=True on a URI outside the seeded set would, without the isinstance
    guard, look like a genuine 0->1 transition and fire a signal -- so this only stays
    silent because the type check skips it, not because it happens to be a no-op.
    """
    provider, signal, _unregister, _mass = plugin
    track = Track(
        item_id="5",
        provider="library",
        name="So What",
        provider_mappings=set(),
        uri="library://track/5",
        listen_later=True,
    )
    await provider._on_media_item_updated(_event(track))
    signal.assert_not_called()


async def test_unload_releases_the_subscription(
    plugin: tuple[ListenLaterProvider, MagicMock, MagicMock, MagicMock],
) -> None:
    """unload() releases every handle it registered."""
    provider, _signal, unregister, _mass = plugin
    await provider.unload()
    unregister.assert_called_once()
    assert provider._unregister_handles == []


async def test_subscribes_to_media_item_updated(
    plugin: tuple[ListenLaterProvider, MagicMock, MagicMock, MagicMock],
) -> None:
    """
    loaded_in_mass() subscribes on MEDIA_ITEM_UPDATED specifically.

    Subscribing to the wrong EventType would make the whole feature a silent
    no-op while every other test -- which drives _on_media_item_updated directly --
    stays green.
    """
    provider, _signal, _unregister, mass = plugin
    mass.subscribe.assert_called_once_with(
        provider._on_media_item_updated, EventType.MEDIA_ITEM_UPDATED
    )


async def test_seed_query_requests_the_full_shelf(
    plugin: tuple[ListenLaterProvider, MagicMock, MagicMock, MagicMock],
) -> None:
    """The seed snapshot in loaded_in_mass() must not truncate a large shelf."""
    _provider, _signal, _unregister, mass = plugin
    mass.music.albums.library_items.assert_awaited_once_with(
        listen_later=True, order_by="listen_later_added_at_desc", limit=0
    )


async def test_unloading_before_subscribe_skips_the_subscription() -> None:
    """
    An unload completing while the seed query is in flight must win the race.

    loaded_in_mass() awaits the seed query before subscribing; if unload() has
    already run by the time that await returns, subscribing here would pin a live
    handler onto a provider that is already gone -- mass.subscribe holds a strong
    reference to the bound method, keeping the dead instance alive and emitting.
    """
    mass = MagicMock()
    manifest = MagicMock()
    manifest.type = ProviderType.PLUGIN
    manifest.domain = "listen_later"
    config = MagicMock()
    config.name = "Listen Later"
    config.instance_id = "listen_later"
    config.get_value = MagicMock(return_value="GLOBAL")
    mass.music.albums.library_items = AsyncMock(
        return_value=[_album("library://album/1", listen_later=True)]
    )
    mass.subscribe = MagicMock(return_value=MagicMock())
    provider = ListenLaterProvider(mass, manifest, config, SUPPORTED_FEATURES)
    await provider.handle_async_init()
    provider.unloading = True

    await provider.loaded_in_mass()

    mass.subscribe.assert_not_called()
    assert provider._unregister_handles == []
