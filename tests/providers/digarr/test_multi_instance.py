"""
Tests for two loaded digarr instances sharing one Music Assistant.

digarr is ``multi_instance: true`` -- one instance per digarr user -- but the
four context-menu commands it registers live in a single, global command
registry (``mass.command_handlers``). Every other test in this package builds
exactly one instance, which is why a second instance blowing up in
``loaded_in_mass`` with "Command digarr/approve is already registered" shipped
undetected. These tests build two.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.errors import InsufficientPermissions
from music_assistant_models.media_items import Artist

from music_assistant.providers.digarr import DigarrProvider

from .conftest import as_user, load_provider, make_provider, make_shared_mass, unload_provider

COMMANDS = {"digarr/approve", "digarr/reject", "digarr/block", "digarr/undo"}


def make_pair() -> tuple[MagicMock, DigarrProvider, DigarrProvider]:
    """Build a shared mass plus a Tom instance and a Lera instance, unloaded."""
    mass = make_shared_mass()
    tom = make_provider(mass, ma_user="tom", instance_id="digarr--tom", name="Digarr - Tom")
    lera = make_provider(mass, ma_user="lera", instance_id="digarr--lera", name="Digarr - Lera")
    return mass, tom, lera


async def test_loading_two_instances_does_not_raise_and_registers_once() -> None:
    """
    The second instance to load must not blow up in loaded_in_mass.

    Against the pre-fix code, Lera's ``load_provider`` call raises RuntimeError
    ("Command digarr/approve is already registered") because both instances
    register unconditionally into the same global registry.
    """
    mass, tom, lera = make_pair()

    await load_provider(mass, tom)
    await load_provider(mass, lera)  # must not raise

    assert set(mass.command_handlers) == COMMANDS
    # Registered once each, not once per instance.
    assert mass.register_api_command.call_count == len(COMMANDS)


async def test_only_the_registering_instance_holds_the_unregister_handles() -> None:
    """The first instance to load owns the registration; the second owns nothing."""
    mass, tom, lera = make_pair()

    await load_provider(mass, tom)
    await load_provider(mass, lera)

    assert len(tom._unregister_handles) == len(COMMANDS)
    assert lera._unregister_handles == []


async def test_approve_as_tom_acts_through_toms_instance_only() -> None:
    """A command invoked as tom must call tom's client, not lera's."""
    mass, tom, lera = make_pair()
    await load_provider(mass, tom)
    await load_provider(mass, lera)

    artist = Artist(item_id="ytm1", provider="ytmusic", name="Opeth", provider_mappings=set())
    mass.music.get_item_by_uri = AsyncMock(return_value=artist)
    tom._rec_ids = {artist.uri: 42}
    lera._rec_ids = {artist.uri: 99}
    tom._client.set_status = AsyncMock(return_value={"status": "approved"})
    lera._client.set_status = AsyncMock(return_value={"status": "approved"})
    tom._refresh = AsyncMock()
    lera._refresh = AsyncMock()

    handler = mass.command_handlers["digarr/approve"]
    with as_user("tom"):
        result = await handler(artist.uri)

    tom._client.set_status.assert_awaited_once_with(42, "approved")
    lera._client.set_status.assert_not_awaited()
    assert result["status"] == "approved"


async def test_approve_as_lera_acts_through_leras_instance_only() -> None:
    """The same shared command, invoked as lera, must call lera's client, not tom's."""
    mass, tom, lera = make_pair()
    await load_provider(mass, tom)
    await load_provider(mass, lera)

    artist = Artist(item_id="ytm1", provider="ytmusic", name="Opeth", provider_mappings=set())
    mass.music.get_item_by_uri = AsyncMock(return_value=artist)
    tom._rec_ids = {artist.uri: 42}
    lera._rec_ids = {artist.uri: 99}
    tom._client.set_status = AsyncMock(return_value={"status": "approved"})
    lera._client.set_status = AsyncMock(return_value={"status": "approved"})
    tom._refresh = AsyncMock()
    lera._refresh = AsyncMock()

    handler = mass.command_handlers["digarr/approve"]
    with as_user("lera"):
        result = await handler(artist.uri)

    lera._client.set_status.assert_awaited_once_with(99, "approved")
    tom._client.set_status.assert_not_awaited()
    assert result["status"] == "approved"


async def test_a_viewer_bound_to_neither_instance_is_refused() -> None:
    """A third Music Assistant user, bound to no digarr instance, must be refused."""
    mass, tom, lera = make_pair()
    await load_provider(mass, tom)
    await load_provider(mass, lera)

    artist = Artist(item_id="ytm1", provider="ytmusic", name="Opeth", provider_mappings=set())
    mass.music.get_item_by_uri = AsyncMock(return_value=artist)
    tom._rec_ids = {artist.uri: 42}
    lera._rec_ids = {artist.uri: 99}
    tom._client.set_status = AsyncMock()
    lera._client.set_status = AsyncMock()

    handler = mass.command_handlers["digarr/approve"]
    with as_user("stranger"), pytest.raises(InsufficientPermissions):
        await handler(artist.uri)

    tom._client.set_status.assert_not_awaited()
    lera._client.set_status.assert_not_awaited()


async def test_unloading_the_registering_instance_hands_off_to_the_remaining_one() -> None:
    """
    Tom unloading must not take Lera's commands down with him.

    The commands are global: if Tom's instance (the original registrant)
    simply unregistered them on unload, Lera would lose approve/reject/block/
    undo entirely even though her own instance is still loaded.
    """
    mass, tom, lera = make_pair()
    await load_provider(mass, tom)
    await load_provider(mass, lera)
    assert len(tom._unregister_handles) == len(COMMANDS)

    await unload_provider(mass, tom)

    # Still registered, now owned by lera instead of the departed tom instance.
    assert set(mass.command_handlers) == COMMANDS
    assert len(lera._unregister_handles) == len(COMMANDS)
    assert tom._unregister_handles == []

    artist = Artist(item_id="ytm1", provider="ytmusic", name="Opeth", provider_mappings=set())
    mass.music.get_item_by_uri = AsyncMock(return_value=artist)
    lera._rec_ids = {artist.uri: 99}
    lera._client.set_status = AsyncMock(return_value={"status": "approved"})
    lera._refresh = AsyncMock()

    handler = mass.command_handlers["digarr/approve"]
    with as_user("lera"):
        result = await handler(artist.uri)

    lera._client.set_status.assert_awaited_once_with(99, "approved")
    assert result["status"] == "approved"


async def test_unloading_the_last_instance_actually_unregisters() -> None:
    """Once no digarr instance remains, the commands must genuinely disappear."""
    mass, tom, lera = make_pair()
    await load_provider(mass, tom)
    await load_provider(mass, lera)

    await unload_provider(mass, tom)
    await unload_provider(mass, lera)

    assert mass.command_handlers == {}
