"""Tests for the digarr context-menu actions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.auth import Scope
from music_assistant_models.errors import InsufficientPermissions, InvalidDataError
from music_assistant_models.media_items import Artist

from music_assistant.providers.digarr.client import DigarrError

from .conftest import as_user


async def test_approve_patches_the_mapped_recommendation(provider) -> None:
    """Approve resolves the uri to a digarr id and sets the status."""
    artist = Artist(item_id="ytm1", provider="ytmusic", name="Opeth", provider_mappings=set())
    provider.mass.music.get_item_by_uri = AsyncMock(return_value=artist)
    provider._rec_ids = {artist.uri: 42}
    provider._client.set_status = AsyncMock(return_value={"status": "approved"})
    provider._refresh = AsyncMock()

    result = await provider.approve(artist.uri)

    provider._client.set_status.assert_awaited_once_with(42, "approved")
    assert result["status"] == "approved"
    assert result["artist"] == "Opeth"


async def test_approve_refreshes_so_the_card_leaves_the_row(provider) -> None:
    """After acting, the row must drop the actioned artist."""
    artist = Artist(item_id="ytm1", provider="ytmusic", name="Opeth", provider_mappings=set())
    provider.mass.music.get_item_by_uri = AsyncMock(return_value=artist)
    provider._rec_ids = {artist.uri: 42}
    provider._client.set_status = AsyncMock(return_value={"status": "approved"})
    provider._refresh = AsyncMock()

    await provider.approve(artist.uri)

    provider._refresh.assert_awaited_once()


async def test_reject_sets_the_rejected_status(provider) -> None:
    """Reject is the same path with a different status."""
    artist = Artist(item_id="ytm1", provider="ytmusic", name="Opeth", provider_mappings=set())
    provider.mass.music.get_item_by_uri = AsyncMock(return_value=artist)
    provider._rec_ids = {artist.uri: 42}
    provider._client.set_status = AsyncMock(return_value={"status": "rejected"})
    provider._refresh = AsyncMock()

    await provider.reject(artist.uri)

    provider._client.set_status.assert_awaited_once_with(42, "rejected")


async def test_block_rejects_and_blocks_the_artist(provider) -> None:
    """Block rejects the recommendation, then permanently blocks its digarr artist id."""
    artist = Artist(item_id="ytm1", provider="ytmusic", name="Opeth", provider_mappings=set())
    provider.mass.music.get_item_by_uri = AsyncMock(return_value=artist)
    provider._rec_ids = {artist.uri: 42}
    provider._artist_ids = {artist.uri: 101}
    provider._client.set_status = AsyncMock(return_value={"status": "rejected"})
    provider._client.block_artist = AsyncMock()
    provider._refresh = AsyncMock()

    result = await provider.block(artist.uri)

    provider._client.set_status.assert_awaited_once_with(42, "rejected")
    provider._client.block_artist.assert_awaited_once_with(101)
    provider._refresh.assert_awaited_once()
    assert result["status"] == "rejected"


async def test_block_refuses_an_artist_with_no_tracked_artist_id(provider) -> None:
    """Block must not call the block endpoint with a guessed or missing artist id."""
    artist = Artist(item_id="ytm1", provider="ytmusic", name="Opeth", provider_mappings=set())
    provider.mass.music.get_item_by_uri = AsyncMock(return_value=artist)
    provider._rec_ids = {artist.uri: 42}
    provider._artist_ids = {}
    provider._client.set_status = AsyncMock()
    provider._client.block_artist = AsyncMock()

    with pytest.raises(InvalidDataError):
        await provider.block(artist.uri)
    provider._client.set_status.assert_not_awaited()
    provider._client.block_artist.assert_not_awaited()


async def test_undo_reverts_and_reports_the_lidarr_outcome(provider) -> None:
    """Undo reverts the row and tells the caller whether Lidarr was unwound."""
    artist = Artist(item_id="ytm1", provider="ytmusic", name="Opeth", provider_mappings=set())
    provider.mass.music.get_item_by_uri = AsyncMock(return_value=artist)
    provider._rec_ids = {artist.uri: 42}
    provider._client.set_status = AsyncMock(
        return_value={
            "status": "pending",
            "lidarrArtistRemoved": False,
            "lidarrRemovalSkippedReason": "has_files",
        }
    )
    provider._refresh = AsyncMock()

    result = await provider.undo(artist.uri)

    provider._client.set_status.assert_awaited_once_with(42, "pending", remove_lidarr_artist=True)
    assert result["lidarr_artist_removed"] is False
    assert result["detail"] == "has_files"


async def test_an_unmapped_item_is_refused_rather_than_guessed(provider) -> None:
    """Never approve the wrong artist because a uri rotated."""
    artist = Artist(
        item_id="ytm-unknown", provider="ytmusic", name="Someone Else", provider_mappings=set()
    )
    provider.mass.music.get_item_by_uri = AsyncMock(return_value=artist)
    provider._rec_ids = {}
    provider._client.set_status = AsyncMock()

    with pytest.raises(InvalidDataError):
        await provider.approve(artist.uri)
    provider._client.set_status.assert_not_awaited()


async def test_a_digarr_failure_propagates_to_the_caller(provider) -> None:
    """The toast must show the real reason; do not swallow it."""
    artist = Artist(item_id="ytm1", provider="ytmusic", name="Opeth", provider_mappings=set())
    provider.mass.music.get_item_by_uri = AsyncMock(return_value=artist)
    provider._rec_ids = {artist.uri: 42}
    provider._client.set_status = AsyncMock(side_effect=DigarrError("digarr is down"))
    provider._refresh = AsyncMock()

    with pytest.raises(DigarrError):
        await provider.approve(artist.uri)


async def test_a_non_bound_user_is_refused(provider) -> None:
    """
    One Music Assistant user must not be able to act on another's digarr account.

    These handlers act with this instance's own API key, i.e. as its bound
    digarr user. Without this check, any viewer could approve/reject/block
    against someone else's digarr account and trigger a real Lidarr download in
    their name -- the write-path equivalent of the get_recommendation_items gap
    closed for the read path.
    """
    artist = Artist(item_id="ytm1", provider="ytmusic", name="Opeth", provider_mappings=set())
    provider.mass.music.get_item_by_uri = AsyncMock(return_value=artist)
    provider._rec_ids = {artist.uri: 42}
    provider._client.set_status = AsyncMock(return_value={"status": "approved"})
    provider._refresh = AsyncMock()

    with as_user("lera"), pytest.raises(InsufficientPermissions):
        await provider.approve(artist.uri)
    provider._client.set_status.assert_not_awaited()


async def test_commands_are_registered_unconditionally(provider) -> None:
    """
    The menu entries exist even when digarr is unreachable at load.

    Registering conditionally would leave the provider marked available but with
    the actions silently missing, which is far harder to diagnose than an error
    on first use.
    """
    provider._client.whoami = AsyncMock(side_effect=DigarrError("unreachable"))
    provider.mass.register_api_command = MagicMock(return_value=lambda: None)

    await provider.loaded_in_mass()

    calls = provider.mass.register_api_command.call_args_list
    registered = {call.args[0] for call in calls}
    assert registered == {"digarr/approve", "digarr/reject", "digarr/block", "digarr/undo"}
    # A silently-dropped required_scope would expose these write commands to any
    # authenticated user, and nothing else here would notice.
    assert all(call.kwargs.get("required_scope") == Scope.LIBRARY_MANAGE for call in calls)
    assert len(provider._unregister_handles) == 4
