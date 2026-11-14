"""Test YouTube Music artist tracks."""

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.enums import ProviderFeature

from music_assistant.providers.ytmusic import SUPPORTED_FEATURES, YoutubeMusicProvider


@pytest.fixture
def provider() -> YoutubeMusicProvider:
    """Return a YoutubeMusicProvider instance with mocked dependencies."""
    mass = AsyncMock()
    manifest = MagicMock()
    manifest.domain = "ytmusic"
    config = MagicMock()
    config.get_value.return_value = "GLOBAL"
    prov = YoutubeMusicProvider(mass, manifest, config)
    prov._headers = {}
    prov._yt_user = None
    prov.language = "en"
    return prov


def _track(item_id: str, name: str) -> MagicMock:
    track = MagicMock()
    track.item_id = item_id
    track.name = name
    track.version = ""
    return track


def _undecorated(name: str) -> Any:
    # call the undecorated function so the @use_cache wrapper stays out of the test
    return cast("Any", getattr(YoutubeMusicProvider, name)).__wrapped__


def test_artist_tracks_is_supported() -> None:
    """The artist tracks listing is answered by the provider, not by an album walk."""
    assert ProviderFeature.ARTIST_TRACKS in SUPPORTED_FEATURES


async def test_artist_tracks_are_the_songs_list(provider: YoutubeMusicProvider) -> None:
    """An artist's tracks are its YTM songs list, fetched in one playlist call."""
    songs = [_track(f"v{i}", f"Song {i}") for i in range(40)]
    artist_obj = {"songs": {"browseId": "VLOLAK5uy_songs"}}
    with (
        patch.object(provider, "_get_artist_obj", AsyncMock(return_value=artist_obj)),
        patch.object(provider, "get_playlist_tracks", AsyncMock(return_value=songs)) as playlist,
        patch.object(provider, "get_artist_albums", AsyncMock()) as albums,
    ):
        tracks = await _undecorated("get_artist_tracks")(provider, "UCartist")

    assert tracks == songs
    playlist.assert_awaited_once_with("VLOLAK5uy_songs")
    albums.assert_not_awaited()


async def test_artist_tracks_without_songs_list_are_empty(
    provider: YoutubeMusicProvider,
) -> None:
    """An artist with no songs list has no tracks; its albums are never walked."""
    with (
        patch.object(provider, "_get_artist_obj", AsyncMock(return_value={})),
        patch.object(provider, "get_artist_albums", AsyncMock()) as albums,
        patch.object(provider, "get_album_tracks", AsyncMock()) as album_tracks,
    ):
        tracks = await _undecorated("get_artist_tracks")(provider, "UCartist")

    assert tracks == []
    albums.assert_not_awaited()
    album_tracks.assert_not_awaited()


async def test_artist_top_tracks_are_the_first_25_songs(provider: YoutubeMusicProvider) -> None:
    """The top tracks are the first 25 of the songs list, which YTM ranks by popularity."""
    songs = [_track(f"v{i}", f"Song {i}") for i in range(40)]
    with patch.object(provider, "get_artist_tracks", AsyncMock(return_value=songs)):
        tracks = await _undecorated("get_artist_toptracks")(provider, "UCartist")

    assert tracks == songs[:25]
