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
        patch("music_assistant.providers.ytmusic.get_artist", AsyncMock(return_value=artist_obj)),
        patch.object(provider, "get_playlist_tracks", AsyncMock(return_value=songs)) as playlist,
        patch.object(provider, "get_artist_albums", AsyncMock()) as albums,
    ):
        tracks = await _undecorated("get_artist_tracks")(provider, "UCartist")

    assert tracks == songs
    playlist.assert_awaited_once_with("VLOLAK5uy_songs")
    albums.assert_not_awaited()


async def test_artist_tracks_without_songs_list_walk_the_albums(
    provider: YoutubeMusicProvider,
) -> None:
    """An artist with no songs list plays its albums' tracks, each song once."""
    album1, album2 = MagicMock(item_id="a1"), MagicMock(item_id="a2")
    album_tracks = {
        "a1": [_track("v1", "One"), _track("v2", "Two")],
        "a2": [_track("v3", "Two"), _track("v4", "Three")],
    }
    with (
        patch("music_assistant.providers.ytmusic.get_artist", AsyncMock(return_value={})),
        patch.object(provider, "get_artist_albums", AsyncMock(return_value=[album1, album2])),
        patch.object(
            provider,
            "get_album_tracks",
            AsyncMock(side_effect=lambda album_id: album_tracks[album_id]),
        ),
    ):
        tracks = await _undecorated("get_artist_tracks")(provider, "UCartist")

    assert [track.item_id for track in tracks] == ["v1", "v2", "v4"]


async def test_artist_top_tracks_are_the_first_25_songs(provider: YoutubeMusicProvider) -> None:
    """The top tracks are the first 25 of the songs list, which YTM ranks by popularity."""
    songs = [_track(f"v{i}", f"Song {i}") for i in range(40)]
    with patch.object(provider, "get_artist_tracks", AsyncMock(return_value=songs)):
        tracks = await _undecorated("get_artist_toptracks")(provider, "UCartist")

    assert tracks == songs[:25]


async def test_artist_tracks_album_walk_keeps_going_past_a_failing_album(
    provider: YoutubeMusicProvider,
) -> None:
    """A failing album drops only its own tracks; when every album fails, the error is raised."""
    album1, album2 = MagicMock(item_id="a1"), MagicMock(item_id="a2")

    async def album_tracks(album_id: str) -> list[MagicMock]:
        if album_id == "a1":
            raise TimeoutError("a1")
        return [_track("v3", "Three")]

    with (
        patch("music_assistant.providers.ytmusic.get_artist", AsyncMock(return_value={})),
        patch.object(provider, "get_artist_albums", AsyncMock(return_value=[album1, album2])),
        patch.object(provider, "get_album_tracks", AsyncMock(side_effect=album_tracks)),
    ):
        tracks = await _undecorated("get_artist_tracks")(provider, "UCartist")
        assert [track.item_id for track in tracks] == ["v3"]

        provider.get_artist_albums = AsyncMock(return_value=[album1])  # type: ignore[method-assign]
        with pytest.raises(TimeoutError):
            await _undecorated("get_artist_tracks")(provider, "UCartist")
