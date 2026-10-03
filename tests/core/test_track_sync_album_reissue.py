"""
Integration tests for album reissues inside the track library sync.

A file-backed provider (subsonic) can hand an album a new id while its tracks keep
theirs, for instance after a retag changes the album title. The album sync then adds
a new library row, and the track sync has to move the tracks over to it: otherwise
they stay on the stale row and the new row shows no library tracks at all.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest
from music_assistant_models.config_entries import ProviderConfig
from music_assistant_models.enums import MediaType, ProviderFeature, ProviderType
from music_assistant_models.errors import MediaNotFoundError
from music_assistant_models.media_items import Album, Artist, ProviderMapping, Track, UniqueList
from music_assistant_models.provider import ProviderManifest

from music_assistant.constants import (
    CONF_ENTRY_LIBRARY_SYNC_DELETIONS,
    DB_TABLE_ALBUM_LISTEN_LATER,
    DB_TABLE_ALBUMS,
)
from music_assistant.models.music_provider import MusicProvider

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant

_DOMAIN = "fake_files"
_INSTANCE = "fake_files--instance"


class _FakeFileProvider(MusicProvider):
    """File-backed provider that serves a mutable canned library."""

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
    ) -> None:
        super().__init__(mass, manifest, config)
        self.library_albums: list[Album] = []
        self.library_tracks: list[Track] = []

    @property
    def supported_features(self) -> set[ProviderFeature]:
        return {ProviderFeature.LIBRARY_ALBUMS, ProviderFeature.LIBRARY_TRACKS}

    @property
    def is_streaming_provider(self) -> bool:
        return False

    def library_supported(self, media_type: MediaType) -> bool:
        return media_type in (MediaType.ALBUM, MediaType.TRACK)

    async def get_library_albums(self) -> AsyncGenerator[Album]:
        for album in self.library_albums:
            yield album

    async def get_library_tracks(self) -> AsyncGenerator[Track]:
        for track in self.library_tracks:
            yield track


def _mapping(item_id: str) -> ProviderMapping:
    return ProviderMapping(
        item_id=item_id, provider_domain=_DOMAIN, provider_instance=_INSTANCE, available=True
    )


_ARTIST = Artist(
    item_id="artist-1",
    provider=_INSTANCE,
    name="Zornheym",
    provider_mappings={_mapping("artist-1")},
)


def _album(item_id: str, name: str, version: str = "") -> Album:
    return Album(
        item_id=item_id,
        provider=_INSTANCE,
        name=name,
        version=version,
        artists=UniqueList([_ARTIST]),
        provider_mappings={_mapping(item_id)},
    )


def _track(item_id: str, number: int, album: Album) -> Track:
    return Track(
        item_id=item_id,
        provider=_INSTANCE,
        name=f"Song {number}",
        duration=200,
        artists=UniqueList([_ARTIST]),
        album=album,
        disc_number=1,
        track_number=number,
        provider_mappings={_mapping(item_id)},
    )


@pytest.fixture
async def provider(mass: MusicAssistant) -> AsyncGenerator[_FakeFileProvider]:
    """Register a fake file-backed provider with library sync deletions enabled."""
    manifest = ProviderManifest(
        type=ProviderType.MUSIC,
        domain=_DOMAIN,
        name="Fake files",
        description="Fake file-backed provider for sync tests",
        codeowners=["@music-assistant"],
    )
    config = ProviderConfig(
        values={}, type=ProviderType.MUSIC, domain=_DOMAIN, instance_id=_INSTANCE, name="Fake"
    )

    def _get_value(key: str, *_args: Any, **_kwargs: Any) -> Any:
        return key == CONF_ENTRY_LIBRARY_SYNC_DELETIONS.key

    config.get_value = _get_value  # type: ignore[method-assign]
    prov = _FakeFileProvider(mass, manifest, config)
    prov.available = True
    mass._providers[prov.instance_id] = prov
    # tracks are only synced when available, which needs the provider in the cache
    await mass._update_available_providers_cache()
    try:
        yield prov
    finally:
        mass._providers.pop(prov.instance_id, None)


async def _sync(provider: _FakeFileProvider) -> None:
    await provider.sync_library(MediaType.ALBUM)
    await provider.sync_library(MediaType.TRACK)


async def _album_db_id(mass: MusicAssistant, prov_album_id: str) -> int:
    album = await mass.music.albums.get_library_item_by_prov_id(prov_album_id, _INSTANCE)
    assert album is not None
    return int(album.item_id)


async def _album_track_ids(mass: MusicAssistant, db_id: int) -> set[str]:
    tracks = await mass.music.albums.get_library_album_tracks(db_id)
    return {mapping.item_id for track in tracks for mapping in track.provider_mappings}


async def test_reissued_album_takes_over_tracks_and_user_state(
    mass: MusicAssistant, provider: _FakeFileProvider
) -> None:
    """A reissued album id moves every track to the new row and folds the stale row into it."""
    old = _album("alb-old", "Descending Into Madness (blue marbled vinyl)", "blue marbled vinyl")
    provider.library_albums = [old]
    provider.library_tracks = [_track("t1", 1, old), _track("t2", 2, old)]
    await _sync(provider)
    old_id = await _album_db_id(mass, "alb-old")
    # play history and a listen-later entry are user state that has to survive the reissue
    await mass.music.database.update(
        DB_TABLE_ALBUMS, {"item_id": old_id}, {"play_count": 3, "last_played": 30}
    )
    await mass.music.database.insert(
        DB_TABLE_ALBUM_LISTEN_LATER, {"item_id": old_id, "userid": "lera", "added_at": 5}
    )

    new = _album("alb-new", "Descending Into Madness")
    provider.library_albums = [new]
    provider.library_tracks = [_track("t1", 1, new), _track("t2", 2, new)]
    await _sync(provider)

    new_id = await _album_db_id(mass, "alb-new")
    assert new_id != old_id
    assert await _album_track_ids(mass, new_id) == {"t1", "t2"}
    with pytest.raises(MediaNotFoundError):
        await mass.music.albums.get_library_item(old_id)
    merged = await mass.music.albums.get_library_item(new_id)
    # the stale row's own data describes the old release and must not leak into the live one
    assert merged.name == "Descending Into Madness"
    assert merged.version == ""
    row = await mass.music.database.get_row(DB_TABLE_ALBUMS, {"item_id": new_id})
    assert row is not None
    assert row["play_count"] == 3
    assert await mass.music.database.get_row(
        DB_TABLE_ALBUM_LISTEN_LATER, {"item_id": new_id, "userid": "lera"}
    )


async def test_partly_moved_album_keeps_its_row(
    mass: MusicAssistant, provider: _FakeFileProvider
) -> None:
    """A track moving to another album is relinked, but its old album keeps its own row."""
    first = _album("alb-1", "First")
    second = _album("alb-2", "Second")
    provider.library_albums = [first, second]
    provider.library_tracks = [_track("t1", 1, first), _track("t2", 2, first)]
    await _sync(provider)
    first_id = await _album_db_id(mass, "alb-1")

    provider.library_tracks = [_track("t1", 1, first), _track("t2", 1, second)]
    await _sync(provider)

    second_id = await _album_db_id(mass, "alb-2")
    assert await _album_track_ids(mass, second_id) == {"t2"}
    assert "t1" in await _album_track_ids(mass, first_id)
    assert await mass.music.albums.get_library_item(first_id)


async def test_unchanged_library_does_not_update_tracks(
    mass: MusicAssistant, provider: _FakeFileProvider
) -> None:
    """A second sync of an unchanged library leaves the tracks alone."""
    album = _album("alb-1", "First")
    provider.library_albums = [album]
    provider.library_tracks = [_track("t1", 1, album)]
    await _sync(provider)

    with patch.object(
        mass.music.tracks,
        "update_item_in_library",
        AsyncMock(wraps=mass.music.tracks.update_item_in_library),
    ) as update:
        await _sync(provider)
    update.assert_not_called()
