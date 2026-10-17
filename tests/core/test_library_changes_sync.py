"""
Integration tests for syncing only what a provider reports as changed (MUSIC-119).

A full library sync only adds and removes items, so an edit that keeps an item's id
(a retag, a re-encode, new artwork) never reaches the library. A changes sync replaces
what the provider stored for each changed item instead, and only touches those.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest
from music_assistant_models.config_entries import ProviderConfig
from music_assistant_models.enums import ImageType, MediaType, ProviderFeature, ProviderType
from music_assistant_models.errors import MediaNotFoundError
from music_assistant_models.media_items import (
    Album,
    Artist,
    MediaItemImage,
    MediaItemMetadata,
    ProviderMapping,
    Track,
    UniqueList,
)
from music_assistant_models.provider import ProviderManifest

from music_assistant.constants import CONF_ENTRY_LIBRARY_SYNC_DELETIONS
from music_assistant.controllers.music.helpers import provider_mappings_for_update
from music_assistant.models.music_provider import (
    LibraryChanges,
    MusicProvider,
    _authoritative_metadata,
)

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
        self.library_artists: list[Artist] = []
        self.library_albums: list[Album] = []
        self.library_tracks: list[Track] = []

    @property
    def supported_features(self) -> set[ProviderFeature]:
        return {
            ProviderFeature.LIBRARY_ARTISTS,
            ProviderFeature.LIBRARY_ALBUMS,
            ProviderFeature.LIBRARY_TRACKS,
        }

    @property
    def is_streaming_provider(self) -> bool:
        return False

    def library_supported(self, media_type: MediaType) -> bool:
        return media_type in (MediaType.ARTIST, MediaType.ALBUM, MediaType.TRACK)

    async def get_library_artists(self) -> AsyncGenerator[Artist]:
        for artist in self.library_artists:
            yield artist

    async def get_library_albums(self) -> AsyncGenerator[Album]:
        for album in self.library_albums:
            yield album

    async def get_library_tracks(self) -> AsyncGenerator[Track]:
        for track in self.library_tracks:
            yield track

    async def get_artist(self, prov_artist_id: str) -> Artist:
        return _find(self.library_artists, prov_artist_id)

    async def get_album(self, prov_album_id: str) -> Album:
        return _find(self.library_albums, prov_album_id)

    async def get_track(self, prov_track_id: str) -> Track:
        return _find(self.library_tracks, prov_track_id)


def _find[T: Artist | Album | Track](items: list[T], item_id: str) -> T:
    for item in items:
        if item.item_id == item_id:
            return item
    raise MediaNotFoundError(item_id)


def _mapping(item_id: str) -> ProviderMapping:
    return ProviderMapping(
        item_id=item_id, provider_domain=_DOMAIN, provider_instance=_INSTANCE, available=True
    )


def _artist() -> Artist:
    return Artist(
        item_id="artist-1",
        provider=_INSTANCE,
        name="Zornheym",
        provider_mappings={_mapping("artist-1")},
    )


def _image(path: str, provider: str = _INSTANCE) -> MediaItemImage:
    return MediaItemImage(type=ImageType.THUMB, path=path, provider=provider)


def _album(item_id: str, name: str, cover: str = "cover-1") -> Album:
    return Album(
        item_id=item_id,
        provider=_INSTANCE,
        name=name,
        artists=UniqueList([_artist()]),
        metadata=MediaItemMetadata(images=UniqueList([_image(cover)])),
        provider_mappings={_mapping(item_id)},
    )


def _track(item_id: str, number: int, album: Album, name: str = "", duration: int = 200) -> Track:
    return Track(
        item_id=item_id,
        provider=_INSTANCE,
        name=name or f"Song {number}",
        duration=duration,
        artists=UniqueList([_artist()]),
        album=album,
        disc_number=1,
        track_number=number,
        provider_mappings={_mapping(item_id)},
    )


@pytest.fixture
async def provider(
    mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[_FakeFileProvider]:
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

    monkeypatch.setattr(config, "get_value", _get_value)
    prov = _FakeFileProvider(mass, manifest, config)
    prov.available = True
    mass._providers[prov.instance_id] = prov
    # tracks are only synced when available, which needs the provider in the cache
    await mass._update_available_providers_cache()
    try:
        yield prov
    finally:
        mass._providers.pop(prov.instance_id, None)


_ALL = [MediaType.ARTIST, MediaType.ALBUM, MediaType.TRACK]


async def _full_sync(provider: _FakeFileProvider) -> None:
    for media_type in _ALL:
        await provider.sync_library(media_type)


async def _sync_changes(
    provider: _FakeFileProvider,
    changed: dict[MediaType, set[str]] | None = None,
    removed: dict[MediaType, set[str]] | None = None,
) -> None:
    changes = LibraryChanges(cursor="c", changed=changed or {}, removed=removed or {})
    await provider.sync_library_changes(changes, _ALL)


async def _library_track(mass: MusicAssistant, prov_track_id: str) -> Track | None:
    return await mass.music.tracks.get_library_item_by_prov_id(prov_track_id, _INSTANCE)


async def _start_library(provider: _FakeFileProvider) -> Album:
    album = _album("alb-1", "First")
    provider.library_artists = [_artist()]
    provider.library_albums = [album]
    provider.library_tracks = [_track("t1", 1, album), _track("t2", 2, album)]
    await _full_sync(provider)
    return album


async def test_changed_track_replaces_stored_fields(
    mass: MusicAssistant, provider: _FakeFileProvider
) -> None:
    """A retagged, trimmed file keeps its id; its new title and duration still land."""
    album = await _start_library(provider)
    provider.library_tracks[0] = _track("t1", 1, album, name="Song 1 (edit)", duration=150)

    await _full_sync(provider)
    stale = await _library_track(mass, "t1")
    assert stale is not None
    assert (stale.name, stale.duration) == ("Song 1", 200)

    await _sync_changes(provider, changed={MediaType.TRACK: {"t1"}})

    track = await _library_track(mass, "t1")
    assert track is not None
    assert (track.name, track.duration) == ("Song 1 (edit)", 150)
    assert all(x.in_library for x in track.provider_mappings)


async def test_only_changed_items_are_updated(
    mass: MusicAssistant, provider: _FakeFileProvider
) -> None:
    """A changes sync leaves every item it was not told about alone."""
    await _start_library(provider)

    with patch.object(
        mass.music.tracks,
        "update_item_in_library",
        AsyncMock(wraps=mass.music.tracks.update_item_in_library),
    ) as update:
        await _sync_changes(provider, changed={MediaType.TRACK: {"t2"}})

    assert update.call_count == 1
    assert update.call_args.args[1].item_id == "t2"


async def test_new_artwork_replaces_the_providers_old_cover(
    mass: MusicAssistant, provider: _FakeFileProvider
) -> None:
    """A new cover file reaches the album, without dropping images of other sources."""
    await _start_library(provider)
    library_album = await mass.music.albums.get_library_item_by_prov_id("alb-1", _INSTANCE)
    assert library_album is not None
    library_album.metadata.images = UniqueList(
        [*(library_album.metadata.images or []), _image("fanart-1", provider="fanarttv")]
    )
    await mass.music.albums.update_item_in_library(library_album.item_id, library_album)
    provider.library_albums[0] = _album("alb-1", "First", cover="cover-2")

    await _sync_changes(provider, changed={MediaType.ALBUM: {"alb-1"}})

    album = await mass.music.albums.get_library_item(library_album.item_id)
    assert [x.path for x in album.metadata.images or []] == ["cover-2", "fanart-1"]


async def test_new_album_and_tracks_are_added(
    mass: MusicAssistant, provider: _FakeFileProvider
) -> None:
    """Items the provider gained since the last sync are added to the library."""
    await _start_library(provider)
    second = _album("alb-2", "Second")
    provider.library_albums.append(second)
    provider.library_tracks.append(_track("t3", 1, second))

    await _sync_changes(provider, changed={MediaType.ALBUM: {"alb-2"}, MediaType.TRACK: {"t3"}})

    assert await mass.music.albums.get_library_item_by_prov_id("alb-2", _INSTANCE)
    track = await _library_track(mass, "t3")
    assert track is not None
    assert track.album is not None
    assert track.album.name == "Second"


async def test_removed_track_leaves_the_library(
    mass: MusicAssistant, provider: _FakeFileProvider
) -> None:
    """A deleted file without any user stake in it is removed from the library."""
    await _start_library(provider)
    provider.library_tracks.pop()

    await _sync_changes(provider, removed={MediaType.TRACK: {"t2"}})

    assert await _library_track(mass, "t2") is None
    assert await _library_track(mass, "t1") is not None


async def test_replaced_file_only_drops_the_old_mapping(
    mass: MusicAssistant, provider: _FakeFileProvider
) -> None:
    """A file replaced under a new id keeps its library track, only the dead id goes."""
    album = await _start_library(provider)
    track = await _library_track(mass, "t2")
    assert track is not None
    await mass.music.tracks.add_provider_mappings(track.item_id, [_mapping("t2-new")])
    provider.library_tracks[1] = _track("t2-new", 2, album)

    await _sync_changes(
        provider, changed={MediaType.TRACK: {"t2-new"}}, removed={MediaType.TRACK: {"t2"}}
    )

    replaced = await mass.music.tracks.get_library_item(track.item_id)
    assert {x.item_id for x in replaced.provider_mappings} == {"t2-new"}


async def test_removals_wait_for_deletions_to_be_enabled(
    mass: MusicAssistant, provider: _FakeFileProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With library sync deletions off, a changes sync keeps removed items like a full one."""
    await _start_library(provider)
    monkeypatch.setattr(provider, "library_sync_deletions_enabled", lambda: False)

    await _sync_changes(provider, removed={MediaType.TRACK: {"t2"}})

    assert await _library_track(mass, "t2") is not None


def test_authoritative_metadata_keeps_other_sources() -> None:
    """The provider's own values win; other sources' images and empty fields are kept."""
    stored = MediaItemMetadata(
        description="bio",
        genres={"Metal"},
        lyrics="old words",
        images=UniqueList([_image("cover-1"), _image("fanart-1", provider="fanarttv")]),
    )
    update = MediaItemMetadata(
        description="other bio", lyrics="new words", images=UniqueList([_image("cover-2")])
    )

    merged = _authoritative_metadata(stored, update, {_INSTANCE, _DOMAIN})

    assert merged.lyrics == "new words"
    assert merged.genres == {"Metal"}
    assert merged.description == "bio"
    assert [x.path for x in merged.images or []] == ["cover-2", "fanart-1"]
    assert stored.lyrics == "old words"


def test_overwrite_keeps_library_state_of_sibling_copies() -> None:
    """A sibling instance's copy without a library state inherits the stored one."""
    sibling = "fake_files--sibling"
    stored = [
        ProviderMapping(
            item_id="t1", provider_domain=_DOMAIN, provider_instance=sibling, in_library=True
        ),
        ProviderMapping(
            item_id="x1", provider_domain="tidal", provider_instance="tidal--1", in_library=True
        ),
    ]
    update = [
        ProviderMapping(item_id="t1", provider_domain=_DOMAIN, provider_instance=_INSTANCE),
        ProviderMapping(item_id="t1", provider_domain=_DOMAIN, provider_instance=sibling),
    ]

    result = {
        (x.provider_instance, x.item_id): x.in_library
        for x in provider_mappings_for_update(stored, update, overwrite=True)
    }

    assert result == {(_INSTANCE, "t1"): None, (sibling, "t1"): True, ("tidal--1", "x1"): True}


async def test_unknown_changes_fall_back_to_a_full_sync(
    mass: MusicAssistant, provider: _FakeFileProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a baseline the provider gets a full sync, and its cursor is kept for next time."""
    cursors: list[str | None] = []

    async def _changes(cursor: str | None) -> LibraryChanges:
        cursors.append(cursor)
        return LibraryChanges(cursor=f"c{len(cursors)}", complete=len(cursors) > 1)

    monkeypatch.setattr(provider, "get_library_changes", _changes)
    full_sync = AsyncMock()
    changes_sync = AsyncMock()
    monkeypatch.setattr(provider, "sync_library", full_sync)
    monkeypatch.setattr(provider, "sync_library_changes", changes_sync)

    async def run() -> None:
        await mass.music._sync_provider_changes(provider, _ALL)

    await run()
    assert [x.args[0] for x in full_sync.call_args_list] == _ALL
    changes_sync.assert_not_called()

    full_sync.reset_mock()
    await run()
    full_sync.assert_not_called()
    changes_sync.assert_called_once()
    assert cursors == [None, "c1"]

    await run()
    assert cursors == [None, "c1", "c2"]


async def test_provider_without_changes_keeps_full_syncs(
    mass: MusicAssistant, provider: _FakeFileProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider that can't report changes is fully synced every time."""
    full_sync = AsyncMock()
    monkeypatch.setattr(provider, "sync_library", full_sync)

    await mass.music._sync_provider_changes(provider, [MediaType.TRACK])
    await mass.music._sync_provider_changes(provider, [MediaType.TRACK])

    assert full_sync.call_count == 2


async def test_request_during_a_run_runs_it_again(
    mass: MusicAssistant, provider: _FakeFileProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changes made while a changes sync runs are picked up by one more run, not dropped."""
    runs = 0

    async def _sync(*_args: Any) -> None:
        nonlocal runs
        runs += 1
        if runs == 1:
            # a trigger arriving mid-run finds the task still active
            mass.music._changes_sync_requested.add(provider.instance_id)

    monkeypatch.setattr(mass.music, "_sync_provider_changes", _sync)
    mass.music._changes_sync_requested.add(provider.instance_id)

    await mass.music._create_provider_changes_sync_handler(provider, _ALL)()

    assert runs == 2
    assert provider.instance_id not in mass.music._changes_sync_requested
