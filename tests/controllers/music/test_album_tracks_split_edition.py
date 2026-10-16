"""Tests for a provider album that splits the library's audio into more tracks (MUSIC-109)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from music_assistant_models.enums import AlbumType
from music_assistant_models.media_items import (
    Album,
    Artist,
    ProviderMapping,
    Track,
    UniqueList,
)

from music_assistant.controllers.music.media.albums import AlbumsController
from music_assistant.mass import MusicAssistant

pytestmark = pytest.mark.asyncio

# Insomnium's Winter's Gate: one 40:00 track on CD, seven parts on the digital release
PART_SECONDS = [372, 398, 352, 326, 261, 313, 378]


def _mapping(provider_instance: str, item_id: str) -> ProviderMapping:
    return ProviderMapping(
        item_id=item_id,
        provider_domain=provider_instance.removesuffix("_inst"),
        provider_instance=provider_instance,
        in_library=True,
    )


async def _seed_album(mass: MusicAssistant, tracks: list[tuple[str, int]]) -> Album:
    """Seed a library album of the given (title, seconds) tracks, also mapped to prov_b."""
    artist = Artist(
        item_id="0",
        provider="library",
        name="Test Artist",
        provider_mappings={_mapping("prov_a_inst", "artist_a")},
    )
    db_artist = await mass.music.artists.add_item_to_library(artist)
    album = Album(
        item_id="0",
        provider="library",
        name="Test Album",
        album_type=AlbumType.ALBUM,
        provider_mappings={
            _mapping("prov_a_inst", "album_a"),
            _mapping("prov_b_inst", "album_b"),
        },
        artists=UniqueList([db_artist]),
    )
    db_album = await mass.music.albums.add_item_to_library(album)
    for idx, (name, seconds) in enumerate(tracks, start=1):
        await mass.music.tracks.add_item_to_library(
            Track(
                item_id="0",
                provider="library",
                name=name,
                duration=seconds,
                provider_mappings={_mapping("prov_a_inst", f"track_a{idx}")},
                artists=UniqueList([db_artist]),
                album=db_album,
                disc_number=1,
                track_number=idx,
            )
        )
    return db_album


def _provider_track(name: str, track_number: int, seconds: int) -> Track:
    return Track(
        item_id=f"prov_b_{track_number}",
        provider="prov_b_inst",
        name=name,
        duration=seconds,
        provider_mappings={_mapping("prov_b_inst", f"prov_b_{track_number}")},
        disc_number=1,
        track_number=track_number,
    )


def _patch_provider_tracks(listing: list[Track]) -> Any:
    async def _fake(
        _self: AlbumsController, _item_id: str, provider_instance_id_or_domain: str
    ) -> list[Track]:
        return list(listing) if provider_instance_id_or_domain == "prov_b_inst" else []

    return patch.object(AlbumsController, "_get_provider_album_tracks", _fake)


def _parts() -> list[Track]:
    return [
        _provider_track(f"Winter's Gate, Pt. {idx}", idx, seconds)
        for idx, seconds in enumerate(PART_SECONDS, start=1)
    ]


async def test_a_finer_split_of_held_audio_is_not_listed(mass: MusicAssistant) -> None:
    """The parts of a song the library holds whole are not tracks the library lacks."""
    db_album = await _seed_album(mass, [("Winter’s Gate", 2400)])  # noqa: RUF001
    with _patch_provider_tracks(_parts()):
        tracks = await mass.music.albums.tracks(db_album.item_id, "library")
    assert [track.name for track in tracks] == ["Winter’s Gate"]  # noqa: RUF001


async def test_a_bonus_track_past_the_held_audio_is_still_listed(mass: MusicAssistant) -> None:
    """A provider listing longer than what the library holds still offers what it adds."""
    held = [("One", 300), ("Two", 300), ("Three", 300)]
    db_album = await _seed_album(mass, held)
    listing = [
        _provider_track("One", 1, 300),
        _provider_track("Two", 2, 300),
        _provider_track("Three", 3, 300),
        _provider_track("Four (Bonus Track)", 4, 300),
    ]
    with _patch_provider_tracks(listing):
        tracks = await mass.music.albums.tracks(db_album.item_id, "library")
    assert [track.name for track in tracks] == ["One", "Two", "Three", "Four (Bonus Track)"]


async def test_a_short_bonus_track_is_still_listed(mass: MusicAssistant) -> None:
    """A bonus track short enough to hide in the album's running time is still offered."""
    held = [(f"Song {idx}", 360) for idx in range(1, 11)]
    db_album = await _seed_album(mass, held)
    listing = [_provider_track(name, idx, seconds) for idx, (name, seconds) in enumerate(held, 1)]
    listing.append(_provider_track("Hidden Outro", 11, 120))
    with _patch_provider_tracks(listing):
        tracks = await mass.music.albums.tracks(db_album.item_id, "library")
    assert [track.name for track in tracks] == [name for name, _ in held] + ["Hidden Outro"]


async def test_unknown_durations_change_nothing(mass: MusicAssistant) -> None:
    """Without durations to compare the listing is merged as before."""
    db_album = await _seed_album(mass, [("Winter’s Gate", 0)])  # noqa: RUF001
    with _patch_provider_tracks(_parts()):
        tracks = await mass.music.albums.tracks(db_album.item_id, "library")
    assert len(tracks) == 7
