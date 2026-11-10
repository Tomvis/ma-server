"""Removing an album keeps the tracks another album still holds (MUSIC-213)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from music_assistant_models.errors import MediaNotFoundError

from .helpers import create_album, create_track

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant


@pytest.fixture(name="mass")
def mass_fixture(music_mass_with_audio_analysis: MusicAssistant) -> MusicAssistant:
    """Run on a library-only instance whose removals reach the audio analysis."""
    return music_mass_with_audio_analysis


async def _track_on_two_albums(mass: MusicAssistant) -> tuple[int, int, int]:
    """Store one recording linked to two albums, as a reissue and an EP share it."""
    albums = mass.music.albums
    reissue = await albums.add_item_to_library(create_album("subsonic_1", "a1", "Reissue"))
    ep = await albums.add_item_to_library(create_album("subsonic_1", "a2", "EP"))
    track = create_track("subsonic_1", "t1", "Parasite")
    track.album = reissue
    track.disc_number, track.track_number = 1, 4
    db_track = await mass.music.tracks.add_item_to_library(track)
    await mass.music.tracks._set_track_album(
        db_id=int(db_track.item_id), album=ep, disc_number=1, track_number=4
    )
    return int(reissue.item_id), int(ep.item_id), int(db_track.item_id)


async def test_album_removal_keeps_a_track_another_album_holds(mass: MusicAssistant) -> None:
    """The track stays in the library, linked to the album that was not removed."""
    reissue_id, ep_id, track_id = await _track_on_two_albums(mass)

    await mass.music.albums.remove_item_from_library(reissue_id)

    assert await mass.music.tracks.get_library_item(track_id)
    ep_tracks = await mass.music.albums.get_library_album_tracks(ep_id)
    assert [int(t.item_id) for t in ep_tracks] == [track_id]


async def test_album_removal_still_removes_its_own_tracks(mass: MusicAssistant) -> None:
    """A track no other album holds goes with its album."""
    reissue_id, ep_id, track_id = await _track_on_two_albums(mass)

    await mass.music.albums.remove_item_from_library(ep_id)
    await mass.music.albums.remove_item_from_library(reissue_id)

    with pytest.raises(MediaNotFoundError):
        await mass.music.tracks.get_library_item(track_id)
