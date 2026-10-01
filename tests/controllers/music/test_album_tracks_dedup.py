"""Tests for de-duplicating a library album's tracks against its other providers."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from music_assistant_models.enums import AlbumType
from music_assistant_models.errors import MediaNotFoundError
from music_assistant_models.helpers import set_global_cache_values
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

LIBRARY_TITLES = [
    "Sisyphus’ Bliss",  # noqa: RUF001 - the typographic apostrophe is the point
    "…Of The Dead Who Never Rest",
    "And a Cross Now Marks His Place",
    "Blood Red Sunset",
    "Demiurg",
]


def _mapping(provider_instance: str, item_id: str) -> ProviderMapping:
    return ProviderMapping(
        item_id=item_id,
        provider_domain=provider_instance.removesuffix("_inst"),
        provider_instance=provider_instance,
        in_library=True,
    )


async def _seed_album(mass: MusicAssistant) -> Album:
    """Seed a library album of disc-1 tracks, also mapped to a second provider."""
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
    for idx, name in enumerate(LIBRARY_TITLES, start=1):
        await mass.music.tracks.add_item_to_library(
            Track(
                item_id="0",
                provider="library",
                name=name,
                provider_mappings={_mapping("prov_a_inst", f"track_a{idx}")},
                artists=UniqueList([db_artist]),
                album=db_album,
                disc_number=1,
                track_number=idx,
            )
        )
    return db_album


def _provider_track(name: str, track_number: int, disc_number: int = 0) -> Track:
    """Build a listing entry as a single-disc provider serves it (disc 0)."""
    return Track(
        item_id=f"prov_b_{track_number}",
        provider="prov_b_inst",
        name=name,
        provider_mappings={_mapping("prov_b_inst", f"prov_b_{track_number}")},
        disc_number=disc_number,
        track_number=track_number,
    )


def _patch_provider_tracks(listing: list[Track] | Exception) -> Any:
    """Serve the given listing for prov_b, nothing for the album's own provider."""

    async def _fake(
        _self: AlbumsController, _item_id: str, provider_instance_id_or_domain: str
    ) -> list[Track]:
        if provider_instance_id_or_domain != "prov_b_inst":
            return []
        if isinstance(listing, Exception):
            raise listing
        return list(listing)

    return patch.object(AlbumsController, "_get_provider_album_tracks", _fake)


async def test_disc_zero_listing_does_not_duplicate_the_album(mass: MusicAssistant) -> None:
    """
    A provider that numbers every track disc 0 must not re-list a disc-1 album.

    The titles here share nothing with the library's, so position is the only thing
    that can pair them up -- which is the case a raw "0.1" != "1.1" key gets wrong.
    """
    db_album = await _seed_album(mass)
    listing = [
        _provider_track(name, idx)
        for idx, name in enumerate(
            [
                "Prelude In Ash",
                "The Hollow Year",
                "Salt And Ember",
                "Vantablack Rite",
                "Coda For No One",
            ],
            start=1,
        )
    ]
    with _patch_provider_tracks(listing):
        tracks = await mass.music.albums.tracks(db_album.item_id, "library")
    assert [track.name for track in tracks] == LIBRARY_TITLES


async def test_title_drift_does_not_duplicate_a_held_track(mass: MusicAssistant) -> None:
    """Punctuation, credit and edition drift must not admit a second copy of a track."""
    db_album = await _seed_album(mass)
    # positions past the album's own, so only the titles can match them up
    listing = [
        _provider_track("Sisyphus' Bliss", 11),
        _provider_track("...Of The Dead Who Never Rest", 12),
        _provider_track("And a Cross Now Marks His Place (ft. Nick Holmes)", 13),
        _provider_track("Blood Red Sunset (Bonus Track)", 14),
        _provider_track("01-Test Artist-Demiurg", 15),
    ]
    with _patch_provider_tracks(listing):
        tracks = await mass.music.albums.tracks(db_album.item_id, "library")
    assert [track.name for track in tracks] == LIBRARY_TITLES


async def test_genuine_extra_track_is_still_listed(mass: MusicAssistant) -> None:
    """Content the provider has and the library does not must still be offered."""
    db_album = await _seed_album(mass)
    listing = [_provider_track("An Entirely Different Song", 11)]
    with _patch_provider_tracks(listing):
        tracks = await mass.music.albums.tracks(db_album.item_id, "library")
    assert "An Entirely Different Song" in {track.name for track in tracks}


async def test_dead_mapping_does_not_fail_the_whole_listing(mass: MusicAssistant) -> None:
    """A mapping pointing at a gone album costs its own listing, not the album."""
    db_album = await _seed_album(mass)
    # the library copies must be playable, or the dead mapping's error is all there is
    await set_global_cache_values({"available_providers": {"prov_a_inst", "prov_b_inst"}})
    with _patch_provider_tracks(MediaNotFoundError("Album album_b not found")):
        tracks = await mass.music.albums.tracks(db_album.item_id, "library")
    assert [track.name for track in tracks] == LIBRARY_TITLES
