"""Tests for resolving digarr recommendations to playable items."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import ExternalID, MediaType, ProviderFeature
from music_assistant_models.errors import ProviderUnavailableError
from music_assistant_models.media_items import Artist, ItemMapping, Track

from music_assistant.providers.digarr.client import DigarrRecommendation
from music_assistant.providers.digarr.parsers import resolve_artist, resolve_track

MBID = "c14b4180-dc87-481e-b17a-64e4150f90f6"


def make_rec(**overrides) -> DigarrRecommendation:
    """Build a recommendation with sensible defaults."""
    base = {
        "id": 1,
        "kind": "artist",
        "score": 1.0,
        "status": "pending",
        "artist_id": 101,
        "artist_name": "Opeth",
        "artist_mbid": MBID,
        "image_url": None,
        "genres": (),
        "ai_reasoning": None,
        "release_group_mbid": None,
        "release_group_title": None,
    }
    base.update(overrides)
    return DigarrRecommendation(**base)


def make_provider(instance_id: str) -> MagicMock:
    """
    Build a streaming-provider double that passes `_get_streaming_providers`'s gates.

    :param instance_id: The provider instance id to expose.
    """
    provider = MagicMock()
    provider.instance_id = instance_id
    provider.is_streaming_provider = True
    provider.supported_features = {ProviderFeature.LIBRARY_ARTISTS}
    return provider


@pytest.fixture
def mass() -> MagicMock:
    """Return a mass double whose library and providers both miss by default."""
    mass = MagicMock()
    mass.music.artists.get_library_item_by_external_ids = AsyncMock(return_value=None)
    mass.music.artists.search = AsyncMock(return_value=[])
    mass.music.providers = []
    return mass


async def test_prefers_the_library_copy(mass: MagicMock) -> None:
    """An artist already in the library resolves to the library row, no search."""
    library_artist = Artist(item_id="5", provider="library", name="Opeth", provider_mappings=set())
    mass.music.artists.get_library_item_by_external_ids = AsyncMock(return_value=library_artist)

    resolved = await resolve_artist(make_rec(), mass, "digarr--x")

    assert resolved is library_artist
    mass.music.artists.search.assert_not_called()


async def test_looks_up_the_library_by_musicbrainz_id(mass: MagicMock) -> None:
    """Digarr always supplies an MBID, so it is the primary key for the lookup."""
    await resolve_artist(make_rec(), mass, "digarr--x")
    (external_ids,), _ = mass.music.artists.get_library_item_by_external_ids.call_args
    assert (ExternalID.MB_ARTIST, MBID) in external_ids


async def test_falls_back_to_a_streaming_provider(mass: MagicMock) -> None:
    """Not in the library: the artist resolves to a streaming provider's copy."""
    streaming_artist = Artist(
        item_id="ytm123", provider="ytmusic", name="Opeth", provider_mappings=set()
    )
    mass.music.providers = [make_provider("ytmusic--1")]
    mass.music.artists.search = AsyncMock(return_value=[streaming_artist])

    resolved = await resolve_artist(make_rec(), mass, "digarr--x")

    assert resolved is not None
    assert resolved.provider == "ytmusic"


async def test_rejects_a_mismatched_search_hit(mass: MagicMock) -> None:
    """A provider returning an unrelated artist must not be accepted."""
    wrong = Artist(
        item_id="ytm999",
        provider="ytmusic",
        name="Completely Different Band",
        provider_mappings=set(),
    )
    mass.music.providers = [make_provider("ytmusic--1")]
    mass.music.artists.search = AsyncMock(return_value=[wrong])

    assert await resolve_artist(make_rec(), mass, "digarr--x") is None


async def test_returns_none_when_nothing_resolves(mass: MagicMock) -> None:
    """An artist absent everywhere resolves to nothing, and that is not an error."""
    assert await resolve_artist(make_rec(artist_name="Nonexistent"), mass, "digarr--x") is None


async def test_never_searches_its_own_instance(mass: MagicMock) -> None:
    """The digarr instance is skipped: it owns no music and would recurse."""
    mass.music.providers = [make_provider("digarr--x")]

    await resolve_artist(make_rec(), mass, "digarr--x")

    mass.music.artists.search.assert_not_called()


async def test_a_provider_error_does_not_sink_the_whole_resolution(mass: MagicMock) -> None:
    """One failing provider must not prevent another from resolving the artist."""
    mass.music.providers = [make_provider("tidal--1"), make_provider("ytmusic--1")]

    def fake_search(name: str, instance_id: str, limit: int) -> list[Artist]:  # noqa: ARG001
        if instance_id == "tidal--1":
            raise ProviderUnavailableError("tidal is down")
        return [Artist(item_id="y", provider="ytmusic", name="Opeth", provider_mappings=set())]

    mass.music.artists.search = AsyncMock(side_effect=fake_search)

    resolved = await resolve_artist(make_rec(), mass, "digarr--x")

    assert resolved is not None
    assert resolved.provider == "ytmusic"


def make_track_rec(**overrides) -> DigarrRecommendation:
    """Build a track recommendation with sensible defaults."""
    fields = {
        "kind": "track",
        "recording_mbid": "rec-mbid",
        "track_title": "Windowpane",
        "track_album": "Damnation",
    }
    fields.update(overrides)
    return make_rec(**fields)


def make_track(item_id: str, name: str, artist: str, provider: str = "ytmusic") -> Track:
    """
    Build a streaming track credited to one artist.

    :param item_id: The track's provider item id.
    :param name: The track title.
    :param artist: The credited artist's name.
    :param provider: The provider the track comes from.
    """
    track = Track(item_id=item_id, provider=provider, name=name, provider_mappings=set())
    track.artists.append(
        ItemMapping(media_type=MediaType.ARTIST, item_id="a", provider=provider, name=artist)
    )
    return track


@pytest.fixture
def track_mass() -> MagicMock:
    """Return a mass double whose track library and providers both miss by default."""
    mass = MagicMock()
    mass.music.tracks.get_library_item_by_external_ids = AsyncMock(return_value=None)
    mass.music.tracks.search = AsyncMock(return_value=[])
    provider = make_provider("ytmusic--1")
    provider.supported_features = {ProviderFeature.LIBRARY_TRACKS}
    mass.music.providers = [provider]
    return mass


async def test_track_looks_up_the_library_by_recording_mbid(track_mass: MagicMock) -> None:
    """The recording MBID is the library key, so an owned copy wins over streaming."""
    owned = Track(item_id="9", provider="library", name="Windowpane", provider_mappings=set())
    track_mass.music.tracks.get_library_item_by_external_ids = AsyncMock(return_value=owned)

    resolved = await resolve_track(make_track_rec(), track_mass, "digarr--x")

    assert resolved is owned
    (external_ids,), _ = track_mass.music.tracks.get_library_item_by_external_ids.call_args
    assert (ExternalID.MB_RECORDING, "rec-mbid") in external_ids
    track_mass.music.tracks.search.assert_not_called()


async def test_track_searches_artist_and_title(track_mass: MagicMock) -> None:
    """Not owned: search "Artist - Title" on a streaming provider."""
    hit = make_track("y1", "Windowpane", "Opeth")
    track_mass.music.tracks.search = AsyncMock(return_value=[hit])

    resolved = await resolve_track(make_track_rec(), track_mass, "digarr--x")

    assert resolved is hit
    (query, _instance), _ = track_mass.music.tracks.search.call_args
    assert query == "Opeth - Windowpane"


async def test_track_rejects_the_same_title_by_another_artist(track_mass: MagicMock) -> None:
    """A cover or a namesake song must not stand in for the recommendation."""
    track_mass.music.tracks.search = AsyncMock(
        return_value=[make_track("y1", "Windowpane", "Some Cover Band")]
    )

    assert await resolve_track(make_track_rec(), track_mass, "digarr--x") is None


async def test_track_without_a_title_is_not_resolved(track_mass: MagicMock) -> None:
    """Nothing to search for: no card rather than a guess."""
    assert await resolve_track(make_track_rec(track_title=None), track_mass, "digarr--x") is None
    track_mass.music.tracks.search.assert_not_called()


async def test_track_accepts_the_artist_credited_with_a_leading_the(
    track_mass: MagicMock,
) -> None:
    """Digarr's "Traveling Wilburys" is the providers' "The Traveling Wilburys"."""
    hit = make_track("t1", "Handle With Care", "The Traveling Wilburys")
    track_mass.music.tracks.search = AsyncMock(return_value=[hit])

    rec = make_track_rec(artist_name="Traveling Wilburys", track_title="Handle With Care")
    assert await resolve_track(rec, track_mass, "digarr--x") is hit


async def test_track_skips_a_top_ranked_cover_for_the_original(track_mass: MagicMock) -> None:
    """A cover ranked first must not hide the original further down the same results."""
    cover = make_track("y1", "Windowpane", "Some Cover Band")
    original = make_track("y2", "Windowpane", "Opeth")
    track_mass.music.tracks.search = AsyncMock(return_value=[cover, original])

    assert await resolve_track(make_track_rec(), track_mass, "digarr--x") is original


async def test_artist_accepts_a_leading_the(mass: MagicMock) -> None:
    """An artist search hit differing only by a leading "The" is the same artist."""
    hit = Artist(
        item_id="t", provider="tidal", name="The Traveling Wilburys", provider_mappings=set()
    )
    mass.music.providers = [make_provider("tidal--1")]
    mass.music.artists.search = AsyncMock(return_value=[hit])

    assert (
        await resolve_artist(make_rec(artist_name="Traveling Wilburys"), mass, "digarr--x") is hit
    )
