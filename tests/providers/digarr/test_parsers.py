"""Tests for resolving digarr recommendations to playable items."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import ExternalID, ProviderFeature
from music_assistant_models.errors import ProviderUnavailableError
from music_assistant_models.media_items import Artist

from music_assistant.providers.digarr.client import DigarrRecommendation
from music_assistant.providers.digarr.parsers import resolve_artist

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
