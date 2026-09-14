"""Tests for the digarr Discover row."""

from __future__ import annotations

from contextlib import AbstractContextManager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.enums import ProviderType
from music_assistant_models.media_items import Artist

from music_assistant.providers.digarr import SUPPORTED_FEATURES, DigarrProvider
from music_assistant.providers.digarr.client import DigarrError, DigarrRecommendation
from music_assistant.providers.digarr.constants import (
    CONF_API_KEY,
    CONF_MA_USER,
    CONF_URL,
    EVENT_RECOMMENDATIONS_UPDATED,
    ROW_ID,
)

MBID = "c14b4180-dc87-481e-b17a-64e4150f90f6"


def make_rec(rec_id: int = 1, name: str = "Opeth", score: float = 1.0) -> DigarrRecommendation:
    """Build a pending artist recommendation."""
    return DigarrRecommendation(
        id=rec_id,
        kind="artist",
        score=score,
        status="pending",
        artist_id=101,
        artist_name=name,
        artist_mbid=MBID,
        image_url=None,
        genres=(),
        ai_reasoning=None,
        release_group_mbid=None,
        release_group_title=None,
    )


@pytest.fixture
def provider() -> DigarrProvider:
    """Construct the provider bound to MA user 'tom'."""
    mass = MagicMock()
    mass.http_session = MagicMock()
    manifest = MagicMock()
    manifest.type = ProviderType.PLUGIN
    manifest.domain = "digarr"
    config = MagicMock()
    config.name = "digarr - Tom"
    config.instance_id = "digarr--abcd1234"
    values = {CONF_URL: "http://digarr:3000", CONF_API_KEY: "k", CONF_MA_USER: "tom"}
    config.get_value = MagicMock(side_effect=lambda key, default=None: values.get(key, default))
    prov = DigarrProvider(mass, manifest, config, SUPPORTED_FEATURES)
    prov._items = []
    prov._rec_ids = {}
    return prov


def as_user(username: str | None) -> AbstractContextManager[MagicMock]:
    """Patch the current-user lookup the provider consults."""
    user = None if username is None else MagicMock(username=username)
    return patch("music_assistant.providers.digarr.get_current_user", return_value=user)


async def test_row_is_shown_to_the_bound_user(provider: DigarrProvider) -> None:
    """The configured user sees exactly one row."""
    with as_user("tom"):
        rows = await provider.get_recommendations()
    assert len(rows) == 1
    assert rows[0].item_id == ROW_ID
    assert rows[0].provider == "digarr--abcd1234"
    assert rows[0].uri == "digarr--abcd1234://folder/up_next"


async def test_row_is_hidden_from_every_other_user(provider: DigarrProvider) -> None:
    """Lera must not see Tom's digarr row: plugin rows bypass the core filter."""
    with as_user("lera"):
        assert await provider.get_recommendations() == []


async def test_row_is_hidden_when_there_is_no_current_user(provider: DigarrProvider) -> None:
    """No identifiable viewer means no per-user row."""
    with as_user(None):
        assert await provider.get_recommendations() == []


async def test_row_survives_an_empty_result(provider: DigarrProvider) -> None:
    """An empty row still renders, so a broken integration looks broken, not absent."""
    with as_user("tom"):
        rows = await provider.get_recommendations()
    assert len(rows) == 1
    assert await provider.get_recommendation_items(ROW_ID) == []


async def test_unknown_row_id_returns_empty(provider: DigarrProvider) -> None:
    """An unrecognised row id yields nothing rather than raising."""
    assert await provider.get_recommendation_items("not-a-row") == []


async def test_refresh_populates_items_and_the_id_map(provider: DigarrProvider) -> None:
    """Refresh resolves each recommendation and records its digarr id by uri."""
    resolved = Artist(item_id="ytm1", provider="ytmusic", name="Opeth", provider_mappings=set())
    provider._client.get_pending = AsyncMock(return_value=[make_rec(rec_id=42)])
    with patch("music_assistant.providers.digarr.resolve_artist", AsyncMock(return_value=resolved)):
        await provider._refresh()

    assert provider._items == [resolved]
    assert provider._rec_ids[resolved.uri] == 42


async def test_refresh_drops_unresolvable_recommendations(provider: DigarrProvider) -> None:
    """An artist on no streaming provider is skipped, not emitted as a dead card."""
    provider._client.get_pending = AsyncMock(return_value=[make_rec(rec_id=1)])
    with patch("music_assistant.providers.digarr.resolve_artist", AsyncMock(return_value=None)):
        await provider._refresh()
    assert provider._items == []


async def test_refresh_keeps_the_previous_generation_when_digarr_fails(
    provider: DigarrProvider,
) -> None:
    """A failed refresh must not blank a working row."""
    existing = Artist(item_id="ytm1", provider="ytmusic", name="Opeth", provider_mappings=set())
    provider._items = [existing]
    provider._rec_ids = {existing.uri: 42}
    provider._client.get_pending = AsyncMock(side_effect=DigarrError("digarr is down"))

    await provider._refresh()

    assert provider._items == [existing]
    assert provider._rec_ids == {existing.uri: 42}


async def test_refresh_signals_the_frontend(provider: DigarrProvider) -> None:
    """The Discover page refreshes itself off this event; no client code needed."""
    provider.signal_provider_event = MagicMock()
    provider._client.get_pending = AsyncMock(return_value=[])
    with patch("music_assistant.providers.digarr.resolve_artist", AsyncMock(return_value=None)):
        await provider._refresh()
    provider.signal_provider_event.assert_called_once_with({"event": EVENT_RECOMMENDATIONS_UPDATED})


async def test_min_score_filters_low_confidence_recommendations(
    provider: DigarrProvider,
) -> None:
    """Recommendations below the configured floor never reach resolution."""
    provider._min_score = 0.9
    provider._client.get_pending = AsyncMock(
        return_value=[make_rec(rec_id=1, score=0.5), make_rec(rec_id=2, score=1.0)]
    )
    resolve = AsyncMock(
        return_value=Artist(item_id="y", provider="ytmusic", name="Opeth", provider_mappings=set())
    )
    with patch("music_assistant.providers.digarr.resolve_artist", resolve):
        await provider._refresh()
    assert resolve.await_count == 1


async def test_recommendation_id_falls_back_to_musicbrainz(provider: DigarrProvider) -> None:
    """A uri that has rotated still maps back via the artist's MusicBrainz id."""
    artist = Artist(item_id="ytm-new", provider="ytmusic", name="Opeth", provider_mappings=set())
    provider._rec_ids = {}
    provider._mbid_rec_ids = {MBID: 42}
    with patch("music_assistant.providers.digarr.mbid_of", MagicMock(return_value=MBID)):
        assert provider.recommendation_id_for(artist.uri, artist) == 42


async def test_recommendation_id_is_none_when_nothing_matches(
    provider: DigarrProvider,
) -> None:
    """Never guess: an unmatched item must not resolve to some other artist."""
    artist = Artist(
        item_id="ytm-unknown",
        provider="ytmusic",
        name="Someone Else",
        provider_mappings=set(),
    )
    assert provider.recommendation_id_for(artist.uri, artist) is None
