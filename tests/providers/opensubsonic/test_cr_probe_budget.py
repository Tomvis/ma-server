"""Tests for the whole-album time budget around the critical-reception probe loop."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from music_assistant_models.media_items.metadata import CriticalReception

from music_assistant.providers.opensubsonic import sonic_provider
from music_assistant.providers.opensubsonic.sonic_provider import OpenSonicProvider


def _sonic_album(*song_ids: str) -> Mock:
    album = Mock()
    album.song = [Mock(id=song_id) for song_id in song_ids]
    return album


def _stub_cache(provider: OpenSonicProvider) -> tuple[AsyncMock, AsyncMock]:
    """Point the provider at an always-miss cache and return its (get, set) mocks."""
    cache_get = AsyncMock(return_value=None)
    cache_set = AsyncMock()
    provider.mass.cache.get = cache_get  # type: ignore[method-assign]
    provider.mass.cache.set = cache_set  # type: ignore[method-assign]
    return cache_get, cache_set


@pytest.mark.asyncio
async def test_probe_budget_returns_partial_and_never_caches(
    provider: OpenSonicProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe hanging past the budget returns the partial result, uncached."""
    monkeypatch.setattr(sonic_provider, "_CR_PROBE_ALBUM_BUDGET_SECONDS", 0.25)
    _, cache_set = _stub_cache(provider)
    partial_cr = CriticalReception(amg_dr=9.0)
    hang_started = asyncio.Event()

    async def _probe(song_id: str) -> tuple[CriticalReception | None, float | None] | None:
        if song_id == "song-1":
            # Clean probe: CR found, album DR still missing, so the loop keeps going.
            return partial_cr, None
        hang_started.set()
        await asyncio.sleep(30)
        pytest.fail("probe should have been cancelled by the album budget")

    provider._extract_critical_reception_from_song = AsyncMock(side_effect=_probe)  # type: ignore[method-assign]

    # The outer wait_for is the "must not hang" assertion: without the budget the
    # sleeping probe would hold this call for 30s and blow the 5s ceiling.
    cr, album_dr = await asyncio.wait_for(
        provider._get_album_critical_reception("album-1", _sonic_album("song-1", "song-2")),
        timeout=5,
    )

    assert hang_started.is_set()
    assert cr is partial_cr  # partial result survives the timeout
    assert album_dr is None
    cache_set.assert_not_called()  # a timeout must never pin a partial result for 24h


@pytest.mark.asyncio
async def test_probe_budget_exhausted_with_nothing_collected(
    provider: OpenSonicProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Timing out before any probe completes matches the all-probes-errored path."""
    monkeypatch.setattr(sonic_provider, "_CR_PROBE_ALBUM_BUDGET_SECONDS", 0.25)
    _, cache_set = _stub_cache(provider)

    async def _probe(_song_id: str) -> tuple[CriticalReception | None, float | None] | None:
        await asyncio.sleep(30)
        pytest.fail("probe should have been cancelled by the album budget")

    provider._extract_critical_reception_from_song = AsyncMock(side_effect=_probe)  # type: ignore[method-assign]

    result = await asyncio.wait_for(
        provider._get_album_critical_reception("album-1", _sonic_album("song-1")),
        timeout=5,
    )

    assert result == (None, None)
    cache_set.assert_not_called()


@pytest.mark.asyncio
async def test_probes_completing_within_budget_still_merge_and_cache(
    provider: OpenSonicProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The budget doesn't disturb the normal path: OR-merge, early break, cache write."""
    monkeypatch.setattr(sonic_provider, "_CR_PROBE_ALBUM_BUDGET_SECONDS", 30.0)
    _, cache_set = _stub_cache(provider)
    found_cr = CriticalReception(amg_dr=9.0)
    probed: list[str] = []

    async def _probe(song_id: str) -> tuple[CriticalReception | None, float | None] | None:
        probed.append(song_id)
        if song_id == "song-1":
            return found_cr, None
        return None, 11.5

    provider._extract_critical_reception_from_song = AsyncMock(side_effect=_probe)  # type: ignore[method-assign]

    cr, album_dr = await provider._get_album_critical_reception(
        "album-1", _sonic_album("song-1", "song-2", "song-3")
    )

    assert cr is found_cr
    assert album_dr == 11.5
    assert probed == ["song-1", "song-2"]  # broke out once both signals were seen
    cached: dict[str, Any] = cache_set.await_args.kwargs["data"]
    assert cached["dr"] == 11.5
    assert cached["cr"] is not None


@pytest.mark.asyncio
async def test_probe_raising_its_own_timeout_is_treated_as_transient(
    provider: OpenSonicProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A TimeoutError from inside a probe aborts the loop without pinning the cache."""
    # aiohttp's ServerTimeoutError subclasses TimeoutError, and conn.stream errors are
    # not covered by the probe helper's `except Exception`, so one can reach the album
    # budget's handler having consumed almost none of the budget.
    monkeypatch.setattr(sonic_provider, "_CR_PROBE_ALBUM_BUDGET_SECONDS", 30.0)
    _, cache_set = _stub_cache(provider)
    partial_cr = CriticalReception(amg_dr=9.0)

    async def _probe(song_id: str) -> tuple[CriticalReception | None, float | None] | None:
        if song_id == "song-1":
            return partial_cr, None
        raise TimeoutError("aiohttp read timeout")

    provider._extract_critical_reception_from_song = AsyncMock(side_effect=_probe)  # type: ignore[method-assign]

    cr, album_dr = await asyncio.wait_for(
        provider._get_album_critical_reception("album-1", _sonic_album("song-1", "song-2")),
        timeout=5,
    )

    # same contract as a budget timeout: partial result out, nothing cached, no raise
    assert cr is partial_cr
    assert album_dr is None
    cache_set.assert_not_called()


@pytest.mark.asyncio
async def test_unparsable_album_cached_briefly(
    provider: OpenSonicProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Files ffprobe cannot read are a short negative, not a re-probe every sync."""
    monkeypatch.setattr(sonic_provider, "_CR_PROBE_ALBUM_BUDGET_SECONDS", 30.0)
    _, cache_set = _stub_cache(provider)

    async def _probe(_song_id: str) -> sonic_provider._Unparsable:
        return sonic_provider._PROBE_UNPARSABLE

    provider._extract_critical_reception_from_song = AsyncMock(side_effect=_probe)  # type: ignore[method-assign]

    result = await provider._get_album_critical_reception(
        "album-1", _sonic_album("song-1", "song-2")
    )

    assert result == (None, None)
    cache_set.assert_awaited_once()
    stored = cache_set.await_args_list[0].kwargs
    assert stored["data"] == {"cr": None, "dr": None}
    assert stored["expiration"] == sonic_provider._CR_UNPARSABLE_CACHE_TTL


@pytest.mark.asyncio
async def test_cache_entry_from_before_review_text_is_reprobed(
    provider: OpenSonicProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-v2 entry with review data re-probes once; v2 entries and negatives are served."""
    monkeypatch.setattr(sonic_provider, "_CR_PROBE_ALBUM_BUDGET_SECONDS", 30.0)
    cache_get, cache_set = _stub_cache(provider)
    fresh = CriticalReception(amg_dr=9.0)
    provider._extract_critical_reception_from_song = AsyncMock(return_value=(fresh, 10.0))  # type: ignore[method-assign]

    cache_get.return_value = {"cr": {"amg_dr": 8.0}, "dr": 10.0}
    cr, _ = await provider._get_album_critical_reception("album-1", _sonic_album("song-1"))
    assert cr is fresh
    assert cache_set.await_args.kwargs["data"]["v"] == sonic_provider._CR_CACHE_VERSION

    provider._extract_critical_reception_from_song.reset_mock()
    for entry in (
        {"cr": {"amg_dr": 8.0}, "dr": 10.0, "v": sonic_provider._CR_CACHE_VERSION},
        {"cr": None, "dr": 10.0},
    ):
        cache_get.return_value = entry
        await provider._get_album_critical_reception("album-1", _sonic_album("song-1"))
    provider._extract_critical_reception_from_song.assert_not_called()
