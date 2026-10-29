"""Tests for song bpm/key read from Navidrome's native API (MUSIC-185)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from music_assistant.providers.opensubsonic import sonic_provider
from music_assistant.providers.opensubsonic.sonic_provider import (
    CACHE_CATEGORY_ND_SONG_TAGS,
    OpenSonicProvider,
    _nd_song_tags,
)

from .test_library_changes import T0, T1, T2, _cursor, _Library, _serve


@pytest.fixture(autouse=True)
def _no_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read changes from the cursor exactly."""
    monkeypatch.setattr(sonic_provider, "_ND_CHANGES_OVERLAP", timedelta(0))


def _song(
    song_id: str,
    updated: str = T0,
    bpm: int = 0,
    key: list[str] | None = None,
    missing: bool = False,
) -> dict[str, Any]:
    tags = {"key": key} if key is not None else {}
    return {
        "id": song_id,
        "updatedAt": updated,
        "missing": missing,
        "albumId": None,
        "bpm": bpm,
        "tags": tags,
    }


@pytest.fixture
def stored(provider: OpenSonicProvider) -> list[tuple[str, float | None, str | None]]:
    """Record what the provider hands the audio analysis controller."""
    calls: list[tuple[str, float | None, str | None]] = []

    async def _set_track_tags(
        item_id: str, instance: str, bpm: float | None, key: str | None
    ) -> None:
        assert instance == provider.instance_id
        calls.append((item_id, bpm, key))

    provider.mass.streams.audio_analysis.set_track_tags = _set_track_tags  # type: ignore[method-assign,assignment]
    return calls


@pytest.fixture
def cache(provider: OpenSonicProvider) -> MagicMock:
    """Return the provider's cache, which holds nothing unless a test says so."""
    return cast("MagicMock", provider.mass.cache)


def test_song_tags_read_bpm_and_first_usable_key() -> None:
    """Bpm 0 means none; the first key value is used."""
    assert _nd_song_tags(_song("s", bpm=99, key=["G#m"])) == (99.0, "G#m")
    assert _nd_song_tags(_song("s", key=["C", "Am"])) == (None, "C")
    assert _nd_song_tags(_song("s", key=["", "Am"])) == (None, "Am")
    assert _nd_song_tags({"id": "s"}) == (None, None)


@pytest.mark.asyncio
async def test_changed_songs_store_their_tags(
    provider: OpenSonicProvider, stored: list[Any], cache: MagicMock
) -> None:
    """Songs from the change feed carry their tags, so no extra request is needed."""
    library = _Library(
        song=[
            _song("s1", T2, bpm=120, key=["Em"]),
            _song("s2", T2, missing=True, key=["Am"]),
            _song("s3", T1, bpm=90),
            _song("s4", T0, bpm=80, key=["C"]),
        ],
    )
    _serve(provider, library)

    changes = await provider.get_library_changes(_cursor(artist=T0, album_imported=T0, song=T0))

    assert changes is not None
    assert changes.complete
    assert sorted(stored) == [("s1", 120.0, "Em"), ("s3", 90.0, None)]
    cache.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_incomplete_change_feed_queues_a_backfill(
    provider: OpenSonicProvider, stored: list[Any], cache: MagicMock
) -> None:
    """When the feed can't list every changed song, the next full sync re-reads all tags."""
    _serve(provider, _Library(song=[_song("s1", T1, bpm=120)]))

    changes = await provider.get_library_changes(None)

    assert changes is not None
    assert not changes.complete
    assert stored == []
    cache.delete.assert_awaited_once_with(
        "song_tags", category=CACHE_CATEGORY_ND_SONG_TAGS, provider=provider.instance_id
    )


@pytest.mark.asyncio
async def test_backfill_pages_through_all_songs_once(
    provider: OpenSonicProvider,
    stored: list[Any],
    cache: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All songs are read page by page, then the backfill is marked done for a month."""
    monkeypatch.setattr(sonic_provider, "_ND_CHANGES_PAGE_SIZE", 2)
    songs = [_song("s0", bpm=100, key=["Em"]), _song("s1"), _song("s2", bpm=80)]
    starts: list[str] = []

    async def _get(path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        assert path == "/api/song"
        assert params["missing"] == "false"
        starts.append(params["_start"])
        return songs[int(params["_start"]) : int(params["_end"])]

    provider._nd_get = AsyncMock(side_effect=_get)  # type: ignore[method-assign]

    await provider._nd_backfill_song_tags()

    assert starts == ["0", "2"]
    assert stored == [("s0", 100.0, "Em"), ("s1", None, None), ("s2", 80.0, None)]
    cache.set.assert_awaited_once()
    assert cache.set.await_args.kwargs["expiration"] == 30 * 24 * 3600


@pytest.mark.asyncio
async def test_backfill_skipped_while_marked_done(
    provider: OpenSonicProvider, stored: list[Any], cache: MagicMock
) -> None:
    """A recent backfill means no full pass over the native API."""
    cache.get = AsyncMock(return_value=True)
    provider._nd_get = AsyncMock()  # type: ignore[method-assign]

    await provider._nd_backfill_song_tags()

    provider._nd_get.assert_not_awaited()
    assert stored == []


@pytest.mark.asyncio
async def test_backfill_without_native_api_is_not_marked_done(
    provider: OpenSonicProvider, stored: list[Any], cache: MagicMock
) -> None:
    """A server without the native API has nothing to backfill and nothing to remember."""
    provider._nd_get = AsyncMock(return_value=None)  # type: ignore[method-assign]

    await provider._nd_backfill_song_tags()

    assert stored == []
    cache.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_library_sync_survives_backfill_failure(
    provider: OpenSonicProvider, cache: MagicMock
) -> None:
    """A native API error during the backfill still lets the tracks sync."""
    provider._nd_get = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
    provider.conn = AsyncMock()
    empty = MagicMock()
    empty.song = []
    provider.conn.search3 = AsyncMock(return_value=empty)

    assert [t async for t in provider.get_library_tracks()] == []
    cache.set.assert_not_awaited()
