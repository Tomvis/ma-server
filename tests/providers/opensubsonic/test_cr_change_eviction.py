"""Tests for evicting cached critical reception of albums whose files changed on Navidrome."""

from __future__ import annotations

from typing import Any, Self
from unittest.mock import AsyncMock

import pytest

from music_assistant.providers.opensubsonic import sonic_provider
from music_assistant.providers.opensubsonic.sonic_provider import (
    CACHE_CATEGORY_CR_WATERMARK,
    CACHE_CATEGORY_CRITICAL_RECEPTION,
    OpenSonicProvider,
)


def _song(album_id: str, updated: str) -> dict[str, Any]:
    return {"albumId": album_id, "updatedAt": updated}


def _serve_songs(provider: OpenSonicProvider, songs: list[dict[str, Any]] | None) -> list[int]:
    """Answer native song pages from ``songs`` (newest first); return the requested starts."""
    starts: list[int] = []

    async def _page(start: int, end: int) -> list[dict[str, Any]] | None:
        starts.append(start)
        return None if songs is None else songs[start:end]

    provider._nd_song_page = AsyncMock(side_effect=_page)  # type: ignore[method-assign]
    return starts


def _stub_cache(provider: OpenSonicProvider, watermark: str | None) -> tuple[AsyncMock, AsyncMock]:
    """Return (set, delete) mocks of a cache holding only the change watermark."""

    async def _get(category: int = 0, **_kwargs: Any) -> Any:
        return watermark if category == CACHE_CATEGORY_CR_WATERMARK else None

    cache_set = AsyncMock()
    cache_delete = AsyncMock()
    provider.mass.cache.get = AsyncMock(side_effect=_get)  # type: ignore[method-assign]
    provider.mass.cache.set = cache_set  # type: ignore[method-assign]
    provider.mass.cache.delete = cache_delete  # type: ignore[method-assign]
    return cache_set, cache_delete


def _evicted(cache_delete: AsyncMock) -> set[str]:
    assert all(
        c.kwargs["category"] == CACHE_CATEGORY_CRITICAL_RECEPTION
        for c in cache_delete.await_args_list
    )
    return {c.kwargs["key"].split(":", 1)[1] for c in cache_delete.await_args_list}


def _stored_watermark(cache_set: AsyncMock) -> str:
    (call,) = cache_set.await_args_list
    assert call.kwargs["category"] == CACHE_CATEGORY_CR_WATERMARK
    return str(call.kwargs["data"])


@pytest.mark.asyncio
async def test_albums_changed_since_watermark_are_evicted(provider: OpenSonicProvider) -> None:
    """Every album with a song updated after the watermark loses its CR cache entry."""
    cache_set, cache_delete = _stub_cache(provider, "2026-10-14T10:00:00Z")
    _serve_songs(
        provider,
        [
            _song("album-a", "2026-10-14T11:03:06.499158875Z"),
            _song("album-a", "2026-10-14T11:03:06.4991588Z"),
            _song("album-b", "2026-10-14T10:30:00Z"),
            _song("album-c", "2026-10-14T10:00:00Z"),  # == watermark: already handled
            _song("album-d", "2026-10-13T08:00:00Z"),
        ],
    )

    await provider._evict_changed_album_cr()

    assert _evicted(cache_delete) == {"album-a", "album-b"}
    assert _stored_watermark(cache_set) == "2026-10-14T11:03:06.499158875Z"


@pytest.mark.asyncio
async def test_first_run_records_watermark_without_evicting(provider: OpenSonicProvider) -> None:
    """With no watermark yet there is no baseline, so nothing is evicted."""
    cache_set, cache_delete = _stub_cache(provider, None)
    _serve_songs(provider, [_song("album-a", "2026-10-14T11:00:00Z")])

    await provider._evict_changed_album_cr()

    cache_delete.assert_not_called()
    assert _stored_watermark(cache_set) == "2026-10-14T11:00:00Z"


@pytest.mark.asyncio
async def test_pages_until_watermark_is_reached(
    provider: OpenSonicProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changes spanning several pages are all collected, and paging stops at the watermark."""
    monkeypatch.setattr(sonic_provider, "_ND_SONG_PAGE_SIZE", 2)
    _, cache_delete = _stub_cache(provider, "2026-10-14T09:00:00Z")
    starts = _serve_songs(
        provider,
        [
            _song("album-a", "2026-10-14T12:00:00Z"),
            _song("album-b", "2026-10-14T11:00:00Z"),
            _song("album-c", "2026-10-14T10:00:00Z"),
            _song("album-d", "2026-10-14T08:00:00Z"),
            _song("album-e", "2026-10-14T07:00:00Z"),
            _song("album-f", "2026-10-14T06:00:00Z"),
        ],
    )

    await provider._evict_changed_album_cr()

    assert _evicted(cache_delete) == {"album-a", "album-b", "album-c"}
    assert starts == [0, 2]


@pytest.mark.asyncio
async def test_unchanged_library_evicts_nothing(provider: OpenSonicProvider) -> None:
    """A quiet library leaves the cache and the watermark alone."""
    cache_set, cache_delete = _stub_cache(provider, "2026-10-14T11:00:00Z")
    _serve_songs(provider, [_song("album-a", "2026-10-14T11:00:00Z")])

    await provider._evict_changed_album_cr()

    cache_delete.assert_not_called()
    cache_set.assert_not_called()


@pytest.mark.asyncio
async def test_without_native_api_falls_back_to_ttl(provider: OpenSonicProvider) -> None:
    """A server without Navidrome's native API keeps the plain TTL behaviour."""
    cache_set, cache_delete = _stub_cache(provider, "2026-10-14T11:00:00Z")
    _serve_songs(provider, None)

    await provider._evict_changed_album_cr()

    cache_delete.assert_not_called()
    cache_set.assert_not_called()


@pytest.mark.asyncio
async def test_eviction_failure_never_aborts_the_sync(provider: OpenSonicProvider) -> None:
    """Any error while checking for changes is swallowed; the sync goes on."""
    _, cache_delete = _stub_cache(provider, "2026-10-14T11:00:00Z")
    provider._nd_song_page = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

    await provider._evict_changed_album_cr()

    cache_delete.assert_not_called()


@pytest.mark.asyncio
async def test_library_album_sync_evicts_before_enriching(provider: OpenSonicProvider) -> None:
    """The library album sync checks for changed files before reading any cached CR."""
    calls: list[str] = []
    provider._evict_changed_album_cr = AsyncMock(  # type: ignore[method-assign]
        side_effect=lambda: calls.append("evict")
    )

    async def _album_list(**_kwargs: Any) -> list[Any]:
        calls.append("list")
        return []

    provider.conn = AsyncMock()
    provider.conn.get_album_list2 = AsyncMock(side_effect=_album_list)

    assert [a async for a in provider.get_library_albums()] == []
    assert calls[0] == "evict"


class _Resp:
    def __init__(self, status: int, body: Any = None) -> None:
        self.status = status
        self._body = body

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def json(self, **_kwargs: Any) -> Any:
        return self._body

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


class _Session:
    """Replay canned responses for login (post) and song pages (get)."""

    def __init__(self, logins: list[_Resp], pages: list[_Resp]) -> None:
        self.logins = logins
        self.pages = pages
        self.tokens: list[str] = []

    def post(self, url: str, **_kwargs: Any) -> _Resp:
        assert url == "http://navidrome:80/auth/login"
        return self.logins.pop(0)

    def get(self, url: str, headers: dict[str, str], **_kwargs: Any) -> _Resp:
        assert url == "http://navidrome:80/api/song"
        self.tokens.append(headers["x-nd-authorization"])
        return self.pages.pop(0)


def _native(provider: OpenSonicProvider, session: _Session) -> None:
    provider._nd_base_url = "http://navidrome:80"
    provider._nd_login = {"username": "u", "password": "p"}
    provider.mass.http_session = session  # type: ignore[misc,assignment]


@pytest.mark.asyncio
async def test_expired_session_logs_in_again(provider: OpenSonicProvider) -> None:
    """A 401 on the song list re-authenticates once and retries."""
    page = [_song("album-a", "2026-10-14T11:00:00Z")]
    session = _Session(
        logins=[_Resp(200, {"token": "t1"}), _Resp(200, {"token": "t2"})],
        pages=[_Resp(200, page), _Resp(401), _Resp(200, page)],
    )
    _native(provider, session)

    assert await provider._nd_song_page(0, 1) == page
    assert await provider._nd_song_page(0, 1) == page
    assert session.tokens == ["Bearer t1", "Bearer t1", "Bearer t2"]


@pytest.mark.asyncio
async def test_server_without_native_api_is_not_asked_again(provider: OpenSonicProvider) -> None:
    """A login endpoint that isn't Navidrome's disables the native API for the session."""
    session = _Session(logins=[_Resp(404)], pages=[])
    _native(provider, session)

    assert await provider._nd_song_page(0, 1) is None
    assert await provider._nd_song_page(0, 1) is None  # no second login attempt


@pytest.mark.asyncio
async def test_login_answering_html_disables_native_api(provider: OpenSonicProvider) -> None:
    """A 200 that isn't Navidrome's JSON login reply also disables the native API."""

    class _Html(_Resp):
        async def json(self, **_kwargs: Any) -> Any:
            raise ValueError("not JSON")

    session = _Session(logins=[_Html(200)], pages=[])
    _native(provider, session)

    assert await provider._nd_song_page(0, 1) is None
    assert provider._nd_native is False
