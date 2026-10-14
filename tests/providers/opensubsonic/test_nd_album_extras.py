"""Tests for album DR and release-group ids read from Navidrome's native API (MUSIC-84)."""

from __future__ import annotations

from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import ExternalID

from music_assistant.providers.opensubsonic import sonic_provider
from music_assistant.providers.opensubsonic.sonic_provider import (
    OpenSonicProvider,
    _nd_album_extras,
)

from .test_provider import _use_dict_cache

RGID = "809cb66e-5ba7-4600-a6e6-32c0822fdc38"


def _nd_album(album_id: str, dr: list[str] | None = None, rgid: str = "") -> dict[str, Any]:
    tags = {"genre": ["Black Metal"]}
    if dr is not None:
        tags["album dynamic range"] = dr
    return {"id": album_id, "mbzReleaseGroupId": rgid, "tags": tags}


def test_extras_read_dr_and_release_group() -> None:
    """The album-scope DR tag and the files' release-group id are both returned."""
    assert _nd_album_extras(_nd_album("a", ["9"], RGID)) == (9.0, RGID)


def test_conflicting_dr_values_report_none() -> None:
    """Tracks disagreeing on the album DR give no value, so the stored one is kept."""
    assert _nd_album_extras(_nd_album("a", ["10", "7"])) == (None, None)


def test_album_without_tags_reports_nothing() -> None:
    """No DR tag and no release-group id: nothing to report."""
    assert _nd_album_extras({"id": "a"}) == (None, None)
    assert _nd_album_extras(_nd_album("a", ["n/a"])) == (None, None)


def _sonic_album(album_id: str) -> MagicMock:
    album = MagicMock()
    album.id = album_id
    return album


@pytest.mark.asyncio
async def test_library_sync_applies_extras(
    provider: OpenSonicProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each synced album gets its DR and release-group id from one pass over the native API."""
    parsed: dict[str, MagicMock] = {}

    def _parse(_logger: Any, _instance: str, sonic: MagicMock, *_args: Any) -> MagicMock:
        parsed[sonic.id] = MagicMock()
        return parsed[sonic.id]

    monkeypatch.setattr(sonic_provider, "parse_album", _parse)
    provider.conn = AsyncMock()
    provider.conn.get_album_list2 = AsyncMock(
        side_effect=[[_sonic_album("a"), _sonic_album("b")], []]
    )
    pages = [[_nd_album("a", ["9"], RGID), _nd_album("b")]]
    provider._nd_get = AsyncMock(side_effect=lambda *_a, **_k: pages.pop(0) if pages else [])  # type: ignore[method-assign]

    albums = [a async for a in provider.get_library_albums()]

    assert albums == [parsed["a"], parsed["b"]]
    assert parsed["a"].metadata.dynamic_range == 9.0
    parsed["a"].add_external_id.assert_called_once_with(ExternalID.MB_RELEASEGROUP, RGID)
    parsed["b"].add_external_id.assert_not_called()


@pytest.mark.asyncio
async def test_library_sync_survives_native_api_failure(
    provider: OpenSonicProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A native API error leaves the albums without extras but still syncs them."""
    monkeypatch.setattr(sonic_provider, "parse_album", lambda *_a: MagicMock())
    provider.conn = AsyncMock()
    provider.conn.get_album_list2 = AsyncMock(side_effect=[[_sonic_album("a")], []])
    provider._nd_get = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

    assert len([a async for a in provider.get_library_albums()]) == 1


@pytest.mark.asyncio
async def test_library_extras_page_through_all_albums(
    provider: OpenSonicProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Album pages are requested until a short page comes back."""
    monkeypatch.setattr(sonic_provider, "_ND_ALBUM_PAGE_SIZE", 2)
    albums = [_nd_album(str(i), ["8"]) for i in range(3)]
    starts: list[str] = []

    async def _get(path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        assert path == "/api/album"
        starts.append(params["_start"])
        return albums[int(params["_start"]) : int(params["_end"])]

    provider._nd_get = AsyncMock(side_effect=_get)  # type: ignore[method-assign]

    extras = await provider._nd_library_album_extras()

    assert extras == {"0": (8.0, None), "1": (8.0, None), "2": (8.0, None)}
    assert starts == ["0", "2"]


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
    """Replay canned responses for login (post) and native API reads (get)."""

    def __init__(self, logins: list[_Resp], pages: list[_Resp]) -> None:
        self.logins = logins
        self.pages = pages
        self.tokens: list[str] = []

    def post(self, url: str, **_kwargs: Any) -> _Resp:
        assert url == "http://navidrome:80/auth/login"
        return self.logins.pop(0)

    def get(self, url: str, headers: dict[str, str], **_kwargs: Any) -> _Resp:
        assert url == "http://navidrome:80/api/album"
        self.tokens.append(headers["x-nd-authorization"])
        return self.pages.pop(0)


def _native(provider: OpenSonicProvider, session: _Session) -> None:
    provider._nd_base_url = "http://navidrome:80"
    provider._nd_login = {"username": "u", "password": "p"}
    provider.mass.http_session = session  # type: ignore[misc,assignment]


@pytest.mark.asyncio
async def test_expired_session_logs_in_again(provider: OpenSonicProvider) -> None:
    """A 401 re-authenticates once and retries."""
    page = [_nd_album("a")]
    session = _Session(
        logins=[_Resp(200, {"token": "t1"}), _Resp(200, {"token": "t2"})],
        pages=[_Resp(200, page), _Resp(401), _Resp(200, page)],
    )
    _native(provider, session)

    assert await provider._nd_get("/api/album") == page
    assert await provider._nd_get("/api/album") == page
    assert session.tokens == ["Bearer t1", "Bearer t1", "Bearer t2"]


@pytest.mark.asyncio
async def test_server_without_native_api_is_not_asked_again(provider: OpenSonicProvider) -> None:
    """A login endpoint that isn't Navidrome's disables the native API for the session."""
    session = _Session(logins=[_Resp(404)], pages=[])
    _native(provider, session)

    assert await provider._nd_get("/api/album") is None
    assert await provider._nd_get("/api/album") is None  # no second login attempt


@pytest.mark.asyncio
async def test_login_answering_html_disables_native_api(provider: OpenSonicProvider) -> None:
    """A 200 that isn't Navidrome's JSON login reply also disables the native API."""

    class _Html(_Resp):
        async def json(self, **_kwargs: Any) -> Any:
            raise ValueError("not JSON")

    session = _Session(logins=[_Html(200)], pages=[])
    _native(provider, session)

    assert await provider._nd_get("/api/album") is None
    assert provider._nd_native is False


@pytest.mark.asyncio
async def test_get_album_applies_extras(
    provider: OpenSonicProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single album fetch reads its DR and release-group id from the native API too."""
    parsed = MagicMock()
    monkeypatch.setattr(sonic_provider, "parse_album", lambda *_a: parsed)
    _use_dict_cache(provider)
    provider.conn = AsyncMock()
    provider._nd_get = AsyncMock(return_value=_nd_album("a", ["11"], RGID))  # type: ignore[method-assign]

    await provider.get_album("a")

    provider._nd_get.assert_awaited_once_with("/api/album/a")
    assert parsed.metadata.dynamic_range == 11.0
    parsed.add_external_id.assert_called_once_with(ExternalID.MB_RELEASEGROUP, RGID)
