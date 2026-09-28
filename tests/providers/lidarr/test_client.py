"""Tests for the Lidarr HTTP client."""

from __future__ import annotations

from typing import Any, Self

import pytest
from music_assistant_models.errors import LoginFailed, ProviderUnavailableError

from music_assistant.providers.lidarr.client import LidarrClient, LidarrError


class _Response:
    def __init__(self, status: int, body: Any = None) -> None:
        self.status = status
        self._body = body
        self.content_length = None if body is not None else 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def json(self) -> Any:
        return self._body

    async def text(self) -> str:
        return str(self._body)


class _Session:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> _Response:
        self.calls.append((method, url, kwargs))
        return self.response


def _client(response: _Response) -> tuple[LidarrClient, _Session]:
    session = _Session(response)
    return LidarrClient("http://lidarr/", "k3y", session), session  # type: ignore[arg-type]


async def test_sends_the_api_key_header_to_the_v1_api() -> None:
    """Auth is Lidarr's X-Api-Key header; paths are rooted at /api/v1."""
    client, session = _client(_Response(200, [{"id": 2}]))

    assert await client.list_root_folders() == [{"id": 2}]

    method, url, kwargs = session.calls[0]
    assert (method, url) == ("GET", "http://lidarr/api/v1/rootfolder")
    assert kwargs["headers"]["X-Api-Key"] == "k3y"


async def test_rejected_key_raises_login_failed() -> None:
    """401 means a wrong key, surfaced as such rather than as a generic error."""
    client, _ = _client(_Response(401, "no"))

    with pytest.raises(LoginFailed):
        await client.system_status()


async def test_server_error_raises_provider_unavailable() -> None:
    """5xx means Lidarr itself is in trouble."""
    client, _ = _client(_Response(503, "down"))

    with pytest.raises(ProviderUnavailableError):
        await client.list_artists()


async def test_client_error_raises_lidarr_error_with_the_body() -> None:
    """Other 4xx carry Lidarr's validation message to the toast."""
    client, _ = _client(_Response(400, "Path is already configured"))

    with pytest.raises(LidarrError, match="already configured"):
        await client.add_artist({"foreignArtistId": "x"})


async def test_no_content_returns_none() -> None:
    """A 202/204 with no body (e.g. PUT album/monitor) is not an error."""
    client, _ = _client(_Response(202))

    await client.set_albums_monitored([1])  # must not raise
