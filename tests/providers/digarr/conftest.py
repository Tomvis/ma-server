"""A dependency-free fake aiohttp session for the digarr client tests."""

from __future__ import annotations

import json as jsonlib
from typing import Any

import pytest


class FakeResponse:
    """Minimal stand-in for aiohttp.ClientResponse."""

    def __init__(self, status: int, payload: Any = None, body: str = "") -> None:
        """Initialize with the status and either a JSON payload or a raw body."""
        self.status = status
        self._payload = payload
        self._body = body

    async def json(self) -> Any:
        """Return the queued JSON payload."""
        return self._payload

    async def text(self) -> str:
        """Return the queued raw body, or the JSON payload serialized as text."""
        return self._body or jsonlib.dumps(self._payload or {})


class _RequestContext:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> FakeResponse:
        return self._response

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class FakeSession:
    """
    Records every request and replays queued responses in order.

    `calls` entries are (method, url, kwargs) so a test can assert on the
    Authorization header and the JSON body without a network stack.
    """

    def __init__(self) -> None:
        """Initialize with an empty call log and an empty response queue."""
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self._queue: list[FakeResponse | Exception] = []

    def queue(self, response: FakeResponse | Exception) -> None:
        """Queue a response, or an exception to raise, for the next request."""
        self._queue.append(response)

    def request(self, method: str, url: str, **kwargs: Any) -> _RequestContext:
        """Record the call and return the next queued response as a context manager."""
        self.calls.append((method, url, kwargs))
        if not self._queue:
            raise AssertionError(f"unexpected request: {method} {url}")
        nxt = self._queue.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return _RequestContext(nxt)


@pytest.fixture
def session() -> FakeSession:
    """Return a fresh fake session per test."""
    return FakeSession()
