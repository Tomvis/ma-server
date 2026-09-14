"""Shared fixtures and fakes for the digarr provider tests."""

from __future__ import annotations

import json as jsonlib
from collections.abc import Generator
from contextlib import AbstractContextManager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from music_assistant_models.enums import ProviderType

from music_assistant.providers.digarr import SUPPORTED_FEATURES, DigarrProvider
from music_assistant.providers.digarr.constants import CONF_API_KEY, CONF_MA_USER, CONF_URL


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


def as_user(username: str | None) -> AbstractContextManager[MagicMock]:
    """Patch the current-user lookup the provider consults."""
    user = None if username is None else MagicMock(username=username)
    return patch("music_assistant.providers.digarr.get_current_user", return_value=user)


@pytest.fixture
def provider() -> Generator[DigarrProvider]:
    """
    Construct the provider bound to MA user 'tom', viewed by 'tom' unless overridden.

    Wraps the whole test in ``as_user("tom")`` so every action/recommendation test
    gets a bound viewer by default without repeating the boilerplate; a test that
    cares about a *different* viewer nests its own ``as_user(...)`` inside the test
    body, which shadows this default for the duration of its own ``with`` block.
    """
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
    with as_user("tom"):
        yield prov
