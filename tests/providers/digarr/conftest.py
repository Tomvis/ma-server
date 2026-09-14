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
from music_assistant.providers.digarr.constants import CONF_API_KEY, CONF_MA_USER, CONF_URL, DOMAIN


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


def make_shared_mass() -> MagicMock:
    """
    Build a mass double with real, in-memory command-registration semantics.

    The single-instance ``provider`` fixture above gives ``mass`` (and therefore
    ``mass.command_handlers``/``mass.register_api_command``/
    ``mass.get_provider_instances``) no real behaviour at all -- every attribute
    access just returns another auto-mock, which is exactly what hides the
    "already registered" RuntimeError this provider is multi-instance: true.
    This double backs those three with a real dict and a real loaded-instances
    list, so two DigarrProvider instances sharing it interact the way they
    would under the real ``MusicAssistant.register_api_command``
    (mass.py:878-905) and ``get_provider_instances`` (mass.py:616-636).

    Every other ``mass`` attribute stays an unconfigured auto-mock, same as the
    single-instance fixture.
    """
    mass = MagicMock()
    mass.command_handlers = {}
    loaded: dict[str, DigarrProvider] = {}

    def register_api_command(command: str, handler: Any, *_args: Any, **_kwargs: Any) -> Any:
        if command in mass.command_handlers:
            msg = f"Command {command} is already registered"
            raise RuntimeError(msg)
        mass.command_handlers[command] = handler

        def unregister() -> None:
            mass.command_handlers.pop(command, None)

        return unregister

    def get_provider_instances(domain: str, **_kwargs: Any) -> list[DigarrProvider]:
        return [prov for prov in loaded.values() if prov.domain == domain]

    mass.register_api_command = MagicMock(side_effect=register_api_command)
    mass.get_provider_instances = MagicMock(side_effect=get_provider_instances)
    mass._loaded = loaded
    return mass


def make_provider(mass: MagicMock, *, ma_user: str, instance_id: str, name: str) -> DigarrProvider:
    """
    Construct a DigarrProvider instance bound to ``ma_user``, sharing ``mass``.

    Pairs with :func:`make_shared_mass`: multiple instances built with the same
    ``mass`` double see each other's registrations via that double's shared
    ``command_handlers``/loaded-instances state, exactly as multiple real
    digarr instances share one running MusicAssistant.

    :param mass: The shared mass double, from :func:`make_shared_mass`.
    :param ma_user: The Music Assistant username this instance is bound to.
    :param instance_id: This instance's unique instance_id.
    :param name: This instance's display (config) name.
    """
    manifest = MagicMock()
    manifest.type = ProviderType.PLUGIN
    manifest.domain = DOMAIN
    config = MagicMock()
    config.name = name
    config.instance_id = instance_id
    values = {CONF_URL: "http://digarr:3000", CONF_API_KEY: "k", CONF_MA_USER: ma_user}
    config.get_value = MagicMock(side_effect=lambda key, default=None: values.get(key, default))
    prov = DigarrProvider(mass, manifest, config, SUPPORTED_FEATURES)
    prov._items = []
    prov._rec_ids = {}
    return prov


async def load_provider(mass: MagicMock, prov: DigarrProvider) -> None:
    """
    Simulate the real load sequence for an already-constructed instance.

    Mirrors ``MusicAssistant._register_loaded_provider``/``_load_provider``
    (mass.py:1355-1424): the instance is registered into the loaded-providers
    registry *before* ``loaded_in_mass`` runs, so a second instance's
    ``get_provider_instances`` lookup during its own ``loaded_in_mass`` already
    sees the first.

    :param mass: The shared mass double the instance was built with.
    :param prov: The constructed (but not yet loaded) instance.
    """
    mass._loaded[prov.instance_id] = prov
    await prov.loaded_in_mass()


async def unload_provider(mass: MagicMock, prov: DigarrProvider, is_removed: bool = False) -> None:
    """
    Simulate the real unload sequence for a loaded instance.

    Mirrors ``MusicAssistant.unload_provider`` (mass.py:1094-1132): the
    instance is only removed from the loaded-providers registry *after*
    ``provider.unload()`` returns, so a hand-off inside ``unload()`` still
    sees itself as loaded while deciding who takes over.

    :param mass: The shared mass double the instance was built with.
    :param prov: The loaded instance to unload.
    :param is_removed: Forwarded to ``prov.unload()``.
    """
    await prov.unload(is_removed)
    mass._loaded.pop(prov.instance_id, None)
