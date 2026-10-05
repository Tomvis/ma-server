"""Tests for the aiohttp client session helper."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp_asyncmdnsresolver.api import AsyncDualMDNSResolver

from music_assistant.helpers import aiohttp_client


@pytest.fixture
async def proxy_server() -> AsyncGenerator[tuple[str, list[str]]]:
    """Run a local forward proxy that records the URLs it is asked to fetch."""
    requested: list[str] = []

    async def _handler(request: web.Request) -> web.Response:
        requested.append(request.raw_path)
        return web.Response(text="via proxy")

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", _handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    yield f"http://127.0.0.1:{port}", requested
    await runner.cleanup()


async def test_clientsession_honors_proxy_env(
    proxy_server: tuple[str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test requests are routed through the proxy set in the environment."""
    proxy_url, requested = proxy_server
    monkeypatch.setenv("HTTP_PROXY", proxy_url)
    monkeypatch.setenv("http_proxy", proxy_url)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    mass = MagicMock()
    mass.version = "test"
    # the hostname is unresolvable, so the request can only succeed via the proxy
    with patch.object(aiohttp_client, "_get_resolver", return_value=None):
        session = aiohttp_client.create_clientsession(mass)
        async with session.get("http://origin.invalid/hello") as response:
            assert await response.text() == "via proxy"
        await session.close()
    assert requested == ["http://origin.invalid/hello"]


async def test_clientsession_ignores_env_without_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the environment is not trusted when no proxy is configured."""
    monkeypatch.setattr(aiohttp_client, "getproxies", dict)
    mass = MagicMock()
    mass.version = "test"
    with patch.object(aiohttp_client, "_get_resolver", return_value=None):
        session = aiohttp_client.create_clientsession(mass)
        assert session.trust_env is False
        await session.close()


async def test_resolver_retries_failed_lookup_with_system_resolver() -> None:
    """A lookup c-ares fails (possibly from its negative cache) is retried uncached."""
    resolver = aiohttp_client.MassAsyncDNSResolver()
    found = [{"hostname": "kokoro", "host": "172.16.1.12", "port": 8880}]
    with (
        patch.object(
            AsyncDualMDNSResolver, "resolve", AsyncMock(side_effect=OSError(None, "not found"))
        ),
        patch.object(resolver, "_system_resolver") as system_resolver,
    ):
        system_resolver.resolve = AsyncMock(return_value=found)
        assert await resolver.resolve("kokoro", 8880) == found
        system_resolver.resolve.assert_awaited_once()
    await resolver.real_close()


async def test_resolver_skips_system_resolver_on_success() -> None:
    """A successful c-ares lookup is returned as is."""
    resolver = aiohttp_client.MassAsyncDNSResolver()
    found = [{"hostname": "kokoro", "host": "172.16.1.12", "port": 8880}]
    with (
        patch.object(AsyncDualMDNSResolver, "resolve", AsyncMock(return_value=found)),
        patch.object(resolver, "_system_resolver") as system_resolver,
    ):
        assert await resolver.resolve("kokoro", 8880) == found
        system_resolver.resolve.assert_not_called()
    await resolver.real_close()
