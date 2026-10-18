"""Tests for the song-pick service client (MUSIC-115)."""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from music_assistant.providers.lidarr import song_pick
from music_assistant.providers.lidarr.client import LidarrError
from music_assistant.providers.lidarr.song_pick import want_song

BODY = {
    "root_folder": "/music/tom",
    "artist_mbid": "opeth-mbid",
    "artist_name": "Opeth",
    "track_name": "Bleak",
    "recording_mbid": None,
    "isrc": "SE1234567890",
    "album_rg_mbid": "rg-mbid",
}


async def _serve(status: int, payload: dict[str, Any], seen: list[Any]) -> TestServer:
    async def handler(request: web.Request) -> web.Response:
        seen.append((request.method, request.path, await request.json()))
        return web.json_response(payload, status=status)

    app = web.Application()
    app.router.add_post("/song", handler)
    server = TestServer(app)
    await server.start_server()
    return server


async def test_posts_body_and_returns_the_reply() -> None:
    """The body goes to {base}/song verbatim; a 200 comes back parsed."""
    reply = {"status": "wanted", "artist_name": "Opeth", "artist_added": False}
    seen: list[Any] = []
    server = await _serve(200, reply, seen)
    try:
        async with aiohttp.ClientSession() as session:
            # trailing slash on the base URL must not double up
            result = await want_song(session, f"{server.make_url('/')}", BODY)
    finally:
        await server.close()
    assert result == reply
    assert seen == [("POST", "/song", BODY)]


async def test_error_body_becomes_lidarr_error() -> None:
    """A 409 {"error": ...} surfaces its message to the UI."""
    server = await _serve(409, {"error": "Bleak is on no release Lidarr monitors"}, [])
    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(LidarrError, match="Bleak is on no release"):
                await want_song(session, str(server.make_url("")), BODY)
    finally:
        await server.close()


async def test_non_json_error_still_raises_lidarr_error() -> None:
    """A proxy's HTML 502 yields a LidarrError naming the status, not a decode crash."""

    async def handler(_: web.Request) -> web.Response:
        return web.Response(status=502, text="<html>bad gateway</html>")

    app = web.Application()
    app.router.add_post("/song", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(LidarrError, match="502"):
                await want_song(session, str(server.make_url("")), BODY)
    finally:
        await server.close()


async def test_timeout_becomes_lidarr_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A service that never answers ends in a LidarrError, not a hang."""

    async def handler(_: web.Request) -> web.Response:
        await asyncio.sleep(5)
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/song", handler)
    server = TestServer(app)
    await server.start_server()
    monkeypatch.setattr(song_pick, "REQUEST_TIMEOUT", 0.2)
    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(LidarrError, match="timed out"):
                await want_song(session, str(server.make_url("")), BODY)
    finally:
        await server.close()


def test_timeout_covers_a_new_artist() -> None:
    """New artist plus a profile switch's settle wait is ~480 s before MB/DRDB (R12)."""
    assert song_pick.REQUEST_TIMEOUT == 600
