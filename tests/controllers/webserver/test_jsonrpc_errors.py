"""Tests for the error responses of the HTTP JSON-RPC API endpoint."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from aiohttp import StreamReader
from aiohttp.test_utils import make_mocked_request
from music_assistant_models.errors import (
    InsufficientPermissions,
    InvalidDataError,
    MediaNotFoundError,
    MusicAssistantError,
)

from music_assistant.controllers.webserver.controller import WebserverController
from music_assistant.helpers.api import APICommandHandler

if TYPE_CHECKING:
    from aiohttp import web

    from music_assistant.mass import MusicAssistant


@pytest.fixture
def webserver(mass_minimal: MusicAssistant) -> WebserverController:
    """Return a WebserverController that dispatches unauthenticated JSON-RPC commands."""
    webserver = WebserverController(mass_minimal)
    mass_minimal.webserver = webserver
    # without users the handler answers 503 before it ever dispatches a command
    webserver.auth._has_users = True
    return webserver


async def dispatch_failing_command(
    webserver: WebserverController, mass: MusicAssistant, error: Exception
) -> web.Response:
    """Dispatch a JSON-RPC command whose handler raises the given error."""

    async def _raise() -> None:
        raise error

    mass.command_handlers["test/raise"] = APICommandHandler.parse(
        "test/raise", _raise, authenticated=False
    )
    payload = StreamReader(Mock(), 2**16, loop=asyncio.get_running_loop())
    payload.feed_data(b'{"message_id": "1", "command": "test/raise"}')
    payload.feed_eof()
    request = make_mocked_request("POST", "/api", payload=payload)
    return await webserver._handle_jsonrpc_api_command(request)


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        # 403 and 400 are what the dedicated InsufficientPermissions/InvalidDataError
        # arms returned before they were folded into the generic one; pinned literally
        # so a change to the models' http_status surfaces here as a failure.
        (InsufficientPermissions("no access"), 403),
        (InvalidDataError("bad data"), 400),
        (MediaNotFoundError("gone"), 404),
    ],
    ids=["insufficient_permissions", "invalid_data", "media_not_found"],
)
async def test_domain_error_returns_json_body(
    mass_minimal: MusicAssistant,
    webserver: WebserverController,
    error: MusicAssistantError,
    expected_status: int,
) -> None:
    """Every MusicAssistantError is reported as JSON with the error's own http_status."""
    response = await dispatch_failing_command(webserver, mass_minimal, error)

    assert response.status == expected_status
    assert response.content_type == "application/json"
    assert json.loads(response.body) == {  # type: ignore[arg-type]
        "error": type(error).__name__,
        "message": str(error),
        "code": error.error_code,
    }


async def test_unexpected_error_returns_opaque_json_body(
    mass_minimal: MusicAssistant, webserver: WebserverController
) -> None:
    """A non-MusicAssistantError yields a 500 JSON body that leaks no exception details."""
    response = await dispatch_failing_command(
        webserver, mass_minimal, RuntimeError("secret internals")
    )

    assert response.status == 500
    assert response.content_type == "application/json"
    assert json.loads(response.body) == {  # type: ignore[arg-type]
        "error": "MusicAssistantError",
        "message": "Internal server error",
        "code": 999,
    }
