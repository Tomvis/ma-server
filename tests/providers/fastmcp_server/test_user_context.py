"""Tests for ``provider.user_context.CurrentUserMiddleware``."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp.server.auth.auth import AccessToken

from music_assistant.controllers.webserver.helpers.auth_middleware import (
    get_current_user,
    set_current_user,
)
from music_assistant.providers.fastmcp_server import user_context
from music_assistant.providers.fastmcp_server.user_context import CurrentUserMiddleware


def _token(client_id: str) -> AccessToken:
    return AccessToken(token="t", client_id=client_id, scopes=[], expires_at=None)


async def _run(mw: CurrentUserMiddleware) -> Any:
    """Run the middleware and return the current user seen by the handler."""
    seen: dict[str, Any] = {}

    async def call_next(context: Any) -> str:  # noqa: ARG001
        seen["user"] = get_current_user()
        return "ok"

    assert await mw.on_request(MagicMock(), call_next) == "ok"
    return seen["user"]


@pytest.mark.asyncio
async def test_binds_token_user_during_request(
    mock_mass: MagicMock, mock_user: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The handler runs as the user the Bearer token belongs to, then the context is restored."""
    mock_mass.webserver.auth.get_user = AsyncMock(return_value=mock_user)
    monkeypatch.setattr(user_context, "get_access_token", lambda: _token("u1"))
    set_current_user(None)

    assert await _run(CurrentUserMiddleware(mock_mass)) is mock_user
    mock_mass.webserver.auth.get_user.assert_awaited_once_with("u1")
    assert get_current_user() is None


@pytest.mark.asyncio
async def test_no_token_stays_anonymous(
    mock_mass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without auth (require_auth off) the request keeps anonymous playback."""
    mock_mass.webserver.auth.get_user = AsyncMock()
    monkeypatch.setattr(user_context, "get_access_token", lambda: None)
    set_current_user(None)

    assert await _run(CurrentUserMiddleware(mock_mass)) is None
    mock_mass.webserver.auth.get_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_user_stays_anonymous(
    mock_mass: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A token whose user no longer exists never widens access."""
    mock_mass.webserver.auth.get_user = AsyncMock(return_value=None)
    monkeypatch.setattr(user_context, "get_access_token", lambda: _token("gone"))
    set_current_user(None)

    assert await _run(CurrentUserMiddleware(mock_mass)) is None
