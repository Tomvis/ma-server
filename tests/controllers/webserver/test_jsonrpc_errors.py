"""Fork: HTTP status of JSON-RPC command errors (upstream 2.11.0.dev2026112003 owns the body)."""

from __future__ import annotations

import pytest
from music_assistant_models.errors import (
    AlreadyInLibraryError,
    AlreadyInListenLaterError,
    InsufficientPermissions,
    InvalidDataError,
    MediaNotFoundError,
    MusicAssistantError,
)

from music_assistant.controllers.webserver.controller import _error_status


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        # fork (fb924ce8b): re-adding what is already there is a no-op clients can ignore
        (AlreadyInLibraryError("in library"), 409),
        (AlreadyInListenLaterError("saved"), 409),
        (InsufficientPermissions("no access"), 403),
        (InvalidDataError("bad data"), 400),
        (MediaNotFoundError("gone"), 404),
    ],
    ids=["already_in_library", "already_in_listen_later", "forbidden", "invalid", "not_found"],
)
def test_error_status(error: MusicAssistantError, expected_status: int) -> None:
    """Each command error answers with its status, the fork's AlreadyIn* ones with 409."""
    assert _error_status(error) == expected_status
