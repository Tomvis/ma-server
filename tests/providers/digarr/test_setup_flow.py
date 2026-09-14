"""Tests for the digarr setup flow."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from music_assistant_models.enums import FlowStepType

from music_assistant.models.setup_flow import SetupFlowContext, SetupSession
from music_assistant.providers.digarr.constants import CONF_API_KEY, CONF_MA_USER, CONF_URL
from music_assistant.providers.digarr.setup_flow import run_setup


def _make_session(finish_handler: Any, users: list[str] | None = None) -> SetupSession:
    """Build a SetupSession backed by a Mock mass for driving run_setup directly."""
    mass = MagicMock()
    if users is None:
        mass.webserver.auth.list_users = AsyncMock(side_effect=Exception("no scope"))
    else:
        mass.webserver.auth.list_users = AsyncMock(
            return_value=[MagicMock(username=name) for name in users]
        )
    context = SetupFlowContext(kind="setup", reason="user", domain="digarr")
    return SetupSession(mass, "flow-test", context, finish_handler)


async def _wait_for(predicate: Any, timeout: float = 5.0) -> Any:
    """Wait until the predicate returns truthy (or fail the test)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if result := predicate():
            return result
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met within timeout")


async def _wait_for_form(session: SetupSession, step_id: str = "user") -> Any:
    """Wait until the flow publishes the FORM step with the given step_id."""
    return await _wait_for(
        lambda: (
            session.current_step
            if session.current_step
            and session.current_step.type == FlowStepType.FORM
            and session.current_step.step_id == step_id
            else None
        )
    )


async def test_run_setup_collects_ma_user_as_a_picker() -> None:
    """
    ma_user is asked for during setup, as a picker when the MA user list is available.

    Before this fix, CONF_MA_USER was an options-only entry, so a freshly added
    instance's ma_user was always None and its Discover row was invisible to every
    viewer with no error explaining why.
    """
    collected: dict[str, Any] = {}

    async def finish_handler(_session: SetupSession, values: dict[str, Any]) -> dict[str, str]:
        collected.update(values)
        return {"instance_id": "digarr--test"}

    session = _make_session(finish_handler, users=["tom", "lera"])
    task = asyncio.create_task(run_setup(session))
    await _wait_for_form(session)
    entries = {entry.key: entry for entry in session.current_step.entries}
    assert CONF_MA_USER in entries
    assert [option.value for option in entries[CONF_MA_USER].options] == ["tom", "lera"]

    session.handle_submit(
        {CONF_URL: "http://digarr:3000", CONF_API_KEY: "dgr_x_y", CONF_MA_USER: "tom"}
    )
    await _wait_for(lambda: session.finished)
    await task

    assert collected[CONF_MA_USER] == "tom"


async def test_run_setup_falls_back_to_free_text_when_the_user_list_is_unavailable() -> None:
    """An empty/failed MA user lookup renders ma_user as free text, not a dead-end picker."""

    async def finish_handler(_session: SetupSession, _values: dict[str, Any]) -> dict[str, str]:
        return {"instance_id": "digarr--test"}

    session = _make_session(finish_handler, users=None)
    task = asyncio.create_task(run_setup(session))
    await _wait_for_form(session)
    entries = {entry.key: entry for entry in session.current_step.entries}
    assert entries[CONF_MA_USER].options == []

    session.handle_submit(
        {CONF_URL: "http://digarr:3000", CONF_API_KEY: "dgr_x_y", CONF_MA_USER: "tom"}
    )
    await _wait_for(lambda: session.finished)
    await task
