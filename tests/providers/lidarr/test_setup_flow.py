"""Tests for the lidarr setup flow."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

from music_assistant_models.enums import FlowStepType

from music_assistant.models.setup_flow import SetupFlowContext, SetupSession
from music_assistant.providers.lidarr.constants import CONF_URL
from music_assistant.providers.lidarr.setup_flow import run_setup


def _make_session(
    finish_handler: Any,
    *,
    setup_data: dict[str, Any] | None = None,
    values: dict[str, Any] | None = None,
) -> SetupSession:
    """Build a SetupSession backed by a Mock mass for driving run_setup directly."""
    mass = MagicMock()
    context = SetupFlowContext(
        kind="reconfigure" if values else "setup",
        reason="user",
        domain="lidarr",
        instance_id="lidarr" if values else None,
        setup_data=setup_data or {},
        values=values or {},
    )
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


async def test_reconfigure_prefills_from_the_effective_value_not_setup_data_alone() -> None:
    """
    Reconfigure must prefill from the active url, not a blank setup_data.

    ``_finish_provider_reconfigure`` (controllers/config/flows.py) only ever writes
    setup_data, never ``values`` -- so an instance whose url lives in ``values`` (the
    deployed shape) has nothing in setup_data at all. Prefilling from setup_data
    alone would render this field blank on Reconfigure, inviting someone to type a
    new url believing the blank field reflects reality, while the stale ``values``
    entry silently keeps winning (see ``LidarrProvider._config_or_setup_value``) --
    the submission would report success and change nothing.
    """

    async def finish_handler(_session: SetupSession, _values: dict[str, Any]) -> dict[str, str]:
        return {"instance_id": "lidarr"}

    session = _make_session(finish_handler, setup_data={}, values={CONF_URL: "http://music-rater"})
    task = asyncio.create_task(run_setup(session))
    await _wait_for_form(session)

    assert session.current_step is not None
    entries = {entry.key: entry for entry in session.current_step.entries}
    assert entries[CONF_URL].value == "http://music-rater"

    session.handle_submit({CONF_URL: "http://music-rater"})
    await _wait_for(lambda: session.finished)
    await task


async def test_reconfigure_prefers_the_options_value_over_a_conflicting_setup_value() -> None:
    """A `values` edit must still win the prefill over a (stale) setup_data value."""

    async def finish_handler(_session: SetupSession, _values: dict[str, Any]) -> dict[str, str]:
        return {"instance_id": "lidarr"}

    session = _make_session(
        finish_handler,
        setup_data={CONF_URL: "http://stale-from-setup"},
        values={CONF_URL: "http://fixed-with-port:4533"},
    )
    task = asyncio.create_task(run_setup(session))
    await _wait_for_form(session)

    assert session.current_step is not None
    entries = {entry.key: entry for entry in session.current_step.entries}
    assert entries[CONF_URL].value == "http://fixed-with-port:4533"

    session.handle_submit({CONF_URL: "http://fixed-with-port:4533"})
    await _wait_for(lambda: session.finished)
    await task
