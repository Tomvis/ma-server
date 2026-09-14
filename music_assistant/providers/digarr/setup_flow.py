"""Setup flow for the digarr plugin provider."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType

from music_assistant.models.setup_flow import SetupFlowError
from music_assistant.providers.digarr import ma_usernames
from music_assistant.providers.digarr.constants import (
    CONF_API_KEY,
    CONF_MA_USER,
    CONF_URL,
    DEFAULT_URL,
)

if TYPE_CHECKING:
    from music_assistant.models.setup_flow import SetupSession

# The digarr URL and API key cannot resolve without user input, so they are
# collected here rather than as options entries (an options entry must be
# answerable offline).
_STATIC_ENTRIES = (
    ConfigEntry(
        key=CONF_URL, type=ConfigEntryType.STRING, required=True, default_value=DEFAULT_URL
    ),
    ConfigEntry(key=CONF_API_KEY, type=ConfigEntryType.SECURE_STRING, required=True),
)


async def run_setup(session: SetupSession) -> None:
    """
    Run the setup flow: collect the digarr URL, API key and bound MA user.

    CONF_MA_USER is an options-only entry in the provider's own declaration, but is
    collected here too: it *could* be answered offline (it only reads the existing MA
    user list), but leaving it options-only meant a freshly added instance was bound
    to no Music Assistant user, so its Discover row was invisible to every viewer with
    no error explaining why. Collecting it here means one pass through setup always
    produces a working instance.
    """
    errors: dict[str, str] | None = None
    setup_data = dict(session.context.setup_data)
    users = await ma_usernames(session.mass, session.mass.logger)
    entries = (
        *_STATIC_ENTRIES,
        ConfigEntry(
            key=CONF_MA_USER,
            type=ConfigEntryType.STRING,
            required=True,
            # An empty list is the framework's own "no options" value, rendering this
            # as free text instead of an unusable empty picker -- see
            # DigarrProvider.get_config_entries, which the picker logic is shared with.
            options=users,
            # The common case is a single-user (or single-member) household: default
            # to it so setup can be finished without hunting for the right username.
            default_value=users[0].value if len(users) == 1 else None,
        ),
    )
    while True:
        form_entries = [
            replace(entry, value=setup_data.get(entry.key, entry.value)) for entry in entries
        ]
        submitted = await session.form(form_entries, step_id="user", errors=errors, last_step=True)
        setup_data.update(submitted)
        try:
            await session.finish(setup_data)
            return
        except SetupFlowError as err:
            errors = {"base": err.translation_key or str(err)}
