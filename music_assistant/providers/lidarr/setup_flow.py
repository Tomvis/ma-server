"""Setup flow for the Lidarr plugin provider."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType

from music_assistant.models.setup_flow import SetupFlowError
from music_assistant.providers.lidarr.constants import CONF_API_KEY, CONF_URL

if TYPE_CHECKING:
    from music_assistant.models.setup_flow import SetupSession

# The Lidarr URL and API key cannot resolve without user input, so they are collected
# here rather than as options entries (an options entry must be answerable offline).
_ENTRIES = (
    ConfigEntry(key=CONF_URL, type=ConfigEntryType.STRING, required=True),
    ConfigEntry(key=CONF_API_KEY, type=ConfigEntryType.SECURE_STRING, required=True),
)


async def run_setup(session: SetupSession) -> None:
    """
    Run the setup flow: collect the Lidarr URL and API key and create the provider.

    On a reconfigure, the form must prefill from the *effective* value -- the same
    values-wins-over-setup_data precedence ``LidarrProvider._config_or_setup_value``
    applies at load -- not from setup_data alone. ``_finish_provider_reconfigure``
    (controllers/config/flows.py) only ever writes setup_data, never touches
    ``values``; the deployed instance has its url in ``values`` with nothing in
    setup_data, so prefilling from setup_data alone renders the field blank on
    Reconfigure. That invites someone to type a new URL believing the (blank) field
    reflects reality, submit it, see the reload report success -- and never notice
    the stale ``values`` entry is still what the client actually uses, since
    ``_config_or_setup_value`` keeps preferring it over the newly-submitted setup
    value. Prefilling from the effective value at least shows the URL that is really
    active, so Reconfigure never presents a misleadingly blank field for a working
    instance.
    """
    errors: dict[str, str] | None = None
    setup_data = dict(session.context.setup_data)
    # Overlay (not merge-under): an explicit options-page value must win over setup_data
    # here exactly as it does in _config_or_setup_value, so the prefill matches what the
    # provider is actually using rather than what setup collected long ago.
    setup_data.update({key: value for key, value in session.context.values.items() if value})
    while True:
        entries = [
            replace(entry, value=setup_data.get(entry.key, entry.value)) for entry in _ENTRIES
        ]
        submitted = await session.form(entries, step_id="user", errors=errors, last_step=True)
        setup_data.update(submitted)
        try:
            await session.finish(setup_data)
            return
        except SetupFlowError as err:
            errors = {"base": err.translation_key or str(err)}
