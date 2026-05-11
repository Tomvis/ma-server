"""Lidarr Plugin Provider for Music Assistant (music-rater bridge).

Adds an "Add to Lidarr" action exposed through the WebSocket API. The action
hands the album off to music-rater (operator-run companion service), which
owns the Lidarr sync logic. MA just provides the album's MA URI; music-rater
resolves its own album record, sets lidarr_manual_add=True, and runs an
inline single-album sync against its configured Lidarr.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType

from music_assistant.providers.lidarr.client import MusicRaterClient, MusicRaterError
from music_assistant.providers.lidarr.constants import (
    CONF_ACTION_TEST,
    CONF_URL,
    CONF_VERIFY_SSL,
)
from music_assistant.providers.lidarr.provider import LidarrProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType, ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return LidarrProvider(mass, manifest, config)


async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,  # noqa: ARG001 — required by framework signature
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Build the setup form.

    Just the music-rater base URL plus an optional connectivity probe.
    """
    values = dict(values or {})

    url = str(values.get(CONF_URL) or "").strip()
    verify_ssl = bool(values.get(CONF_VERIFY_SSL, True))

    test_ok = False
    test_error: str | None = None

    # Probe only when the user explicitly clicks "Test connection". Including
    # `instance_id is not None` would trigger a 30s blocking HTTP probe every
    # time the config dialog opens for an existing instance — if music-rater
    # is down, the form takes the full timeout to render.
    should_probe = bool(url) and action == CONF_ACTION_TEST
    if should_probe:
        client = MusicRaterClient(url, mass.http_session, verify_ssl=verify_ssl)
        try:
            await client.ping()
            test_ok = True
        except MusicRaterError as err:
            test_error = str(err)
        except Exception as err:
            test_error = f"{type(err).__name__}: {err}"

    return (
        ConfigEntry(
            key="intro",
            type=ConfigEntryType.LABEL,
            label=(
                "Connect Music Assistant to your music-rater instance. The "
                "'Add to Lidarr' context-menu action will POST the album to "
                "music-rater, which owns the Lidarr sync."
            ),
        ),
        ConfigEntry(
            key=CONF_URL,
            type=ConfigEntryType.STRING,
            label="music-rater URL",
            required=True,
            description="e.g. http://192.168.1.10:8000 or https://music-rater.example.com",
        ),
        ConfigEntry(
            key=CONF_VERIFY_SSL,
            type=ConfigEntryType.BOOLEAN,
            label="Verify SSL",
            required=False,
            advanced=True,
            default_value=True,
        ),
        ConfigEntry(
            key=CONF_ACTION_TEST,
            type=ConfigEntryType.ACTION,
            label="Test connection",
            action=CONF_ACTION_TEST,
            action_label="Test connection",
        ),
        ConfigEntry(
            key="test_ok_label",
            type=ConfigEntryType.LABEL,
            label="Connected to music-rater.",
            required=False,
            hidden=not test_ok,
        ),
        ConfigEntry(
            key="test_error_label",
            type=ConfigEntryType.ALERT,
            label=f"Couldn't reach music-rater: {test_error}" if test_error else "",
            required=False,
            hidden=test_error is None,
        ),
    )
