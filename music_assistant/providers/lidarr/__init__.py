"""Lidarr Plugin Provider for Music Assistant.

Adds an "Add to Lidarr" action exposed through the WebSocket API. Albums users
discover on streaming providers (Tidal, Spotify, etc.) get sent to the user's
Lidarr instance — the artist is created if missing, then the requested album
is set to monitored so Lidarr will go fetch it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from music_assistant_models.config_entries import (
    ConfigEntry,
    ConfigValueOption,
)
from music_assistant_models.enums import ConfigEntryType

from music_assistant.providers.lidarr.client import LidarrClient, LidarrError
from music_assistant.providers.lidarr.constants import (
    CONF_ACTION_TEST,
    CONF_API_KEY,
    CONF_METADATA_PROFILE_ID,
    CONF_MONITOR_MODE,
    CONF_QUALITY_PROFILE_ID,
    CONF_ROOT_FOLDER,
    CONF_SEARCH_ON_ADD,
    CONF_URL,
    CONF_VERIFY_SSL,
    MONITOR_OPTIONS,
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
    instance_id: str | None = None,
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Build the setup form.

    Two-stage: user enters URL + API key + clicks "Test connection". If the
    probe succeeds, we re-render with root-folder and profile dropdowns
    populated from the live Lidarr; otherwise we surface the error inline.
    """
    values = dict(values or {})

    url = str(values.get(CONF_URL) or "").strip()
    api_key = str(values.get(CONF_API_KEY) or "").strip()
    verify_ssl = bool(values.get(CONF_VERIFY_SSL, True))

    test_ok = False
    test_error: str | None = None
    root_folders: list[dict[str, Any]] = []
    quality_profiles: list[dict[str, Any]] = []
    metadata_profiles: list[dict[str, Any]] = []

    should_probe = bool(url and api_key) and (action == CONF_ACTION_TEST or instance_id is not None)
    if should_probe:
        client = LidarrClient(url, api_key, mass.http_session, verify_ssl=verify_ssl)
        try:
            await client.system_status()
            root_folders = await client.list_root_folders()
            quality_profiles = await client.list_quality_profiles()
            metadata_profiles = await client.list_metadata_profiles()
            test_ok = True
        except LidarrError as err:
            test_error = str(err)
        except Exception as err:
            test_error = f"{type(err).__name__}: {err}"

    options_loaded = test_ok and bool(root_folders) and bool(quality_profiles)

    return (
        ConfigEntry(
            key="intro",
            type=ConfigEntryType.LABEL,
            label=(
                "Connect Music Assistant to your Lidarr instance. Enter the URL "
                "(including http://) and an API key from Lidarr → Settings → "
                "General. Then press Test connection to populate the dropdowns "
                "below from your live Lidarr."
            ),
        ),
        ConfigEntry(
            key=CONF_URL,
            type=ConfigEntryType.STRING,
            label="Lidarr URL",
            required=True,
            description="e.g. http://192.168.1.10:8686 or https://lidarr.example.com",
        ),
        ConfigEntry(
            key=CONF_API_KEY,
            type=ConfigEntryType.SECURE_STRING,
            label="API key",
            required=True,
            description="Found in Lidarr → Settings → General → Security.",
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
            label=(
                f"Connected. Found {len(root_folders)} root folder"
                f"{'' if len(root_folders) == 1 else 's'} and "
                f"{len(quality_profiles)} quality profile"
                f"{'' if len(quality_profiles) == 1 else 's'}."
            ),
            required=False,
            hidden=not test_ok,
        ),
        ConfigEntry(
            key="test_error_label",
            type=ConfigEntryType.ALERT,
            label=f"Couldn't reach Lidarr: {test_error}" if test_error else "",
            required=False,
            hidden=test_error is None,
        ),
        ConfigEntry(
            key=CONF_ROOT_FOLDER,
            type=ConfigEntryType.STRING,
            label="Root folder",
            required=options_loaded,
            hidden=not options_loaded,
            description="Where Lidarr will store music for newly-added artists.",
            options=[ConfigValueOption(rf["path"], rf["path"]) for rf in root_folders],
            default_value=root_folders[0]["path"] if root_folders else None,
        ),
        ConfigEntry(
            key=CONF_QUALITY_PROFILE_ID,
            type=ConfigEntryType.INTEGER,
            label="Quality profile",
            required=options_loaded,
            hidden=not options_loaded,
            options=[ConfigValueOption(qp["name"], qp["id"]) for qp in quality_profiles],
            default_value=quality_profiles[0]["id"] if quality_profiles else None,
        ),
        ConfigEntry(
            key=CONF_METADATA_PROFILE_ID,
            type=ConfigEntryType.INTEGER,
            label="Metadata profile",
            required=options_loaded,
            hidden=not options_loaded,
            options=[ConfigValueOption(mp["name"], mp["id"]) for mp in metadata_profiles],
            default_value=metadata_profiles[0]["id"] if metadata_profiles else None,
        ),
        ConfigEntry(
            key=CONF_MONITOR_MODE,
            type=ConfigEntryType.STRING,
            label="Monitor future releases",
            required=False,
            hidden=not options_loaded,
            description=(
                "When a new artist is created in Lidarr, this controls what "
                "Lidarr does with future releases for that artist. The album "
                "you pushed is always monitored explicitly; existing albums "
                "in the discography are NOT auto-monitored."
            ),
            options=[ConfigValueOption(label, value) for label, value in MONITOR_OPTIONS],
            default_value="all",
        ),
        ConfigEntry(
            key=CONF_SEARCH_ON_ADD,
            type=ConfigEntryType.BOOLEAN,
            label="Trigger album search after monitoring",
            required=False,
            hidden=not options_loaded,
            description=(
                "If enabled, Lidarr will immediately queue an indexer search "
                "for the album you just pushed."
            ),
            default_value=False,
        ),
    )
