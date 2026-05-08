"""Constants for the Lidarr plugin provider."""

from __future__ import annotations

CONF_URL = "url"
CONF_API_KEY = "api_key"
CONF_VERIFY_SSL = "verify_ssl"
CONF_ROOT_FOLDER = "root_folder_path"
CONF_QUALITY_PROFILE_ID = "quality_profile_id"
CONF_METADATA_PROFILE_ID = "metadata_profile_id"
CONF_MONITOR_MODE = "monitor_mode"
CONF_SEARCH_ON_ADD = "search_on_add"

CONF_ACTION_TEST = "test_connection"

# Lidarr's monitor enum values; see NewItemMonitorTypes / MonitorTypes in the Lidarr source.
MONITOR_OPTIONS = (
    ("All albums", "all"),
    ("Future albums", "future"),
    ("Missing albums", "missing"),
    ("Existing albums", "existing"),
    ("First album", "first"),
    ("Latest album", "latest"),
    ("None", "none"),
)
