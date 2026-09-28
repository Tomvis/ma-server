"""Constants for the Lidarr plugin provider."""

from __future__ import annotations

CONF_URL = "url"
CONF_API_KEY = "api_key"
CONF_VERIFY_SSL = "verify_ssl"
# One options entry per MA user: f"{CONF_ROOT_FOLDER_PREFIX}{username}" -> Lidarr root path.
CONF_ROOT_FOLDER_PREFIX = "root_folder__"

CONF_ACTION_TEST = "test_connection"
