"""Constants for the Lidarr plugin provider."""

from __future__ import annotations

CONF_URL = "url"
CONF_API_KEY = "api_key"
CONF_VERIFY_SSL = "verify_ssl"
# One options entry per MA user: f"{CONF_ROOT_FOLDER_PREFIX}{username}" -> Lidarr root path.
CONF_ROOT_FOLDER_PREFIX = "root_folder__"

CONF_ACTION_TEST = "test_connection"

# beets-lidarr "song pick" service: picks the best edition of a song and wants it in Lidarr.
CONF_SONG_PICK_URL = "song_pick_url"
DEFAULT_SONG_PICK_URL = "http://10.0.0.101:9394"
