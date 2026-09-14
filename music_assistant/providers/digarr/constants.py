"""Constants for the digarr plugin provider."""

from __future__ import annotations

from typing import Final

DOMAIN: Final[str] = "digarr"

CONF_URL: Final[str] = "url"
CONF_API_KEY: Final[str] = "api_key"
CONF_MA_USER: Final[str] = "ma_user"
CONF_ROW_SIZE: Final[str] = "row_size"
CONF_MIN_SCORE: Final[str] = "min_score"
CONF_ACTION_TEST: Final[str] = "test_connection"
CONF_ACTION_CLEAR_CACHE: Final[str] = "clear_cache"

DEFAULT_URL: Final[str] = "http://digarr:3000"

# The row's identity. Per-user Discover row preferences (order, hidden) key on the
# row's derived uri, built from instance_id and this item_id.
ROW_ID: Final[str] = "up_next"

# A Discover row is a glance. digarr's own UI remains the full backlog surface.
ROW_ITEM_TARGET: Final[int] = 10
# Over-fetch to absorb artists that resolve to nothing on any streaming provider.
RESOLUTION_BUFFER: Final[int] = 15

# Matches lastfm_recommendations: enough parallelism to be quick, little enough
# to avoid hammering streaming provider search endpoints.
SEARCH_CONCURRENCY_LIMIT: Final[int] = 5
# Work around providers that misbehave at limit=1.
PROVIDER_SEARCH_LIMIT: Final[int] = 2
# Number of library search hits scanned when checking for an existing copy.
LIBRARY_MATCH_SCAN_LIMIT: Final[int] = 5

CACHE_CATEGORY_RESOLVED_ITEMS: Final[int] = 1
CACHE_EXPIRATION_SECONDS: Final[int] = 60 * 60 * 24 * 90  # 90 days

REFRESH_TASK_ID: Final[str] = "digarr_refresh"

# The frontend matches this exact string in HomeWidgetRows.vue. Shared vocabulary,
# not ours to rename.
EVENT_RECOMMENDATIONS_UPDATED: Final[str] = "recommendations_updated"
