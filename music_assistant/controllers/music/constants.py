"""Constants for the music controller."""

from __future__ import annotations

from typing import Final

CONF_RESET_DB = "reset_db"
DEFAULT_SYNC_INTERVAL = 12 * 60  # default sync interval in minutes
CONF_SYNC_INTERVAL = "sync_interval"
CONF_DELETED_PROVIDERS = "deleted_providers"

# 60, not 59: upstream took 59 for the playlist `access` column (owner + sharing),
# and this branch had already spent 59 on listen_later. A fork database is therefore
# stamped 59 while *lacking* upstream's column, and upstream's own step is gated at
# "prev_version <= 58" so it would never fire for it. Worse, the
# "prev_version not in (0, DB_SCHEMA_VERSION)" guard in database.py would skip
# migration entirely at 59, and every playlist read then fails on the missing
# "access" column. The <= 59 step in migrations.py backfills it.
#
# 61: the listen-later shelf moved off the `albums.listen_later` column and into the
# per-user `album_listen_later` association table. The <= 60 step creates that table
# and stops there: every shelf starts empty, by decision, because the old schema
# records only *that* an album was saved and never *who* saved it, so any attribution
# would be a guess. The retired columns are kept as the record of what was on the
# household shelf -- see the <= 60 step in migrations.py for how to read one back.
#
# 62: upstream took 60 for its own step (dropping the bogus "None" provider mappings)
# and gated it at "prev_version <= 59". This branch was already stamped 61, so that
# gate can never fire for a fork database and the bogus mappings would survive
# forever. The catch-up step in migrations.py is therefore widened to "<= 61" and the
# version moved to 62 so every fork database is re-migrated exactly once.
#
# 63: upstream took 61 for its own step (favorites move from a column on every media
# table to the per-user favorites table) and gated it at "prev_version <= 60". This
# branch was already stamped 62, so that gate can never fire for a fork database. The
# step in migrations.py is widened to "<= 62" and the version moved to 63.
#
# 64: upstream took 62 for its own step (dropping the playlist collages the metadata
# controller drew, and the generated art on the builtin system playlists) and gated it
# at "prev_version <= 61". This branch was already stamped 63, so that gate can never
# fire for a fork database. The step in migrations.py is widened to "<= 63" and the
# version moved to 64.
#
# 65: upstream took 63 for its own step (dropping images with an empty path, which
# resolve to one shared picture per provider) and gated it at "prev_version <= 62".
# This branch was already stamped 64, so that gate can never fire for a fork database.
# The step in migrations.py is widened to "<= 64".
#
# 66: upstream took 64 for its own step (moving the audio analysis tables out of
# library.db into audio_analysis.db) and gated it at "prev_version <= 63". Widened to
# "<= 65" for the same reason, and the version moved to 66.
#
# 67: upstream took 65 for its own step (moving misplaced aliases off the default
# classical genre) and gated it at "prev_version <= 64". Widened to "<= 66" for the
# same reason, and the version moved to 67.
DB_SCHEMA_VERSION: Final[int] = 67

# tracks longer that this will not be included in radio mode
RADIO_TRACK_MAX_DURATION_SECS: Final[int] = 20 * 60
DYNAMIC_RADIO_BASE_SAMPLE_SIZE: Final[int] = 5
DYNAMIC_RADIO_DYNAMIC_TARGET: Final[int] = 50

CACHE_CATEGORY_SEARCH_RESULTS: Final[int] = 10

# max time to wait for a single provider's search results before
# contributing empty results for it, so one slow provider can never block
# the whole search; the provider search itself continues in the background
# so its result is cached and available for a next search request
SEARCH_PROVIDER_SOFT_TIMEOUT: Final[int] = 8
# absolute max time a (background) provider search may run,
# rate limited providers can be very slow to respond
SEARCH_PROVIDER_HARD_TIMEOUT: Final[int] = 120
# how long to cache raw per-provider search results; streaming catalogs barely
# change so they can be cached a lot longer than local providers where the
# user may add or change content at any time
SEARCH_CACHE_EXPIRATION_STREAMING_PROVIDER: Final[int] = 24 * 3600
SEARCH_CACHE_EXPIRATION_LOCAL_PROVIDER: Final[int] = 900
# how long to cache combined search results (fast path for repeated searches)
SEARCH_CACHE_EXPIRATION_COMBINED: Final[int] = 600

# max time to wait for a single recommendation row's item fetch before skipping
# the row, so one slow provider row can never block an items request
RECOMMENDATIONS_ITEMS_TIMEOUT: Final[int] = 30

# max time to wait for a single provider's recommendation rows; rows are
# contractually fast (no live backend calls) so a short timeout suffices
RECOMMENDATIONS_ROWS_TIMEOUT: Final[int] = 5

# Budget for grafting library album metadata onto provider row items. Deliberately
# short and non-fatal: when it expires the row is still served, just without badges.
RECOMMENDATIONS_ENRICH_TIMEOUT: Final[int] = 5

# how long after a provider is loaded its library sync runs for the first time,
# leaving the rest of the startup work room to settle first
INITIAL_SYNC_DELAY: Final[int] = 10

DATABASE_CLEANUP_TASK_ID: Final[str] = "music_database_cleanup"
PROVIDER_MAPPING_CORRECTION_TASK_ID: Final[str] = "music_provider_mapping_correction"
MUSIC_SYNC_COMPLETION_CHECK_TASK_ID: Final[str] = "music_sync_completion_check"
TRACK_RECONCILIATION_TASK_ID: Final[str] = "music_track_reconciliation"

# number of duplicate track candidate pairs examined per reconciliation run
TRACK_RECONCILIATION_BATCH_SIZE: Final[int] = 100

# where the duplicate track walk left off, stored so a restart resumes instead of starting
# over: the pair last examined as [item_id, item_id], or an empty list once the walk has
# reached the end of the library
CONF_TRACK_RECONCILIATION_CURSOR: Final[str] = "track_reconciliation_cursor"
# whether a sync has added content that the walk still owes another pass
CONF_TRACK_RECONCILIATION_RESCAN_DUE: Final[str] = "track_reconciliation_rescan_due"
# max difference in seconds between two track durations to still consider them the same
# recording; matches the widest duration window compare_track is willing to accept
TRACK_RECONCILIATION_MAX_DURATION_DELTA: Final[int] = 8
# max number of library rows that may share one normalized title before the duplicate track
# walk skips that title. Pairing the rows of a title is quadratic in their count, and a title
# held by hundreds of rows is a generic one rather than a duplicate
TRACK_RECONCILIATION_MAX_TITLE_ROWS: Final[int] = 200
# Audio analysis rows are moved out of library.db in batches of this many rows, one
# transaction each.
AUDIO_ANALYSIS_MOVE_BATCH_SIZE: Final[int] = 5000
# Legacy analysis JSON rows are packed while they move, in cursor batches of this size, one
# transaction each; a fully analysed row is ~230 KB of JSON, so a batch is held in memory
# twice (decoded and packed) while it converts. Progress is logged once per this many rows.
AUDIO_ANALYSIS_PACK_BATCH_SIZE: Final[int] = 100
AUDIO_ANALYSIS_PACK_PROGRESS_ROWS: Final[int] = 2000
