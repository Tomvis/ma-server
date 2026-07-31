"""Tests for library database migrations."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from music_assistant.constants import (
    DB_TABLE_ALBUMS,
    DB_TABLE_AUDIO_ANALYSIS,
    DB_TABLE_PLAYLOG,
    DB_TABLE_PROVIDER_MAPPINGS,
)
from music_assistant.controllers.music import MusicController
from music_assistant.helpers.database import DatabaseConnection

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from music_assistant.mass import MusicAssistant


@pytest.fixture
async def database(tmp_path: Path) -> AsyncGenerator[DatabaseConnection]:
    """Return an initialized DatabaseConnection backed by a temp file."""
    db = DatabaseConnection(str(tmp_path / "library.db"))
    await db.setup()
    yield db
    await db.close()


# the exact upsert used by MusicController._credit_artist_plays - it targets the
# 4-column unique constraint, so it raises IntegrityError on databases that still
# carry the legacy 3-column constraint (issue #5754)
PLAYLOG_UPSERT = (
    f"INSERT INTO {DB_TABLE_PLAYLOG} "
    "(item_id, provider, media_type, name, image, fully_played, "
    "seconds_played, timestamp, queue_id, user_initiated, userid) "
    "VALUES (:item_id, :provider, :media_type, :name, :image, :fully_played, "
    ":seconds_played, :timestamp, :queue_id, :user_initiated, :userid) "
    "ON CONFLICT(item_id, provider, media_type, userid) DO UPDATE SET "
    "timestamp = excluded.timestamp"
)


def _playlog_entry(userid: str, timestamp: int = 100) -> dict[str, object]:
    return {
        "item_id": "1",
        "provider": "library",
        "media_type": "track",
        "name": "Test Track",
        "image": None,
        "fully_played": 1,
        "seconds_played": 195,
        "timestamp": timestamp,
        "queue_id": "queue1",
        "user_initiated": 1,
        "userid": userid,
    }


async def _create_legacy_playlog_table(database: DatabaseConnection) -> None:
    """Create the playlog table as it exists on pre-userid installs."""
    # original table layout (schema version <= 22) with the 3-column UNIQUE constraint
    await database.execute(
        f"""CREATE TABLE {DB_TABLE_PLAYLOG}(
            [id] INTEGER PRIMARY KEY AUTOINCREMENT,
            [item_id] TEXT NOT NULL,
            [provider] TEXT NOT NULL,
            [media_type] TEXT NOT NULL,
            [name] TEXT NOT NULL,
            [image] json,
            [timestamp] INTEGER DEFAULT 0,
            [fully_played] BOOLEAN,
            [seconds_played] INTEGER,
            UNIQUE(item_id, provider, media_type));"""
    )
    # columns + index added in-place by the later ALTER TABLE migrations
    await database.execute(f"ALTER TABLE {DB_TABLE_PLAYLOG} ADD COLUMN userid TEXT")
    await database.execute(f"ALTER TABLE {DB_TABLE_PLAYLOG} ADD COLUMN queue_id TEXT")
    await database.execute(
        f"ALTER TABLE {DB_TABLE_PLAYLOG} ADD COLUMN user_initiated BOOLEAN NOT NULL DEFAULT 1"
    )
    await database.execute(
        f"CREATE UNIQUE INDEX {DB_TABLE_PLAYLOG}_unique_idx "
        f"ON {DB_TABLE_PLAYLOG}(item_id,provider,media_type,userid)"
    )
    await database.commit()


async def _run_migration(
    mass: MusicAssistant, database: DatabaseConnection, prev_version: int = 41
) -> None:
    """Run the library db migration against the given database."""
    await mass.cache._setup_database()
    controller = MusicController(mass)
    controller._database = database
    await controller._MusicController__migrate_database(prev_version=prev_version)  # type: ignore[attr-defined]


async def test_migration_rebuilds_playlog_with_stale_unique_constraint(
    mass_minimal: MusicAssistant,
    database: DatabaseConnection,
) -> None:
    """The legacy 3-column UNIQUE constraint on playlog is dropped by a table rebuild."""
    await _create_legacy_playlog_table(database)
    await database.execute(PLAYLOG_UPSERT, _playlog_entry("user1"))
    # a legacy row from before the userid column existed
    await database.execute(
        f"INSERT INTO {DB_TABLE_PLAYLOG} (item_id, provider, media_type, name) "
        "VALUES ('2', 'library', 'track', 'Legacy Track')"
    )
    await database.commit()

    await _run_migration(mass_minimal, database)

    # replaying the same item for the same user updates the existing row in place
    await database.execute(PLAYLOG_UPSERT, _playlog_entry("user1", timestamp=200))
    # another user playing the same item gets their own row
    await database.execute(PLAYLOG_UPSERT, _playlog_entry("user2"))
    rows = await database.get_rows(DB_TABLE_PLAYLOG, {"item_id": "1"})
    assert len(rows) == 2
    user1_row = next(row for row in rows if row["userid"] == "user1")
    assert user1_row["timestamp"] == 200
    assert user1_row["name"] == "Test Track"
    # legacy rows without a userid cannot be kept under the NOT NULL schema
    assert not await database.get_rows(DB_TABLE_PLAYLOG, {"item_id": "2"})


async def test_migration_leaves_correct_playlog_untouched(
    mass_minimal: MusicAssistant,
    database: DatabaseConnection,
) -> None:
    """A playlog table that already has the 4-column constraint is not rebuilt."""
    await database.execute(
        f"""CREATE TABLE {DB_TABLE_PLAYLOG}(
            [id] INTEGER PRIMARY KEY AUTOINCREMENT,
            [item_id] TEXT NOT NULL,
            [provider] TEXT NOT NULL,
            [media_type] TEXT NOT NULL,
            [name] TEXT NOT NULL,
            [image] json,
            [timestamp] INTEGER DEFAULT 0,
            [fully_played] BOOLEAN,
            [seconds_played] INTEGER,
            [userid] TEXT NOT NULL,
            [queue_id] TEXT,
            [user_initiated] BOOLEAN NOT NULL DEFAULT 1,
            UNIQUE(item_id, provider, media_type, userid));"""
    )
    await database.execute(PLAYLOG_UPSERT, _playlog_entry("user1"))
    await database.commit()
    table_sql_query = (
        f"SELECT sql FROM sqlite_master WHERE type = 'table' AND name = '{DB_TABLE_PLAYLOG}'"
    )
    table_sql_before = (await database.get_rows_from_query(table_sql_query))[0]["sql"]

    await _run_migration(mass_minimal, database)

    assert (await database.get_rows_from_query(table_sql_query))[0]["sql"] == table_sql_before
    rows = await database.get_rows(DB_TABLE_PLAYLOG)
    assert len(rows) == 1


async def _create_upstream_album_tables(database: DatabaseConnection) -> None:
    """
    Create albums + provider_mappings exactly as upstream stable ships them.

    Copied verbatim from tag 2.9.10, which stamps DB_SCHEMA_VERSION 43. Neither
    listen_later column exists upstream - they are enhanced-branch additions.
    """
    await database.execute(
        f"""CREATE TABLE IF NOT EXISTS {DB_TABLE_ALBUMS}(
                [item_id] INTEGER PRIMARY KEY AUTOINCREMENT,
                [name] TEXT NOT NULL,
                [sort_name] TEXT NOT NULL,
                [version] TEXT,
                [album_type] TEXT NOT NULL,
                [year] INTEGER,
                [favorite] BOOLEAN NOT NULL DEFAULT 0,
                [metadata] json NOT NULL,
                [external_ids] json NOT NULL,
                [play_count] INTEGER NOT NULL DEFAULT 0,
                [last_played] INTEGER NOT NULL DEFAULT 0,
                [timestamp_added] INTEGER DEFAULT (cast(strftime('%s','now') as int)),
                [timestamp_modified] INTEGER NOT NULL DEFAULT 0,
                [search_name] TEXT NOT NULL,
                [search_sort_name] TEXT NOT NULL
            );"""
    )
    await database.execute(
        f"""CREATE TABLE IF NOT EXISTS {DB_TABLE_PROVIDER_MAPPINGS}(
            [media_type] TEXT NOT NULL,
            [item_id] INTEGER NOT NULL,
            [provider_domain] TEXT NOT NULL,
            [provider_instance] TEXT NOT NULL,
            [provider_item_id] TEXT NOT NULL,
            [available] BOOLEAN NOT NULL DEFAULT 1,
            [in_library] BOOLEAN NOT NULL DEFAULT 0,
            [is_unique] BOOLEAN,
            [url] text,
            [audio_format] json,
            [details] TEXT,
            UNIQUE(media_type, provider_instance, provider_item_id)
            );"""
    )
    await database.execute(
        f"INSERT INTO {DB_TABLE_ALBUMS} "
        "(item_id, name, sort_name, album_type, metadata, external_ids, "
        "search_name, search_sort_name) "
        "VALUES (1, 'Kid A', 'kid a', 'album', '{}', '[]', 'KID A', 'KID A')"
    )
    await database.commit()


@pytest.mark.parametrize("prev_version", [43, 44, 45])
async def test_migration_adds_listen_later_columns_to_upstream_database(
    mass_minimal: MusicAssistant,
    database: DatabaseConnection,
    prev_version: int,
) -> None:
    """
    An upstream-schema database gains the enhanced listen_later columns.

    Upstream stable stamps schema version 43, so a database coming from stock
    Music Assistant never passed through the enhanced `prev_version <= 42` gate
    that adds albums.listen_later / listen_later_added_at. A later enhanced block
    writes to those columns, so if the gate does not cover 43-45 the migration
    raises "no such column: listen_later" - and _setup_database responds to a
    failed migration by deleting library.db and rescanning from scratch.
    """
    await _create_upstream_album_tables(database)

    await _run_migration(mass_minimal, database, prev_version=prev_version)

    columns = {
        row["name"]
        for row in await database.get_rows_from_query(
            f"SELECT name FROM pragma_table_info('{DB_TABLE_ALBUMS}')"
        )
    }
    assert "listen_later" in columns
    assert "listen_later_added_at" in columns
    # the pre-existing row survives and picks up the column default
    rows = await database.get_rows(DB_TABLE_ALBUMS)
    assert len(rows) == 1
    assert rows[0]["name"] == "Kid A"
    assert rows[0]["listen_later"] == 0
    assert rows[0]["listen_later_added_at"] is None


async def test_migration_keeps_listen_later_data_on_enhanced_database(
    mass_minimal: MusicAssistant,
    database: DatabaseConnection,
) -> None:
    """
    Re-running the listen_later ALTERs on a database that already has them is a no-op.

    Widening the gate to <= 46 means enhanced databases (which passed through the
    original <= 42 gate) now re-enter this block, so the duplicate-column tolerance
    has to hold and existing listen_later values must survive untouched.
    """
    await _create_upstream_album_tables(database)
    for column_sql in (
        f"ALTER TABLE {DB_TABLE_ALBUMS} ADD COLUMN [listen_later] BOOLEAN NOT NULL DEFAULT 0;",
        f"ALTER TABLE {DB_TABLE_ALBUMS} ADD COLUMN [listen_later_added_at] INTEGER;",
    ):
        await database.execute(column_sql)
    await database.execute(
        f"UPDATE {DB_TABLE_ALBUMS} SET listen_later = 1, listen_later_added_at = 1700000000"
    )
    await database.commit()

    await _run_migration(mass_minimal, database, prev_version=46)

    rows = await database.get_rows(DB_TABLE_ALBUMS)
    assert len(rows) == 1
    # the album carries no in_library provider mapping, so the mutual-exclusivity
    # cleanup must leave its listen_later flag alone
    assert rows[0]["listen_later"] == 1
    assert rows[0]["listen_later_added_at"] == 1700000000


async def test_migration_repairs_null_smart_fades_centroids(
    mass_minimal: MusicAssistant,
    database: DatabaseConnection,
) -> None:
    """Null spectral centroid values in legacy Smart Fades analysis rows become 0.0."""
    await database.execute(
        f"""CREATE TABLE {DB_TABLE_AUDIO_ANALYSIS}(
            [id] INTEGER PRIMARY KEY AUTOINCREMENT,
            [aa_provider_domain] TEXT NOT NULL,
            [analysis_data] json NOT NULL)"""
    )
    rows = {
        1: ("smart_fades", '{"spectral_centroid": [1.5, null, 2.5, null], "bpm": 120}'),
        2: ("smart_fades", '{"spectral_centroid": [1.0, 2.0], "bpm": 100}'),
        # null centroids from another analysis provider must not be touched
        3: ("other_domain", '{"spectral_centroid": [null], "bpm": 100}'),
        # a corrupt payload must not abort the migration
        4: ("smart_fades", '{"spectral_centroid": [null'),
        # a non-array centroid value must not be touched
        5: ("smart_fades", '{"spectral_centroid": null, "bpm": 90}'),
        # "null" appearing only inside a string value must not trigger a rewrite
        6: ("smart_fades", '{"spectral_centroid": [3.5], "key": "nullish"}'),
    }
    for row_id, (domain, analysis_data) in rows.items():
        await database.execute(
            f"INSERT INTO {DB_TABLE_AUDIO_ANALYSIS} (id, aa_provider_domain, analysis_data) "
            "VALUES (:id, :domain, :analysis_data)",
            {"id": row_id, "domain": domain, "analysis_data": analysis_data},
        )
    await database.commit()

    await _run_migration(mass_minimal, database, prev_version=42)

    repaired = {
        row["id"]: row["analysis_data"] for row in await database.get_rows(DB_TABLE_AUDIO_ANALYSIS)
    }
    assert json.loads(repaired[1]) == {"spectral_centroid": [1.5, 0.0, 2.5, 0.0], "bpm": 120}
    # untouched rows must not be rewritten at all, hence the exact-string compare
    for untouched_id in (2, 3, 4, 5, 6):
        assert repaired[untouched_id] == rows[untouched_id][1]
