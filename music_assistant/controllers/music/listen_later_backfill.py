"""
One-shot attribution of the pre-per-user listen-later shelf to its owning account.

The shelf used to be a single bit on the shared album row (`albums.listen_later`),
so every account in a household saw one pile. Schema 61 moved it into the per-user
`album_listen_later` association table; the migration step creates that table but
cannot fill it, because the accounts live in `auth.db` and that database is not open
when the library migration runs (`music.setup()` is in the core-controller TaskGroup,
`webserver.setup()` comes after it).

So the attribution runs from `MusicAssistant.start()` once the webserver is up, the
same shape and for the same reason as `provider_access_migration.py`.

Unlike that one it may also legitimately find *nothing to attribute to*: an install
can carry a shelf while having no human account yet (the pre-auth single-user mode,
or one where only the Home Assistant service user exists). In that case it leaves the
legacy columns alone and does not set its flag, so the next boot tries again -- the
same "wait for a real account" shape as `_migrate_playlog_to_first_user`.

TODO: remove once no install can still be on schema <= 60
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from music_assistant_models.auth import UserRole

from music_assistant.constants import (
    CONF_LISTEN_LATER_BACKFILLED,
    DB_TABLE_ALBUM_LISTEN_LATER,
    DB_TABLE_ALBUMS,
    HOMEASSISTANT_SYSTEM_USER,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from music_assistant.mass import MusicAssistant

LOGGER = logging.getLogger(__name__)

# roles that are not a person, and so can never be the owner of a saved shelf.
# GUEST accounts are ephemeral and SERVICE accounts are integrations (Home Assistant,
# the MCP server); attributing a household's saves to either loses them just as surely
# as attributing them to nobody, but less visibly.
_NON_PERSONAL_ROLES = frozenset({UserRole.GUEST, UserRole.SERVICE})


async def backfill_listen_later(mass: MusicAssistant) -> None:
    """
    Give the rows of the old household-wide listen-later shelf to the oldest account.

    Runs at most once per install and never raises: an install where the attribution
    fails, or that has no account to attribute to yet, is retried on the next startup.

    :param mass: The MusicAssistant instance, with its webserver set up.
    """
    if mass.config.get(CONF_LISTEN_LATER_BACKFILLED, False):
        return
    try:
        owner = await _owning_user_id(mass)
        if owner is None:
            # Nobody to give the shelf to. Leave the legacy columns untouched and the
            # flag unset so the next boot retries: on an install that has not created
            # its first account yet, the shelf is still recoverable, and guessing an
            # owner is not. An empty per-user shelf is fixed by re-saving; a shelf
            # handed to the wrong person is not obviously wrong to anyone.
            LOGGER.debug(
                "Listen-later shelf not attributed yet: this install has no personal "
                "account. Retrying on the next startup."
            )
            return
        moved = await _attribute_shelf(mass, owner)
    except Exception:
        # an escape here would abort the boot. The flag stays unset, so the next
        # startup retries the (idempotent) attribution
        LOGGER.exception("Unable to attribute the listen-later shelf of this install")
        return
    if moved:
        LOGGER.info("Attributed %d listen-later album(s) to user %s", moved, owner)
    mass.config.set(CONF_LISTEN_LATER_BACKFILLED, True, immediate=True)


async def _owning_user_id(mass: MusicAssistant) -> str | None:
    """
    Return the account the pre-existing shelf belongs to, or None if there is none.

    The oldest personal account, which on the single-user install this shelf was
    designed for is the only one. Sorted on the parsed `created_at` rather than in
    SQL: the column is TEXT holding an ISO-8601 timestamp, so an ORDER BY is a string
    sort that only happens to be right while every row carries the same UTC offset.

    :param mass: The MusicAssistant instance to read the accounts of.
    """
    # limit=0 disables the LIMIT clause -- the default of 500 would silently truncate
    rows = await mass.webserver.auth.database.get_rows("users", limit=0)
    personal = [row for row in rows if _is_personal_account(row)]
    if not personal:
        return None
    personal.sort(key=_created_at)
    return str(personal[0]["user_id"])


def _is_personal_account(row: Mapping[str, Any]) -> bool:
    """
    Return whether the given user row is a person who could own a shelf.

    Indexed rather than .get(): the rows arrive as sqlite3.Row, which the codebase
    types as a Mapping but which implements only __getitem__ and keys(). A missing
    column therefore raises rather than defaulting, so it is caught here -- one
    unreadable row must not abort the attribution for everyone else.
    """
    try:
        return bool(
            row["user_id"]
            and row["role"] not in _NON_PERSONAL_ROLES
            and row["username"] != HOMEASSISTANT_SYSTEM_USER
        )
    except KeyError, IndexError, TypeError:
        return False


def _created_at(row: Mapping[str, Any]) -> datetime:
    """
    Return the creation time of a user row, for ordering.

    A row whose timestamp is missing or unparsable sorts last rather than raising:
    it must not be able to win the "oldest account" election on a technicality, and
    it must not be able to abort the attribution for everyone else either.
    """
    try:
        return datetime.fromisoformat(str(row["created_at"]))
    except KeyError, IndexError, TypeError, ValueError:
        return datetime.max.replace(tzinfo=None)


async def _attribute_shelf(mass: MusicAssistant, owner: str) -> int:
    """
    Copy every row of the legacy shelf onto the given user's shelf.

    :param mass: The MusicAssistant instance to read the library database of.
    :param owner: The user id to attribute the shelf to.
    :return: The number of albums attributed.
    """
    database = mass.music.database
    # INSERT OR IGNORE, so a retry after a partial write (or after the user has
    # already re-saved an album by hand) is a no-op rather than a conflict. The
    # legacy columns stay as they are: they are the input of this conversion, and
    # keeping them means a retry still has something to read.
    await database.execute_write(
        f"""INSERT OR IGNORE INTO {DB_TABLE_ALBUM_LISTEN_LATER} (item_id, userid, added_at)
            SELECT item_id, :owner, listen_later_added_at
            FROM {DB_TABLE_ALBUMS} WHERE listen_later = 1""",
        {"owner": owner},
    )
    return await database.get_count_from_query(
        f"SELECT item_id FROM {DB_TABLE_ALBUM_LISTEN_LATER} WHERE userid = :owner",
        {"owner": owner},
    )
