"""
Tests for the per-user listen-later shelf.

The shelf used to be one boolean on the shared album row, so every account in a
household saw one pile. It is now the `album_listen_later` association table, keyed
on (item_id, userid).

Every test here is written to fail against the old global implementation -- a shelf
test with a single user, or over an empty shelf, passes just as happily against a
household-wide bit or against a backfill that does nothing at all.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest
from music_assistant_models.auth import UserRole
from music_assistant_models.enums import AlbumType
from music_assistant_models.errors import InsufficientPermissions
from music_assistant_models.media_items import Album, AudioFormat, ProviderMapping

from music_assistant.constants import (
    CONF_LISTEN_LATER_BACKFILLED,
    DB_TABLE_ALBUM_LISTEN_LATER,
    HOMEASSISTANT_SYSTEM_USER,
)
from music_assistant.controllers.music.listen_later_backfill import backfill_listen_later
from music_assistant.controllers.webserver.helpers.auth_middleware import set_current_user

if TYPE_CHECKING:
    from music_assistant_models.auth import User

    from music_assistant.mass import MusicAssistant


@pytest.fixture(autouse=True)
async def _no_ambient_user() -> AsyncGenerator[None]:
    """Leave no signed-in user leaking between tests in this module."""
    yield
    set_current_user(None)


async def _add_album(mass: MusicAssistant, name: str) -> int:
    """
    Add a listen-later-shaped album to the library and return its database id.

    `in_library=False` on the mapping, because a listen-later row is exactly an album
    that is *not* in the library proper -- an in_library mapping would trip the mutual
    exclusivity rule and take the album straight back off the shelf.
    """
    item_id = uuid4().hex
    album = await mass.music.albums.add_item_to_library(
        Album(
            item_id="0",
            provider="library",
            name=name,
            album_type=AlbumType.ALBUM,
            provider_mappings={
                ProviderMapping(
                    item_id=item_id,
                    provider_domain="prov_a",
                    provider_instance="prov_a_inst",
                    audio_format=AudioFormat(),
                    in_library=False,
                )
            },
        )
    )
    return int(album.item_id)


async def _make_user(mass: MusicAssistant, username: str, **overrides: Any) -> User:
    """Create a user, optionally rewriting stored columns (e.g. created_at) after."""
    user = await mass.webserver.auth.create_user(username=username, role=UserRole.USER)
    if overrides:
        await mass.webserver.auth.database.update("users", {"user_id": user.user_id}, overrides)
        await mass.webserver.auth.database.commit()
    return user


async def _shelf_names(mass: MusicAssistant, order_by: str = "sort_name") -> list[str]:
    """Return the names of the albums on the calling user's shelf, in query order."""
    items = await mass.music.albums.library_items(listen_later=True, order_by=order_by, limit=0)
    return [item.name for item in items]


async def _set_legacy_shelf(mass: MusicAssistant, db_id: int, added_at: int) -> None:
    """Put an album on the *old* household-wide shelf, as a pre-61 database holds it."""
    await mass.music.database.execute_write(
        "UPDATE albums SET listen_later = 1, listen_later_added_at = :added_at "
        "WHERE item_id = :item_id",
        {"item_id": db_id, "added_at": added_at},
    )


async def _shelf_rows(mass: MusicAssistant) -> list[dict[str, Any]]:
    """Return every association row, as plain dicts."""
    return [
        dict(row)
        for row in await mass.music.database.get_rows_from_query(
            f"SELECT * FROM {DB_TABLE_ALBUM_LISTEN_LATER} ORDER BY item_id, userid", limit=0
        )
    ]


# --------------------------------------------------------------------------------------
# 1. two users' shelves are genuinely independent
# --------------------------------------------------------------------------------------


async def test_two_users_shelves_are_independent(mass: MusicAssistant) -> None:
    """
    Each account sees only its own saves, in listings, counts and the item flag.

    The bite: with one user, a per-user shelf and a household-wide bit are
    indistinguishable -- both return "the album I just saved". So there are two users
    here and each saves a *different* album. Under the old `albums.listen_later`
    column both would list both albums and count 2; the assertions below pin 1 each,
    and pin that each user sees the *other's* album as not saved.
    """
    alice = await _make_user(mass, "alice")
    bob = await _make_user(mass, "bob")
    alice_pick = await _add_album(mass, "Alice Pick")
    bob_pick = await _add_album(mass, "Bob Pick")

    set_current_user(alice)
    await mass.music.albums.set_listen_later(alice_pick, True)
    set_current_user(bob)
    await mass.music.albums.set_listen_later(bob_pick, True)

    set_current_user(alice)
    assert await _shelf_names(mass) == ["Alice Pick"]
    assert await mass.music.albums.library_count(listen_later=True) == 1
    assert (await mass.music.albums.get_library_item(alice_pick)).listen_later is True
    # the other account's save must not read as this account's
    assert (await mass.music.albums.get_library_item(bob_pick)).listen_later is False

    set_current_user(bob)
    assert await _shelf_names(mass) == ["Bob Pick"]
    assert await mass.music.albums.library_count(listen_later=True) == 1
    assert (await mass.music.albums.get_library_item(bob_pick)).listen_later is True
    assert (await mass.music.albums.get_library_item(alice_pick)).listen_later is False


async def test_summary_listing_is_per_user_too(mass: MusicAssistant) -> None:
    """
    The slim summary listing carries the caller's shelf flag, not the household's.

    List views default to summary=True and build their rows field-by-field from a
    different SELECT than the full query, so the per-user override has to be applied
    twice. A test that only exercised the full query would miss a summary_query still
    reading `albums.listen_later`.
    """
    alice = await _make_user(mass, "alice")
    bob = await _make_user(mass, "bob")
    saved = await _add_album(mass, "Saved By Alice")

    set_current_user(alice)
    await mass.music.albums.set_listen_later(saved, True)

    async def _summary_flag(item_id: int) -> bool:
        # in_library_only=False: a listen-later row deliberately carries no in_library
        # mapping, so the default library listing would not return it at all and the
        # flag could not be compared between the two accounts.
        rows = await mass.music.albums.get_library_items_by_query(
            summary=True,
            in_library_only=False,
            extra_query_parts=["albums.item_id = :wanted"],
            extra_query_params={"wanted": item_id},
        )
        assert len(rows) == 1
        return bool(rows[0].listen_later)

    set_current_user(bob)
    assert await _summary_flag(saved) is False
    assert await mass.music.albums.library_items(listen_later=True, summary=True, limit=0) == []

    set_current_user(alice)
    assert await _summary_flag(saved) is True
    alice_shelf = await mass.music.albums.library_items(listen_later=True, summary=True, limit=0)
    assert [item.name for item in alice_shelf] == ["Saved By Alice"]


async def test_the_legacy_column_never_wins_a_read(mass: MusicAssistant) -> None:
    """
    A leftover `albums.listen_later = 1` does not put the album on anybody's shelf.

    This is the duplicate-column trap. The base query still selects `albums.*`, which
    carries the retired columns, so the per-user override shares their names. sqlite3
    Row resolves a duplicate name to the *first* column, which is why the computed
    columns are emitted ahead of `albums.*` -- appended after, they would be silently
    ignored and every read would quietly fall back to the household-wide bit. Nothing
    else in the suite would notice, because the backfill makes the two agree.
    """
    alice = await _make_user(mass, "alice")
    stale = await _add_album(mass, "Stale Global Save")
    await _set_legacy_shelf(mass, stale, added_at=1_000)

    set_current_user(alice)
    assert (await mass.music.albums.get_library_item(stale)).listen_later is False
    assert await _shelf_names(mass) == []
    assert await mass.music.albums.library_count(listen_later=True) == 0

    # ...and the inverse: an association with the legacy column still 0 does count
    fresh = await _add_album(mass, "Fresh Per User Save")
    await mass.music.albums.set_listen_later(fresh, True)
    assert (await mass.music.albums.get_library_item(fresh)).listen_later is True
    assert await _shelf_names(mass) == ["Fresh Per User Save"]


async def test_unsaving_is_scoped_to_the_calling_user(mass: MusicAssistant) -> None:
    """One account clearing its save leaves the other account's save alone."""
    alice = await _make_user(mass, "alice")
    bob = await _make_user(mass, "bob")
    shared = await _add_album(mass, "Both Saved This")

    set_current_user(alice)
    await mass.music.albums.set_listen_later(shared, True)
    set_current_user(bob)
    await mass.music.albums.set_listen_later(shared, True)

    set_current_user(alice)
    await mass.music.albums.set_listen_later(shared, False)
    assert await _shelf_names(mass) == []

    set_current_user(bob)
    assert await _shelf_names(mass) == ["Both Saved This"]


async def test_remove_from_listen_later_keeps_a_row_another_user_still_holds(
    mass: MusicAssistant,
) -> None:
    """
    Clearing your own save must not delete an album somebody else has saved.

    `remove_album_from_listen_later` garbage-collects a row that has no anchor left
    (not in the library, not favorited, never played). Another account's shelf is such
    an anchor -- without that check, the first member of a household to tidy their own
    shelf would delete the album out from under everyone else.
    """
    alice = await _make_user(mass, "alice")
    bob = await _make_user(mass, "bob")
    shared = await _add_album(mass, "Anchored By Bob")

    set_current_user(alice)
    await mass.music.albums.set_listen_later(shared, True)
    set_current_user(bob)
    await mass.music.albums.set_listen_later(shared, True)

    set_current_user(alice)
    await mass.music.remove_album_from_listen_later(shared)

    set_current_user(bob)
    assert await _shelf_names(mass) == ["Anchored By Bob"]


async def test_saving_without_a_signed_in_user_is_refused(mass: MusicAssistant) -> None:
    """A save has to be attributable; there is no household shelf to fall back to."""
    album = await _add_album(mass, "Nobody's Album")
    set_current_user(None)
    with pytest.raises(InsufficientPermissions):
        await mass.music.albums.set_listen_later(album, True)


async def test_an_unauthenticated_listing_sees_an_empty_shelf(mass: MusicAssistant) -> None:
    """
    With no calling user the shelf reads empty, never as everyone's saves merged.

    Fails closed on purpose: a background task reading the shelf must not be handed
    the union of the household's saves.
    """
    alice = await _make_user(mass, "alice")
    album = await _add_album(mass, "Alice Only")
    set_current_user(alice)
    await mass.music.albums.set_listen_later(album, True)

    set_current_user(None)
    assert await _shelf_names(mass) == []
    assert await mass.music.albums.library_count(listen_later=True) == 0


# --------------------------------------------------------------------------------------
# 2. the sort key follows the association, not the retired album column
# --------------------------------------------------------------------------------------


async def test_shelf_sorts_on_the_association_timestamp(mass: MusicAssistant) -> None:
    """
    The shelf orders by when *this user* saved each album.

    The bite: `albums.listen_later_added_at` still exists and still holds a plausible
    timestamp, so a sort key left pointing at it returns a sensibly-ordered shelf and
    looks almost right. So the two orders are deliberately made opposite here -- the
    legacy column says Older-Legacy is the newer save, the association says Newer-
    Legacy is. Sorting on the column yields the exact reverse of both assertions.
    """
    alice = await _make_user(mass, "alice")
    first = await _add_album(mass, "Older Legacy")
    second = await _add_album(mass, "Newer Legacy")
    # legacy stamps: `first` looks like the most recent save
    await _set_legacy_shelf(mass, first, added_at=9_000)
    await _set_legacy_shelf(mass, second, added_at=1_000)

    set_current_user(alice)
    # association stamps, the other way round: alice saved `second` most recently
    await mass.music.database.execute_write(
        f"INSERT INTO {DB_TABLE_ALBUM_LISTEN_LATER} (item_id, userid, added_at) "
        "VALUES (:first, :uid, 100), (:second, :uid, 200)",
        {"first": first, "second": second, "uid": alice.user_id},
    )

    assert await _shelf_names(mass, order_by="listen_later_added_at_desc") == [
        "Newer Legacy",
        "Older Legacy",
    ]
    assert await _shelf_names(mass, order_by="listen_later_added_at") == [
        "Older Legacy",
        "Newer Legacy",
    ]


async def test_each_user_sorts_by_their_own_save_time(mass: MusicAssistant) -> None:
    """
    Two accounts that saved the same two albums in opposite orders each get their own.

    A single-user sort test cannot tell a per-user timestamp from a shared one; this
    one gives the same pair of albums opposite orders per account, which only a
    per-user timestamp can satisfy.
    """
    alice = await _make_user(mass, "alice")
    bob = await _make_user(mass, "bob")
    one = await _add_album(mass, "Album One")
    two = await _add_album(mass, "Album Two")

    await mass.music.database.execute_write(
        f"INSERT INTO {DB_TABLE_ALBUM_LISTEN_LATER} (item_id, userid, added_at) VALUES "
        "(:one, :alice, 100), (:two, :alice, 200), "
        "(:one, :bob, 200), (:two, :bob, 100)",
        {"one": one, "two": two, "alice": alice.user_id, "bob": bob.user_id},
    )

    set_current_user(alice)
    assert await _shelf_names(mass, order_by="listen_later_added_at_desc") == [
        "Album Two",
        "Album One",
    ]
    set_current_user(bob)
    assert await _shelf_names(mass, order_by="listen_later_added_at_desc") == [
        "Album One",
        "Album Two",
    ]


# --------------------------------------------------------------------------------------
# 3. the backfill attributes the pre-existing shelf to the oldest account
# --------------------------------------------------------------------------------------


async def _arm_backfill(mass: MusicAssistant) -> None:
    """Clear the run-once flag the booted fixture already set."""
    mass.config.set(CONF_LISTEN_LATER_BACKFILLED, False, immediate=True)


async def test_backfill_attributes_the_shelf_to_the_oldest_account(
    mass: MusicAssistant,
) -> None:
    """
    The rows of the old household shelf land on the oldest personal account.

    Two bites at once. First, the shelf is *not empty*: a backfill that does nothing
    passes an empty-shelf test trivially, so there are two legacy rows here and both
    must arrive, with their original timestamps. Second, the oldest account is not the
    first row: `bob` is created first and therefore sorts first by insertion (and by
    rowid), but `alice`'s created_at is rewritten to be older -- so a backfill that
    takes whatever row the users table hands back first attributes the shelf to bob
    and fails.
    """
    bob = await _make_user(mass, "bob", created_at="2026-06-01T00:00:00+00:00")
    alice = await _make_user(mass, "alice", created_at="2020-01-01T00:00:00+00:00")
    first = await _add_album(mass, "Legacy One")
    second = await _add_album(mass, "Legacy Two")
    await _set_legacy_shelf(mass, first, added_at=111)
    await _set_legacy_shelf(mass, second, added_at=222)

    await _arm_backfill(mass)
    await backfill_listen_later(mass)

    assert await _shelf_rows(mass) == [
        {"item_id": first, "userid": alice.user_id, "added_at": 111},
        {"item_id": second, "userid": alice.user_id, "added_at": 222},
    ]
    assert mass.config.get(CONF_LISTEN_LATER_BACKFILLED, False) is True

    # and it is actually visible as alice's shelf, not merely present as rows
    set_current_user(alice)
    assert sorted(await _shelf_names(mass)) == ["Legacy One", "Legacy Two"]
    set_current_user(bob)
    assert await _shelf_names(mass) == []


async def test_backfill_ignores_non_personal_accounts(mass: MusicAssistant) -> None:
    """
    A service or guest account never inherits the shelf, however old it is.

    The Home Assistant system user is created during onboarding and so is frequently
    the *oldest* row in the table. Attributing a household's saves to it would hide
    them behind an account nobody signs into -- indistinguishable, from the user's
    side, from having lost them.
    """
    await _make_user(mass, HOMEASSISTANT_SYSTEM_USER, created_at="2019-01-01T00:00:00+00:00")
    await mass.webserver.auth.create_user(username="guest-user", role=UserRole.GUEST)
    await mass.webserver.auth.database.update(
        "users", {"username": "guest-user"}, {"created_at": "2019-06-01T00:00:00+00:00"}
    )
    alice = await _make_user(mass, "alice", created_at="2021-01-01T00:00:00+00:00")
    album = await _add_album(mass, "Legacy One")
    await _set_legacy_shelf(mass, album, added_at=111)

    await _arm_backfill(mass)
    await backfill_listen_later(mass)

    assert await _shelf_rows(mass) == [{"item_id": album, "userid": alice.user_id, "added_at": 111}]


async def test_backfill_defers_when_there_is_no_account_to_attribute_to(
    mass: MusicAssistant,
) -> None:
    """
    With no personal account the shelf is left alone and the backfill retries.

    Giving up permanently here would silently discard the shelf of an install that
    upgrades before it finishes onboarding. The legacy columns are the only record of
    it, so they must survive, and the flag must stay unset so the next boot -- by
    which time an account exists -- picks the work back up.
    """
    await _make_user(mass, HOMEASSISTANT_SYSTEM_USER)
    album = await _add_album(mass, "Legacy One")
    await _set_legacy_shelf(mass, album, added_at=111)

    await _arm_backfill(mass)
    await backfill_listen_later(mass)

    assert await _shelf_rows(mass) == []
    assert mass.config.get(CONF_LISTEN_LATER_BACKFILLED, False) is False
    # the legacy record is intact, so a later run can still do the work
    rows = await mass.music.database.get_rows_from_query(
        "SELECT item_id FROM albums WHERE listen_later = 1", limit=0
    )
    assert [row["item_id"] for row in rows] == [album]

    # ...and it does, once somebody signs up
    alice = await _make_user(mass, "alice")
    await backfill_listen_later(mass)
    assert await _shelf_rows(mass) == [{"item_id": album, "userid": alice.user_id, "added_at": 111}]


async def test_backfill_does_not_resurrect_a_save_the_user_has_since_cleared(
    mass: MusicAssistant,
) -> None:
    """
    Re-running the backfill is a no-op, including after the user has tidied up.

    The legacy columns are deliberately kept, so the input of the conversion is still
    sitting there on every later boot. The run-once flag is what stops it being
    replayed -- without it, an album the user took off their shelf would come back at
    the next restart, forever.
    """
    alice = await _make_user(mass, "alice")
    album = await _add_album(mass, "Legacy One")
    await _set_legacy_shelf(mass, album, added_at=111)

    await _arm_backfill(mass)
    await backfill_listen_later(mass)
    assert len(await _shelf_rows(mass)) == 1

    set_current_user(alice)
    await mass.music.albums.set_listen_later(album, False)
    assert await _shelf_rows(mass) == []

    await backfill_listen_later(mass)
    assert await _shelf_rows(mass) == []


async def test_backfill_is_idempotent_when_replayed(mass: MusicAssistant) -> None:
    """An interrupted run that is retried does not duplicate or overwrite rows."""
    alice = await _make_user(mass, "alice")
    album = await _add_album(mass, "Legacy One")
    await _set_legacy_shelf(mass, album, added_at=111)

    await _arm_backfill(mass)
    await backfill_listen_later(mass)
    await _arm_backfill(mass)  # simulate a crash before the flag was persisted
    await backfill_listen_later(mass)

    assert await _shelf_rows(mass) == [{"item_id": album, "userid": alice.user_id, "added_at": 111}]


async def test_backfill_never_raises_when_the_auth_database_is_unreadable(
    mass: MusicAssistant,
) -> None:
    """
    A failure to read the accounts aborts the attribution, not the boot.

    This runs during startup; an escape here would leave the server dead. The flag
    must stay unset so the next boot retries.
    """
    album = await _add_album(mass, "Legacy One")
    await _set_legacy_shelf(mass, album, added_at=111)
    await _arm_backfill(mass)

    original = mass.webserver.auth.database
    mass.webserver.auth.database = None  # type: ignore[assignment]
    try:
        await backfill_listen_later(mass)
    finally:
        mass.webserver.auth.database = original

    assert await _shelf_rows(mass) == []
    assert mass.config.get(CONF_LISTEN_LATER_BACKFILLED, False) is False
