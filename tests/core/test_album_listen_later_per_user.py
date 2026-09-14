"""
Tests for the per-user listen-later shelf.

The shelf used to be one boolean on the shared album row, so every account in a
household saw one pile. It is now the `album_listen_later` association table, keyed
on (item_id, userid).

Nothing attributes the old household shelf to anyone: the pre-61 schema recorded only
*that* an album was saved, never by whom, so every shelf starts empty and the retired
`albums.listen_later` columns are kept purely as the record of what it held.

Every test here is written to fail against the old global implementation. A shelf test
with a single user cannot distinguish a per-user shelf from a household-wide bit at
all -- both hand back "the album I just saved" -- so each one below uses two accounts
that want different things.
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
    DB_TABLE_ALBUM_LISTEN_LATER,
)
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


async def _make_user(mass: MusicAssistant, username: str) -> User:
    """Create a signed-up user on the instance."""
    return await mass.webserver.auth.create_user(username=username, role=UserRole.USER)


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
    else in the suite would notice: nothing writes the legacy columns any more, so on a
    database built by the current code the two always agree and a read falling through
    to the wrong one is invisible. This test is the only place they disagree.

    It doubles as the pin on "every shelf starts empty at 61": a database arriving with
    listen_later = 1 rows puts them on nobody's shelf.
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

    One wrinkle worth knowing before reading this as a weaker test than it is: the
    sort key that bites is the *qualified* `albums.listen_later_added_at`. The bare
    `listen_later_added_at` the old key used now resolves to the SELECT alias, which
    is already the per-user subquery, so it is not a second behaviour this test could
    distinguish -- it is the same one. Reordering the SELECT so the alias lands on the
    legacy column breaks this too, and is pinned by
    test_the_legacy_column_never_wins_a_read.
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
