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
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from music_assistant_models.auth import UserRole
from music_assistant_models.config_entries import ProviderConfig
from music_assistant_models.enums import (
    AlbumType,
    MediaType,
    ProviderFeature,
    ProviderType,
)
from music_assistant_models.errors import InsufficientPermissions
from music_assistant_models.media_items import Album, AudioFormat, ProviderMapping
from music_assistant_models.provider import ProviderManifest

from music_assistant.constants import (
    DB_TABLE_ALBUM_LISTEN_LATER,
)
from music_assistant.controllers.webserver.helpers.auth_middleware import set_current_user
from music_assistant.models.music_provider import (
    CACHE_CATEGORY_PREV_LIBRARY_IDS,
    MusicProvider,
)

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


# --------------------------------------------------------------------------------------
# 3. the household-wide reads, which have no calling user and must not be scoped to one
# --------------------------------------------------------------------------------------


class _DroppedEverythingProvider(MusicProvider):
    """A non-streaming music provider whose library has gone empty."""

    @property
    def supported_features(self) -> set[ProviderFeature]:
        return {ProviderFeature.LIBRARY_ALBUMS}

    @property
    def is_streaming_provider(self) -> bool:
        # non-streaming, so a vanished item is a real deletion candidate rather than
        # something the provider merely un-starred
        return False

    def library_supported(self, media_type: MediaType) -> bool:
        return bool(media_type == MediaType.ALBUM)

    # the provider's library, which this test always leaves empty; typed as a plain
    # tuple so the generator below stays a generator without an unreachable `yield`
    catalog: tuple[Album, ...] = ()

    async def get_library_albums(self) -> AsyncGenerator[Album]:
        """Yield the provider's library, which is empty: everything it had is gone."""
        for album in self.catalog:
            yield album


def _register_dropping_provider(mass: MusicAssistant) -> _DroppedEverythingProvider:
    """Register a provider that reports an empty library on its next sync."""
    manifest = ProviderManifest(
        type=ProviderType.MUSIC,
        domain="dropper",
        name="Dropper",
        description="Provider whose library went empty",
        codeowners=["@music-assistant"],
    )
    config = ProviderConfig(
        values={},
        type=ProviderType.MUSIC,
        domain="dropper",
        instance_id="dropper--instance",
        name="Dropper",
    )
    # "GLOBAL" keeps the base class's log-level resolution valid; every boolean sync
    # option this path reads (library_sync_deletions in particular) is truthy from it
    config.get_value = lambda *_a, **_k: "GLOBAL"
    provider = _DroppedEverythingProvider(mass, manifest, config)
    provider.available = True
    mass._providers[provider.instance_id] = provider
    return provider


async def test_provider_deletion_keeps_an_album_another_user_has_saved(
    mass: MusicAssistant,
) -> None:
    """
    A provider dropping an album must not delete a row somebody has on their shelf.

    This is the data-loss path. The sync loop runs as a background task with no calling
    user, so an anchor check that reads the *item's* listen_later sees False no matter
    who has the album saved, and the row is deleted out from under them -- silently, as
    a side effect of an unrelated provider going quiet.

    Driven through the real `sync_library` rather than a re-implementation of its
    logic in the test body, so reverting the fix actually fails it.
    """
    alice = await _make_user(mass, "alice")
    provider = _register_dropping_provider(mass)
    try:
        album = await mass.music.albums.add_item_to_library(
            Album(
                item_id="0",
                provider="library",
                name="Dropped But Saved",
                album_type=AlbumType.ALBUM,
                provider_mappings={
                    ProviderMapping(
                        item_id="gone-1",
                        provider_domain="dropper",
                        provider_instance=provider.instance_id,
                        audio_format=AudioFormat(),
                        in_library=True,
                    )
                },
            )
        )
        db_id = int(album.item_id)
        # alice saves it; nothing else anchors the row -- not favorited, never played,
        # and after the sync no provider will have it in library either
        set_current_user(alice)
        await mass.music.albums.set_listen_later(db_id, True)
        set_current_user(None)

        # the previous sync saw this album; this one will not
        await mass.cache.set(
            key=MediaType.ALBUM.value,
            data=[db_id],
            provider=provider.instance_id,
            category=CACHE_CATEGORY_PREV_LIBRARY_IDS,
        )
        await provider.sync_library(MediaType.ALBUM)

        # the row survives...
        survivor = await mass.music.albums.get_library_item(db_id)
        assert survivor.name == "Dropped But Saved"
        # ...demoted out of the library proper, which is the whole point of keeping it
        assert not any(pm.in_library for pm in survivor.provider_mappings)
        # ...and it is still on alice's shelf, not merely present as a row
        set_current_user(alice)
        assert await _shelf_names(mass) == ["Dropped But Saved"]
    finally:
        mass._providers.pop(provider.instance_id, None)


async def test_entering_the_library_clears_every_users_shelf(
    mass: MusicAssistant,
) -> None:
    """
    An album that joins the library proper leaves everybody's shelf, not just the caller's.

    Library membership and listen-later are mutually exclusive. The clear used to be
    gated on the item's own listen_later, which is now the *calling* user's bit -- so an
    album added by (or synced for) one account would be cleared from that account's
    shelf and left on every other account's, sitting in both views at once, which is the
    state the rule exists to prevent. The sync loop that most often triggers it has no
    calling user at all, in which case it would clear nobody's.
    """
    alice = await _make_user(mass, "alice")
    bob = await _make_user(mass, "bob")
    db_id = await _add_album(mass, "Graduates To Library")

    set_current_user(alice)
    await mass.music.albums.set_listen_later(db_id, True)
    set_current_user(bob)
    await mass.music.albums.set_listen_later(db_id, True)

    # the album is added to the library proper, by nobody in particular
    set_current_user(None)
    await mass.music.albums.update_item_in_library(
        db_id,
        Album(
            item_id=str(db_id),
            provider="library",
            name="Graduates To Library",
            album_type=AlbumType.ALBUM,
            provider_mappings={
                ProviderMapping(
                    item_id="prov-graduate",
                    provider_domain="prov_a",
                    provider_instance="prov_a_inst",
                    audio_format=AudioFormat(),
                    in_library=True,
                )
            },
        ),
    )

    set_current_user(alice)
    assert await _shelf_names(mass) == []
    set_current_user(bob)
    assert await _shelf_names(mass) == []
