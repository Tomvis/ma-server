"""
Integration tests for music/albums/listen_later_add.

Exercises the two input modes (URI vs. artist+album), the optional
critical_reception payload, and the in-library rejection rule. Uses the
full ``mass`` fixture from ``tests/conftest.py`` so the AlbumsController is
backed by a real SQLite database.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import pytest
from music_assistant_models.auth import UserRole
from music_assistant_models.config_entries import ProviderConfig
from music_assistant_models.enums import MediaType, ProviderFeature, ProviderType
from music_assistant_models.errors import (
    AlreadyInLibraryError,
    InvalidDataError,
    MediaNotFoundError,
)
from music_assistant_models.media_items import (
    Album,
    Artist,
    ProviderMapping,
    SearchResults,
)
from music_assistant_models.media_items.metadata import (
    CriticalReception,
    ReviewLink,
    ReviewSourceEntry,
)
from music_assistant_models.provider import ProviderManifest

from music_assistant.controllers.music import controller as music_controller
from music_assistant.controllers.webserver.helpers.auth_middleware import set_current_user
from music_assistant.models.music_provider import MusicProvider

if TYPE_CHECKING:
    from music_assistant_models.auth import User

    from music_assistant.mass import MusicAssistant


_PROVIDER_DOMAIN = "fake_streaming"
_PROVIDER_INSTANCE = "fake_streaming--instance"
_RIVAL_DOMAIN = "rival_streaming"
_RIVAL_INSTANCE = "rival_streaming--instance"


class _FakeStreamingProvider(MusicProvider):
    """Streaming-flavoured provider that serves canned album records."""

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
        catalog: dict[str, Album],
        is_streaming: bool = True,
    ) -> None:
        # assigned before super().__init__: 2.10 reads is_streaming_provider during
        # construction (max_concurrent_streams), which resolves through this attribute
        self._is_streaming = is_streaming
        super().__init__(mass, manifest, config)
        self._catalog = catalog

    @property
    def supported_features(self) -> set[ProviderFeature]:
        return {ProviderFeature.SEARCH, ProviderFeature.LIBRARY_ALBUMS}

    @property
    def is_streaming_provider(self) -> bool:
        return self._is_streaming

    def library_supported(self, media_type: MediaType) -> bool:
        return bool(media_type == MediaType.ALBUM)

    async def search(
        self,
        search_query: str,
        media_types: list[MediaType] | None = None,
        limit: int = 5,
    ) -> SearchResults:
        if media_types and MediaType.ALBUM not in media_types:
            return SearchResults()
        # Trivial substring match — good enough for the canned catalog.
        query = search_query.lower()
        hits = [
            alb
            for alb in self._catalog.values()
            if query in f"{alb.artists[0].name} - {alb.name}".lower() or query in alb.name.lower()
        ]
        return SearchResults(albums=hits[:limit])

    async def get_album(self, prov_album_id: str) -> Album:
        if prov_album_id not in self._catalog:
            raise MediaNotFoundError(prov_album_id)
        return self._catalog[prov_album_id]

    async def get_artist(self, prov_artist_id: str) -> Artist:
        # Synthesize a minimal artist record from the catalog.
        for alb in self._catalog.values():
            for a in alb.artists:
                if a.item_id == prov_artist_id:
                    return Artist(
                        item_id=a.item_id,
                        provider=self.instance_id,
                        name=a.name,
                        provider_mappings={
                            ProviderMapping(
                                item_id=a.item_id,
                                provider_domain=self.domain,
                                provider_instance=self.instance_id,
                            )
                        },
                    )
        raise MediaNotFoundError(prov_artist_id)


def _make_album(
    item_id: str,
    name: str,
    artist_name: str,
    domain: str = _PROVIDER_DOMAIN,
    instance_id: str = _PROVIDER_INSTANCE,
) -> Album:
    """Compose an Album record carried by the fake provider's catalog."""
    artist = Artist(
        item_id=f"artist-{artist_name}",
        provider=instance_id,
        name=artist_name,
        provider_mappings={
            ProviderMapping(
                item_id=f"artist-{artist_name}",
                provider_domain=domain,
                provider_instance=instance_id,
            )
        },
    )
    return Album(
        item_id=item_id,
        provider=instance_id,
        name=name,
        artists=[artist],
        provider_mappings={
            ProviderMapping(
                item_id=item_id,
                provider_domain=domain,
                provider_instance=instance_id,
                available=True,
            )
        },
    )


@pytest.fixture(autouse=True)
async def signed_in_user(mass: MusicAssistant) -> AsyncGenerator[User]:
    """
    Sign a user in for the duration of each test in this module.

    The listen-later shelf is per-user, so every one of these API calls needs a
    calling user to attribute the save to. Autouse because that is a precondition of
    the endpoints under test, not a property any single test is asserting -- and MA
    itself enforces it: an install with no users refuses every websocket connection
    with "Setup required", so a signed-in caller is what production always has.
    """
    user = await mass.webserver.auth.create_user(username="listener", role=UserRole.USER)
    set_current_user(user)
    try:
        yield user
    finally:
        set_current_user(None)


def _register_provider(
    mass: MusicAssistant,
    domain: str,
    instance_id: str,
    catalog: dict[str, Album],
    is_streaming: bool = True,
) -> _FakeStreamingProvider:
    """Register a fake music provider serving the given catalog on the mass instance."""
    manifest = ProviderManifest(
        type=ProviderType.MUSIC,
        domain=domain,
        name="Fake streaming",
        description="Fake streaming provider for tests",
        codeowners=["@music-assistant"],
    )
    config = ProviderConfig(
        values={},
        type=ProviderType.MUSIC,
        domain=domain,
        instance_id=instance_id,
        name="Fake streaming",
    )
    # ProviderConfig.get_value() returns None for unset entries; the Provider
    # base class stringifies that into the logger level and chokes. Force-feed
    # the "GLOBAL" sentinel so the log level resolution stays valid.
    config.get_value = lambda *_a, **_k: "GLOBAL"
    provider = _FakeStreamingProvider(mass, manifest, config, catalog, is_streaming)
    provider.available = True
    mass._providers[provider.instance_id] = provider
    return provider


@pytest.fixture
async def fake_provider(
    mass: MusicAssistant,
) -> AsyncGenerator[_FakeStreamingProvider]:
    """Register a fake streaming provider with a tiny canned catalog."""
    catalog = {
        "alb-1": _make_album("alb-1", "Kid A", "Radiohead"),
        "alb-2": _make_album("alb-2", "OK Computer", "Radiohead"),
    }
    provider = _register_provider(mass, _PROVIDER_DOMAIN, _PROVIDER_INSTANCE, catalog)
    try:
        yield provider
    finally:
        mass._providers.pop(provider.instance_id, None)


@pytest.fixture
async def rival_provider(
    mass: MusicAssistant,
) -> AsyncGenerator[_FakeStreamingProvider]:
    """
    Register a second provider carrying the same album as ``fake_provider``.

    Registered after ``fake_provider`` when both fixtures are requested in that
    order, so it loses on discovery order alone and only wins on preference.
    """
    catalog = {
        "rival-1": _make_album("rival-1", "Kid A", "Radiohead", _RIVAL_DOMAIN, _RIVAL_INSTANCE),
    }
    provider = _register_provider(mass, _RIVAL_DOMAIN, _RIVAL_INSTANCE, catalog)
    try:
        yield provider
    finally:
        mass._providers.pop(provider.instance_id, None)


def _sample_critical_reception() -> CriticalReception:
    return CriticalReception(
        amg_dr=12.0,
        sources=[
            ReviewSourceEntry(
                source="AMG",
                rating=4.5,
                accolades=["Album of the Year (2000)"],
            ),
            ReviewSourceEntry(
                source="TPS",
                rating=8.5,
                favorite=True,
            ),
        ],
    )


@pytest.mark.usefixtures("fake_provider")
async def test_listen_later_add_by_artist_album_with_critical_reception(
    mass: MusicAssistant,
) -> None:
    """artist+album mode resolves through search and persists CR + listen_later."""
    cr = _sample_critical_reception()
    library_album = await mass.music.add_album_to_listen_later(
        artist="Radiohead",
        album="Kid A",
        critical_reception=cr,
    )

    assert library_album.provider == "library"
    assert library_album.listen_later is True
    assert library_album.metadata.critical_reception is not None
    stored = library_album.metadata.critical_reception
    assert stored.amg_dr == 12.0
    assert stored.sources is not None
    sources_by_id = {s.source: s for s in stored.sources}
    assert sources_by_id["AMG"].rating == 4.5
    assert sources_by_id["AMG"].accolades == ["Album of the Year (2000)"]
    assert sources_by_id["TPS"].rating == 8.5
    assert sources_by_id["TPS"].favorite is True

    # No provider mapping should be flipped to in_library — listen-later entries
    # stay out of the regular Albums view.
    assert all(pm.in_library is not True for pm in library_album.provider_mappings)


@pytest.mark.usefixtures("fake_provider")
async def test_listen_later_add_by_uri_back_compat(
    mass: MusicAssistant,
) -> None:
    """URI-only invocation (frontend's existing call) still works."""
    uri = f"{_PROVIDER_INSTANCE}://album/alb-2"
    library_album = await mass.music.add_album_to_listen_later(item=uri)
    assert library_album.listen_later is True
    assert library_album.name == "OK Computer"


@pytest.mark.usefixtures("fake_provider")
async def test_listen_later_add_rejects_when_album_already_in_library(
    mass: MusicAssistant,
) -> None:
    """An album already added to library proper can't be flagged listen-later."""
    # First add the album to the library normally so a provider_mapping with
    # in_library=True exists.
    uri = f"{_PROVIDER_INSTANCE}://album/alb-1"
    await mass.music.add_item_to_library(uri)

    # AlreadyInLibraryError is deliberately its own type rather than an InvalidDataError:
    # it carries a distinct error_code so clients can tell it apart from
    # AlreadyInListenLaterError, which maps to the same HTTP 409.
    with pytest.raises(AlreadyInLibraryError, match="already in your library"):
        await mass.music.add_album_to_listen_later(artist="Radiohead", album="Kid A")


@pytest.mark.usefixtures("fake_provider")
async def test_adding_to_library_graduates_album_out_of_listen_later(
    mass: MusicAssistant,
) -> None:
    """
    An album moved into the library proper is cleared from the listen-later pile.

    Library membership and listen-later are mutually exclusive; once the album gains
    an in_library mapping it must leave listen-later instead of lingering in both.
    """
    uri = f"{_PROVIDER_INSTANCE}://album/alb-1"
    # save it for later first (listen_later=1, no in_library mapping)
    saved = await mass.music.add_album_to_listen_later(item=uri)
    db_id = saved.item_id
    assert saved.listen_later is True
    assert not any(pm.in_library for pm in saved.provider_mappings)
    # now add the same album to the library proper
    await mass.music.add_item_to_library(uri)
    # it should now be a library item and no longer on listen-later
    refreshed = await mass.music.albums.get_library_item(db_id)
    assert any(pm.in_library for pm in refreshed.provider_mappings)
    assert refreshed.listen_later is False


async def test_listen_later_add_requires_item_or_artist_album(
    mass: MusicAssistant,
) -> None:
    """Calling with no inputs raises a clear error."""
    with pytest.raises(InvalidDataError, match="Either 'item'"):
        await mass.music.add_album_to_listen_later()


@pytest.mark.usefixtures("fake_provider")
async def test_listen_later_add_unmatched_artist_album_raises(
    mass: MusicAssistant,
) -> None:
    """No matching album across loaded providers raises MediaNotFoundError."""
    with pytest.raises(MediaNotFoundError):
        await mass.music.add_album_to_listen_later(artist="Nobody", album="Definitely Not An Album")


@pytest.mark.usefixtures("fake_provider")
async def test_listen_later_add_persists_cr_on_already_library_candidate(
    mass: MusicAssistant,
) -> None:
    """
    CR payload must persist when the candidate resolves as `library://`.

    First call seeds the listen-later row (no in_library mappings), so the
    second call sees `candidate.provider == "library"`. A previous version of
    the handler skipped the library add_item call in that branch, dropping the
    user-supplied CR. The second call's richer payload must end up in the DB.
    """
    # Seed: listen-later-add with a partial CR (just AMG rating, no DR / TPS).
    seed_cr = CriticalReception(
        amg_dr=None,
        sources=[ReviewSourceEntry(source="AMG", rating=4.0)],
    )
    await mass.music.add_album_to_listen_later(
        artist="Radiohead", album="Kid A", critical_reception=seed_cr
    )

    # Second call: richer payload — adds amg_dr and a TPS source. The
    # candidate now resolves as the existing library row (provider="library").
    richer_cr = CriticalReception(
        amg_dr=12.5,
        sources=[
            ReviewSourceEntry(source="AMG", rating=4.0, accolades=["Album of the Year (2000)"]),
            ReviewSourceEntry(source="TPS", rating=8.5, favorite=True),
        ],
    )
    library_album = await mass.music.add_album_to_listen_later(
        artist="Radiohead", album="Kid A", critical_reception=richer_cr
    )

    assert library_album.metadata.critical_reception is not None
    stored = library_album.metadata.critical_reception
    assert stored.amg_dr == 12.5
    sources_by_id = {s.source: s for s in (stored.sources or [])}
    assert "TPS" in sources_by_id
    assert sources_by_id["TPS"].rating == 8.5
    assert sources_by_id["TPS"].favorite is True
    assert sources_by_id["AMG"].accolades == ["Album of the Year (2000)"]


@pytest.mark.usefixtures("fake_provider")
async def test_listen_later_add_folds_legacy_types_labels_payload(
    mass: MusicAssistant,
) -> None:
    """A pre-3.2.0 payload using the deprecated types/labels lists is normalized on ingest."""
    legacy_cr = CriticalReception(
        sources=[
            ReviewSourceEntry(
                source="AMG",
                rating=4.5,
                types=["Review", "AOTM"],
                labels=["AOTY-2000", "RECORD_OF_THE_MONTH", "AOTM-2000-09"],
            ),
        ],
    )
    library_album = await mass.music.add_album_to_listen_later(
        artist="Radiohead", album="Kid A", critical_reception=legacy_cr
    )

    assert library_album.metadata.critical_reception is not None
    amg = {s.source: s for s in (library_album.metadata.critical_reception.sources or [])}["AMG"]
    # AOTM (type) + RECORD_OF_THE_MONTH + AOTM-2000-09 collapse to one dated entry.
    assert amg.accolades == [
        "Album of the Year (2000)",
        "Record of the Month (Sep 2000)",
        "Review",
    ]
    # the deprecated split fields are cleared once folded.
    assert amg.types is None
    assert amg.labels is None


@pytest.mark.usefixtures("fake_provider")
async def test_listen_later_add_persists_links(mass: MusicAssistant) -> None:
    """A payload's structured per-post links round-trip onto the stored album."""
    cr = CriticalReception(
        sources=[
            ReviewSourceEntry(
                source="AMG",
                rating=4.5,
                accolades=["Review", "Album of the Year (2024)"],
                links=[
                    ReviewLink(label="Review", url="https://amg/album-review/"),
                    ReviewLink(label="Album of the Year (2024)", url="https://amg/druhm-top/"),
                    ReviewLink(label="Album of the Year (2024)", url="https://amg/grier-top/"),
                ],
            ),
        ],
    )
    library_album = await mass.music.add_album_to_listen_later(
        artist="Radiohead", album="Kid A", critical_reception=cr
    )
    assert library_album.metadata.critical_reception is not None
    amg = {s.source: s for s in (library_album.metadata.critical_reception.sources or [])}["AMG"]
    assert amg.links is not None
    assert [(lk.label, lk.url) for lk in amg.links] == [
        ("Review", "https://amg/album-review/"),
        ("Album of the Year (2024)", "https://amg/druhm-top/"),
        ("Album of the Year (2024)", "https://amg/grier-top/"),
    ]


@pytest.mark.usefixtures("fake_provider")
async def test_listen_later_add_folds_legacy_review_url_payload(
    mass: MusicAssistant,
) -> None:
    """A pre-3.3.0 payload with a single review_url is folded into a "Review" link."""
    legacy_cr = CriticalReception(
        sources=[
            ReviewSourceEntry(source="AMG", rating=4.0, review_url="https://amg/old-review/"),
        ],
    )
    library_album = await mass.music.add_album_to_listen_later(
        artist="Radiohead", album="Kid A", critical_reception=legacy_cr
    )
    assert library_album.metadata.critical_reception is not None
    amg = {s.source: s for s in (library_album.metadata.critical_reception.sources or [])}["AMG"]
    assert amg.links is not None
    assert [(lk.label, lk.url) for lk in amg.links] == [
        ("Review", "https://amg/old-review/"),
    ]
    # the deprecated single-URL field is cleared once folded.
    assert amg.review_url is None


@pytest.mark.usefixtures("fake_provider")
async def test_listen_later_remove_deletes_orphan_row(
    mass: MusicAssistant,
) -> None:
    """
    A listen-later add followed by remove must not leave an orphan row.

    The row is invisible to every default view once the flag is cleared (no
    in_library mappings, not favorited, never played), so it would otherwise
    accumulate as dead DB state through listen-later churn.
    """
    library_album = await mass.music.add_album_to_listen_later(artist="Radiohead", album="Kid A")
    db_id = int(library_album.item_id)
    # Sanity: row exists with the flag set.
    assert library_album.listen_later is True
    assert await mass.music.albums.get_library_item(db_id) is not None

    await mass.music.remove_album_from_listen_later(db_id)

    # Row must be gone — no anchor remains to keep it.
    with pytest.raises(MediaNotFoundError):
        await mass.music.albums.get_library_item(db_id)


@pytest.mark.usefixtures("fake_provider")
async def test_listen_later_remove_keeps_row_with_other_anchor(
    mass: MusicAssistant, signed_in_user: User
) -> None:
    """If the row has another anchor (favorited), remove only clears the flag."""
    library_album = await mass.music.add_album_to_listen_later(artist="Radiohead", album="Kid A")
    db_id = int(library_album.item_id)
    # Make the row anchored by something other than listen_later.
    await mass.music.albums.set_favorite(db_id, True, [signed_in_user.user_id])

    await mass.music.remove_album_from_listen_later(db_id)

    # Row must still exist, with the flag cleared and the anchor preserved.
    refreshed = await mass.music.albums.get_library_item(db_id)
    assert refreshed.listen_later is False
    assert refreshed.favorite is True


@pytest.mark.usefixtures("fake_provider")
async def test_listen_later_remove_keeps_row_another_user_likes(
    mass: MusicAssistant,
) -> None:
    """
    Another account's like anchors the row too.

    Favorites are per-user, so the caller's own ``favorite`` says nothing about anyone
    else's. Without a household-wide check, one member clearing their shelf would delete
    an album somebody else has hearted.
    """
    library_album = await mass.music.add_album_to_listen_later(artist="Radiohead", album="Kid A")
    db_id = int(library_album.item_id)
    other = await mass.webserver.auth.create_user(username="housemate", role=UserRole.USER)
    await mass.music.albums.set_favorite(db_id, True, [other.user_id])

    await mass.music.remove_album_from_listen_later(db_id)

    refreshed = await mass.music.albums.get_library_item(db_id)
    assert refreshed.listen_later is False
    # the caller never liked it; the row survives on the housemate's like alone
    assert refreshed.favorite is None


@pytest.mark.usefixtures("fake_provider")
async def test_library_count_excludes_listen_later_only_items(
    mass: MusicAssistant,
) -> None:
    """
    library_count must mirror library_items' default in_library JOIN.

    Without the JOIN, a listen-later-only album (in_library=0 on all mappings)
    inflates the count beyond what /library_items returns at the same call.
    """
    # Seed one real library album + one listen-later-only album.
    library_uri = f"{_PROVIDER_INSTANCE}://album/alb-1"
    await mass.music.add_item_to_library(library_uri)
    await mass.music.add_album_to_listen_later(artist="Radiohead", album="OK Computer")

    visible = await mass.music.albums.library_items()
    total = await mass.music.albums.library_count()
    listen_later_total = await mass.music.albums.library_count(listen_later_only=True)

    assert total == len(visible), (
        f"library_count ({total}) must match the number of items library_items returns "
        f"({len(visible)})"
    )
    assert total == 1
    assert listen_later_total == 1


@pytest.mark.usefixtures("fake_provider", "rival_provider")
async def test_resolver_prefers_listed_provider_domains(
    mass: MusicAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A domain listed in the preference tuple outranks an equally good earlier hit."""
    # baseline: with no preference at all, both hits are plain streaming hits so
    # the (stable) discovery order decides — fake_provider is registered first.
    monkeypatch.setattr(music_controller, "_LISTEN_LATER_PREFERRED_PROVIDER_DOMAINS", ())
    resolved = await mass.music._resolve_album_by_artist_title("Radiohead", "Kid A")
    assert resolved.provider == _PROVIDER_INSTANCE

    # listing the later-registered provider's domain flips the winner
    monkeypatch.setattr(
        music_controller, "_LISTEN_LATER_PREFERRED_PROVIDER_DOMAINS", (_RIVAL_DOMAIN,)
    )
    resolved = await mass.music._resolve_album_by_artist_title("Radiohead", "Kid A")
    assert resolved.provider == _RIVAL_INSTANCE


async def test_resolver_prefers_streaming_over_local_provider(
    mass: MusicAssistant,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no listed domains, a streaming hit still beats a local one found first."""
    monkeypatch.setattr(music_controller, "_LISTEN_LATER_PREFERRED_PROVIDER_DOMAINS", ())
    local = _register_provider(
        mass,
        _PROVIDER_DOMAIN,
        _PROVIDER_INSTANCE,
        {"alb-1": _make_album("alb-1", "Kid A", "Radiohead")},
        is_streaming=False,
    )
    streaming = _register_provider(
        mass,
        _RIVAL_DOMAIN,
        _RIVAL_INSTANCE,
        {"rival-1": _make_album("rival-1", "Kid A", "Radiohead", _RIVAL_DOMAIN, _RIVAL_INSTANCE)},
    )
    try:
        resolved = await mass.music._resolve_album_by_artist_title("Radiohead", "Kid A")
        assert resolved.provider == _RIVAL_INSTANCE
    finally:
        mass._providers.pop(local.instance_id, None)
        mass._providers.pop(streaming.instance_id, None)


@pytest.mark.usefixtures("fake_provider")
async def test_a_second_user_can_save_an_album_the_first_already_saved(
    mass: MusicAssistant, signed_in_user: User
) -> None:
    """
    Two accounts can hold the same album on their shelves at the same time.

    This is the headline symptom of the household-wide shelf, at the command level:
    `add_album_to_listen_later` refuses a re-add with AlreadyInListenLaterError, and
    when "already saved" was one bit on the shared album row, the first person in a
    household to save an album locked everybody else out of saving it. The check now
    reads the *caller's* shelf, so the second save is a first save.

    Nothing else asserts this. The per-user reads are covered a level down, in
    tests/core/test_album_listen_later_per_user.py, but the refusal lives here in the
    command and only a second account reaching for an album the first already has can
    reach it.
    """
    uri = f"{_PROVIDER_INSTANCE}://album/alb-1"
    first = await mass.music.add_album_to_listen_later(item=uri)
    assert first.listen_later is True
    assert first.name == "Kid A"

    # a second account reaches for the very same album -- this must not raise
    second_user = await mass.webserver.auth.create_user(username="housemate", role=UserRole.USER)
    set_current_user(second_user)
    second = await mass.music.add_album_to_listen_later(item=uri)
    assert second.listen_later is True
    # the same library row, shared: this is one album on two shelves, not a duplicate
    assert second.item_id == first.item_id

    shelf = await mass.music.albums.library_items(listen_later=True, limit=0)
    assert [album.name for album in shelf] == ["Kid A"]

    # and the first account's shelf is untouched by any of it
    set_current_user(signed_in_user)
    shelf = await mass.music.albums.library_items(listen_later=True, limit=0)
    assert [album.name for album in shelf] == ["Kid A"]
