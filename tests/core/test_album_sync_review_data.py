"""
Integration tests for the review-data refresh inside the album library sync.

`MusicProvider._sync_library_albums` folds the critical_reception / dynamic_range
refresh into the *single* `update_item_in_library` call of the sync loop: a changed
CR (or a changed DR) widens the update condition instead of triggering a second
read-modify-write of the same row. The provider CR replaces the stored one (the
stored amg_dr survives when the files no longer report it); a provider that reports
none leaves it alone.

Uses the full ``mass`` fixture from ``tests/conftest.py`` so the AlbumsController is
backed by a real SQLite database.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from music_assistant_models.config_entries import ProviderConfig
from music_assistant_models.enums import MediaType, ProviderFeature, ProviderType
from music_assistant_models.media_items import Album, Artist, ProviderMapping
from music_assistant_models.media_items.metadata import (
    CriticalReception,
    MediaItemMetadata,
    ReviewLink,
    ReviewSourceEntry,
)
from music_assistant_models.provider import ProviderManifest

from music_assistant.helpers.critical_reception import critical_reception_for_update
from music_assistant.models.music_provider import MusicProvider

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant

_PROVIDER_DOMAIN = "fake_sync"
_PROVIDER_INSTANCE = "fake_sync--instance"
_ALBUM_ID = "alb-1"


class _FakeSyncProvider(MusicProvider):
    """Streaming-flavoured provider that serves a mutable canned library."""

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
    ) -> None:
        super().__init__(mass, manifest, config)
        self.library_albums: list[Album] = []

    @property
    def supported_features(self) -> set[ProviderFeature]:
        return {ProviderFeature.LIBRARY_ALBUMS}

    @property
    def is_streaming_provider(self) -> bool:
        return True

    def library_supported(self, media_type: MediaType) -> bool:
        return bool(media_type == MediaType.ALBUM)

    async def get_library_albums(self) -> AsyncGenerator[Album]:
        for album in self.library_albums:
            yield album


def _make_album(metadata: MediaItemMetadata) -> Album:
    """Compose the provider-side Album record served by the fake library."""
    artist = Artist(
        item_id="artist-1",
        provider=_PROVIDER_INSTANCE,
        name="Radiohead",
        provider_mappings={
            ProviderMapping(
                item_id="artist-1",
                provider_domain=_PROVIDER_DOMAIN,
                provider_instance=_PROVIDER_INSTANCE,
            )
        },
    )
    return Album(
        item_id=_ALBUM_ID,
        provider=_PROVIDER_INSTANCE,
        name="Kid A",
        artists=[artist],
        metadata=metadata,
        provider_mappings={
            ProviderMapping(
                item_id=_ALBUM_ID,
                provider_domain=_PROVIDER_DOMAIN,
                provider_instance=_PROVIDER_INSTANCE,
                available=True,
            )
        },
    )


@pytest.fixture
async def fake_provider(mass: MusicAssistant) -> AsyncGenerator[_FakeSyncProvider]:
    """Register a fake streaming provider whose library album is set per test."""
    manifest = ProviderManifest(
        type=ProviderType.MUSIC,
        domain=_PROVIDER_DOMAIN,
        name="Fake sync",
        description="Fake streaming provider for sync tests",
        codeowners=["@music-assistant"],
    )
    config = ProviderConfig(
        values={},
        type=ProviderType.MUSIC,
        domain=_PROVIDER_DOMAIN,
        instance_id=_PROVIDER_INSTANCE,
        name="Fake sync",
    )
    # Unset config entries would resolve to None; feed a falsy value for every key
    # so the log level falls back to "GLOBAL" and album-track import stays off
    # (the fake provider serves albums only).
    config.get_value = lambda *_a, **_k: False
    provider = _FakeSyncProvider(mass, manifest, config)
    provider.available = True
    mass._providers[provider.instance_id] = provider
    try:
        yield provider
    finally:
        mass._providers.pop(provider.instance_id, None)


async def _stored_album(mass: MusicAssistant, db_id: int) -> Album:
    return await mass.music.albums.get_library_item(db_id)


def _sources_by_name(cr: CriticalReception | None) -> dict[str, ReviewSourceEntry]:
    assert cr is not None
    return {src.source: src for src in cr.sources or []}


def _record_updates(
    mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[str | int, Album]]:
    """Wrap `albums.update_item_in_library` so every write of the sync loop is recorded."""
    updates: list[tuple[str | int, Album]] = []
    original_update = mass.music.albums.update_item_in_library

    async def _counting_update(item_id: str | int, update: Album, overwrite: bool = False) -> Album:
        updates.append((item_id, update))
        return await original_update(item_id, update, overwrite)

    monkeypatch.setattr(mass.music.albums, "update_item_in_library", _counting_update)
    return updates


@pytest.mark.usefixtures("fake_provider")
async def test_sync_persists_richer_cr_through_the_single_update_path(
    mass: MusicAssistant,
    fake_provider: _FakeSyncProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A richer CR is the *only* change: one update, carrying the provider item."""
    fake_provider.library_albums = [
        _make_album(
            MediaItemMetadata(
                description="stored bio",
                dynamic_range=11.0,
                critical_reception=CriticalReception(
                    sources=[ReviewSourceEntry(source="AMG", rating=4.5)]
                ),
            )
        )
    ]
    db_ids = await fake_provider._sync_library_albums()
    db_id = next(iter(db_ids))

    # second pass: identical album except AMG gained an accolade and TPS appeared
    prov_cr = CriticalReception(
        sources=[
            ReviewSourceEntry(source="AMG", rating=4.5, accolades=["Album of the Year (2000)"]),
            ReviewSourceEntry(source="TPS", rating=8.5),
        ]
    )
    prov_item = _make_album(MediaItemMetadata(critical_reception=prov_cr))
    fake_provider.library_albums = [prov_item]
    sync_details = await mass.music.albums.get_library_item_sync_details(
        prov_item.provider_mappings
    )
    assert sync_details is not None
    # nothing but the review data changed - without the widened condition this
    # album would not be written at all
    assert fake_provider._library_item_needs_update(sync_details, prov_item) is False

    updates: list[tuple[str | int, Album]] = []
    original_update = mass.music.albums.update_item_in_library

    async def _counting_update(item_id: str | int, update: Album, overwrite: bool = False) -> Album:
        updates.append((item_id, update))
        return await original_update(item_id, update, overwrite)

    monkeypatch.setattr(mass.music.albums, "update_item_in_library", _counting_update)
    assert await fake_provider._sync_library_albums() == {db_id}

    # exactly one write, and it carries the provider item - a second read-modify-write
    # of the library row would show up here as an extra call with a library payload
    assert len(updates) == 1
    assert updates[0][1] is prov_item

    stored = await _stored_album(mass, db_id)
    stored_sources = _sources_by_name(stored.metadata.critical_reception)
    assert stored_sources["AMG"].accolades == ["Album of the Year (2000)"]
    assert stored_sources["TPS"].rating == 8.5
    # the write went through the merge, so stored-only metadata survived it
    assert stored.metadata.description == "stored bio"
    assert stored.metadata.dynamic_range == 11.0


@pytest.mark.usefixtures("fake_provider")
async def test_sync_replaces_stored_cr_with_provider_cr(
    mass: MusicAssistant,
    fake_provider: _FakeSyncProvider,
) -> None:
    """The provider CR replaces the stored one: an unreported source is dropped (MUSIC-80)."""
    fake_provider.library_albums = [
        _make_album(
            MediaItemMetadata(
                dynamic_range=11.0,
                critical_reception=CriticalReception(
                    sources=[
                        ReviewSourceEntry(source="AMG", rating=4.5),
                        ReviewSourceEntry(source="TPS", rating=8.0),
                    ]
                ),
            )
        )
    ]
    db_ids = await fake_provider._sync_library_albums()
    db_id = next(iter(db_ids))

    prov_item = _make_album(
        MediaItemMetadata(
            dynamic_range=13.0,
            critical_reception=CriticalReception(
                sources=[
                    ReviewSourceEntry(
                        source="TPS", rating=8.5, accolades=["Album of the Year (2000)"]
                    )
                ]
            ),
        )
    )
    fake_provider.library_albums = [prov_item]
    assert await fake_provider._sync_library_albums() == {db_id}

    stored = await _stored_album(mass, db_id)
    assert stored.metadata.dynamic_range == 13.0
    stored_sources = _sources_by_name(stored.metadata.critical_reception)
    assert set(stored_sources) == {"TPS"}
    assert stored_sources["TPS"].rating == 8.5
    assert stored_sources["TPS"].accolades == ["Album of the Year (2000)"]


@pytest.mark.usefixtures("fake_provider")
async def test_sync_lands_corrected_cr_with_fewer_fields(
    mass: MusicAssistant,
    fake_provider: _FakeSyncProvider,
) -> None:
    """A re-tag that drops a mismatched accolade and gains review text lands (MUSIC-80)."""
    fake_provider.library_albums = [
        _make_album(
            MediaItemMetadata(
                critical_reception=CriticalReception(
                    sources=[
                        ReviewSourceEntry(
                            source="AMG",
                            rating=4.5,
                            accolades=["Album of the Year (2013)", "Review"],
                            links=[
                                ReviewLink(label="Album of the Year (2013)", url="https://a/aoty"),
                                ReviewLink(label="Review", url="https://a/wrong-review"),
                            ],
                        )
                    ]
                ),
            )
        )
    ]
    db_id = next(iter(await fake_provider._sync_library_albums()))

    corrected = ReviewSourceEntry(
        source="AMG",
        rating=4.5,
        accolades=["Album of the Year (2013)"],
        links=[ReviewLink(label="Album of the Year (2013)", url="https://a/aoty")],
        review="Full text.",
    )
    fake_provider.library_albums = [
        _make_album(MediaItemMetadata(critical_reception=CriticalReception(sources=[corrected])))
    ]
    await fake_provider._sync_library_albums()

    stored = await _stored_album(mass, db_id)
    assert _sources_by_name(stored.metadata.critical_reception)["AMG"] == corrected


@pytest.mark.usefixtures("fake_provider")
@pytest.mark.parametrize("prov_cr", [None, CriticalReception(sources=[])])
async def test_sync_keeps_stored_cr_when_provider_reports_none(
    mass: MusicAssistant,
    fake_provider: _FakeSyncProvider,
    monkeypatch: pytest.MonkeyPatch,
    prov_cr: CriticalReception | None,
) -> None:
    """No CR (or an empty one) is not a report: the stored CR stays and nothing is written."""
    stored_cr = CriticalReception(sources=[ReviewSourceEntry(source="AMG", rating=4.5)])
    fake_provider.library_albums = [_make_album(MediaItemMetadata(critical_reception=stored_cr))]
    db_id = next(iter(await fake_provider._sync_library_albums()))

    fake_provider.library_albums = [_make_album(MediaItemMetadata(critical_reception=prov_cr))]
    updates = _record_updates(mass, monkeypatch)
    await fake_provider._sync_library_albums()

    assert updates == []
    stored = await _stored_album(mass, db_id)
    assert stored.metadata.critical_reception == stored_cr


@pytest.mark.usefixtures("fake_provider")
async def test_sync_does_not_rewrite_when_only_stored_amg_dr_differs(
    mass: MusicAssistant,
    fake_provider: _FakeSyncProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stored amg_dr the files no longer report must not trigger a write every sync."""
    sources = [ReviewSourceEntry(source="AMG", rating=4.0, review="Text.")]
    fake_provider.library_albums = [
        _make_album(
            MediaItemMetadata(critical_reception=CriticalReception(amg_dr=7.0, sources=sources))
        )
    ]
    await fake_provider._sync_library_albums()

    fake_provider.library_albums = [
        _make_album(MediaItemMetadata(critical_reception=CriticalReception(sources=sources)))
    ]
    updates = _record_updates(mass, monkeypatch)
    await fake_provider._sync_library_albums()
    assert updates == []


async def test_sync_lands_review_text_when_probe_no_longer_reports_amg_dr(
    mass: MusicAssistant,
    fake_provider: _FakeSyncProvider,
) -> None:
    """A stored amg_dr no longer reported by the files must not freeze the CR (MUSIC-78)."""
    fake_provider.library_albums = [
        _make_album(
            MediaItemMetadata(
                critical_reception=CriticalReception(
                    amg_dr=7.0, sources=[ReviewSourceEntry(source="AMG", rating=4.0)]
                ),
            )
        )
    ]
    db_ids = await fake_provider._sync_library_albums()
    db_id = next(iter(db_ids))

    fake_provider.library_albums = [
        _make_album(
            MediaItemMetadata(
                critical_reception=CriticalReception(
                    sources=[ReviewSourceEntry(source="AMG", rating=4.0, review="Full text.")]
                ),
            )
        )
    ]
    assert await fake_provider._sync_library_albums() == {db_id}

    stored = await _stored_album(mass, db_id)
    assert stored.metadata.critical_reception is not None
    assert stored.metadata.critical_reception.amg_dr == 7.0
    assert _sources_by_name(stored.metadata.critical_reception)["AMG"].review == "Full text."


async def test_overwrite_keeps_stored_amg_dr_the_probe_no_longer_reports(
    mass: MusicAssistant,
    fake_provider: _FakeSyncProvider,
) -> None:
    """A forced refresh (overwrite) gains the review text without dropping amg_dr."""
    fake_provider.library_albums = [
        _make_album(
            MediaItemMetadata(
                critical_reception=CriticalReception(
                    amg_dr=7.0, sources=[ReviewSourceEntry(source="AMG", rating=4.0)]
                ),
            )
        )
    ]
    db_id = next(iter(await fake_provider._sync_library_albums()))

    update = _make_album(
        MediaItemMetadata(
            critical_reception=CriticalReception(
                sources=[ReviewSourceEntry(source="AMG", rating=4.0, review="Full text.")]
            ),
        )
    )
    await mass.music.albums.update_item_in_library(db_id, update, overwrite=True)

    stored = await _stored_album(mass, db_id)
    assert stored.metadata.critical_reception is not None
    assert stored.metadata.critical_reception.amg_dr == 7.0
    assert _sources_by_name(stored.metadata.critical_reception)["AMG"].review == "Full text."


@pytest.mark.usefixtures("fake_provider")
async def test_sync_writes_once_when_needs_update_and_review_data_both_changed(
    mass: MusicAssistant,
    fake_provider: _FakeSyncProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """needs_update *and* changed CR *and* changed DR still collapse into a single write."""
    fake_provider.library_albums = [
        _make_album(
            MediaItemMetadata(
                dynamic_range=11.0,
                critical_reception=CriticalReception(
                    sources=[ReviewSourceEntry(source="AMG", rating=4.5)]
                ),
            )
        )
    ]
    db_ids = await fake_provider._sync_library_albums()
    db_id = next(iter(db_ids))

    # a changed date_added makes _library_item_needs_update True on its own, while the
    # CR gained a source and the DR moved - the combination that used to fan out into
    # the plain update plus a second read-modify-write of the same row.
    prov_item = _make_album(
        MediaItemMetadata(
            dynamic_range=13.0,
            critical_reception=CriticalReception(
                sources=[
                    ReviewSourceEntry(source="AMG", rating=4.5),
                    ReviewSourceEntry(source="TPS", rating=8.5),
                ]
            ),
        )
    )
    prov_item.date_added = datetime(2001, 10, 2, tzinfo=UTC)
    fake_provider.library_albums = [prov_item]
    sync_details = await mass.music.albums.get_library_item_sync_details(
        prov_item.provider_mappings
    )
    assert sync_details is not None
    assert fake_provider._library_item_needs_update(sync_details, prov_item) is True
    assert (
        critical_reception_for_update(
            prov_item.metadata.critical_reception, sync_details.critical_reception
        )
        != sync_details.critical_reception
    )

    updates = _record_updates(mass, monkeypatch)
    assert await fake_provider._sync_library_albums() == {db_id}

    # one write only, carrying the provider item: the deleted arm issued a second
    # update_item_in_library with a library payload (and a second MEDIA_ITEM_UPDATED)
    assert len(updates) == 1
    assert updates[0][1] is prov_item

    stored = await _stored_album(mass, db_id)
    assert stored.metadata.dynamic_range == 13.0
    stored_sources = _sources_by_name(stored.metadata.critical_reception)
    assert set(stored_sources) == {"AMG", "TPS"}
    assert stored_sources["AMG"].rating == 4.5
    assert stored_sources["TPS"].rating == 8.5


@pytest.mark.usefixtures("fake_provider")
async def test_sync_stores_provider_dr_of_zero_over_stale_value(
    mass: MusicAssistant,
    fake_provider: _FakeSyncProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider DR of exactly 0.0 replaces a stale value, then converges."""
    fake_provider.library_albums = [_make_album(MediaItemMetadata(dynamic_range=11.0))]
    db_ids = await fake_provider._sync_library_albums()
    db_id = next(iter(db_ids))

    # 0.0 is a legitimate parse of a DYNAMIC_RANGE / DR_ALBUM file tag, so it must
    # overwrite the stored 11.0 like any other re-measured value
    fake_provider.library_albums = [_make_album(MediaItemMetadata(dynamic_range=0.0))]
    updates = _record_updates(mass, monkeypatch)
    assert await fake_provider._sync_library_albums() == {db_id}

    assert len(updates) == 1
    assert (await _stored_album(mass, db_id)).metadata.dynamic_range == 0.0

    # having landed, the value converges: no further sync re-writes this album
    assert await fake_provider._sync_library_albums() == {db_id}
    assert len(updates) == 1


@pytest.mark.usefixtures("fake_provider")
async def test_sync_stores_provider_dr_of_zero_when_nothing_is_stored_yet(
    mass: MusicAssistant,
    fake_provider: _FakeSyncProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 0.0 DR must still land on a row that has no stored DR at all."""
    fake_provider.library_albums = [_make_album(MediaItemMetadata())]
    db_ids = await fake_provider._sync_library_albums()
    db_id = next(iter(db_ids))
    assert (await _stored_album(mass, db_id)).metadata.dynamic_range is None

    # gap-fill rather than overwrite: narrowing the pending-change test to
    # `bool(dr_new)` would silently drop this album's DR
    fake_provider.library_albums = [_make_album(MediaItemMetadata(dynamic_range=0.0))]
    updates = _record_updates(mass, monkeypatch)
    assert await fake_provider._sync_library_albums() == {db_id}

    assert len(updates) == 1
    assert (await _stored_album(mass, db_id)).metadata.dynamic_range == 0.0

    # and having landed, it must then converge like any other stored value
    assert await fake_provider._sync_library_albums() == {db_id}
    assert len(updates) == 1
