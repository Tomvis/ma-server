"""Integration tests for music/albums/listen_later_add.

Exercises the two input modes (URI vs. artist+album), the optional
critical_reception payload, and the in-library rejection rule. Uses the
full ``mass`` fixture from ``tests/conftest.py`` so the AlbumsController is
backed by a real SQLite database.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import pytest
from music_assistant_models.config_entries import ProviderConfig
from music_assistant_models.enums import MediaType, ProviderFeature, ProviderType
from music_assistant_models.errors import InvalidDataError, MediaNotFoundError
from music_assistant_models.media_items import (
    Album,
    Artist,
    ProviderMapping,
    SearchResults,
)
from music_assistant_models.media_items.metadata import (
    CriticalReception,
    ReviewSourceEntry,
)
from music_assistant_models.provider import ProviderManifest

from music_assistant.models.music_provider import MusicProvider

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant


_PROVIDER_DOMAIN = "fake_streaming"
_PROVIDER_INSTANCE = "fake_streaming--instance"


class _FakeStreamingProvider(MusicProvider):
    """Streaming-flavoured provider that serves canned album records."""

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
        catalog: dict[str, Album],
    ) -> None:
        super().__init__(mass, manifest, config)
        self._catalog = catalog

    @property
    def supported_features(self) -> set[ProviderFeature]:
        return {ProviderFeature.SEARCH, ProviderFeature.LIBRARY_ALBUMS}

    @property
    def is_streaming_provider(self) -> bool:
        return True

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


def _make_album(item_id: str, name: str, artist_name: str) -> Album:
    """Compose an Album record carried by the fake provider's catalog."""
    artist = Artist(
        item_id=f"artist-{artist_name}",
        provider=_PROVIDER_INSTANCE,
        name=artist_name,
        provider_mappings={
            ProviderMapping(
                item_id=f"artist-{artist_name}",
                provider_domain=_PROVIDER_DOMAIN,
                provider_instance=_PROVIDER_INSTANCE,
            )
        },
    )
    return Album(
        item_id=item_id,
        provider=_PROVIDER_INSTANCE,
        name=name,
        artists=[artist],
        provider_mappings={
            ProviderMapping(
                item_id=item_id,
                provider_domain=_PROVIDER_DOMAIN,
                provider_instance=_PROVIDER_INSTANCE,
                available=True,
            )
        },
    )


@pytest.fixture
async def fake_provider(
    mass: MusicAssistant,
) -> AsyncGenerator[_FakeStreamingProvider, None]:
    """Register a fake streaming provider with a tiny canned catalog."""
    catalog = {
        "alb-1": _make_album("alb-1", "Kid A", "Radiohead"),
        "alb-2": _make_album("alb-2", "OK Computer", "Radiohead"),
    }
    manifest = ProviderManifest(
        type=ProviderType.MUSIC,
        domain=_PROVIDER_DOMAIN,
        name="Fake streaming",
        description="Fake streaming provider for tests",
        codeowners=["@music-assistant"],
    )
    config = ProviderConfig(
        values={},
        type=ProviderType.MUSIC,
        domain=_PROVIDER_DOMAIN,
        instance_id=_PROVIDER_INSTANCE,
        name="Fake streaming",
    )
    # ProviderConfig.get_value() returns None for unset entries; the Provider
    # base class stringifies that into the logger level and chokes. Force-feed
    # the "GLOBAL" sentinel so the log level resolution stays valid.
    config.get_value = lambda *_a, **_k: "GLOBAL"
    provider = _FakeStreamingProvider(mass, manifest, config, catalog)
    provider.available = True
    mass._providers[provider.instance_id] = provider
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
                labels=["AOTY-2000"],
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
    assert sources_by_id["AMG"].labels == ["AOTY-2000"]
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

    with pytest.raises(InvalidDataError, match="already in your library"):
        await mass.music.add_album_to_listen_later(artist="Radiohead", album="Kid A")


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
