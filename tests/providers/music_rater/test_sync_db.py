"""music-rater refresh against the real albums controller (SQLite-backed ``mass`` fixture)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from music_assistant_models.enums import ProviderType
from music_assistant_models.media_items import Album, Artist, ProviderMapping, UniqueList
from music_assistant_models.media_items.metadata import (
    CriticalReception,
    MediaItemMetadata,
    ReviewSourceEntry,
)

from music_assistant.providers.music_rater import CONF_API_KEY, CONF_URL, MusicRaterProvider

from .test_sync import _Session

if TYPE_CHECKING:
    import pytest

    from music_assistant.mass import MusicAssistant


async def test_refresh_lands_in_the_library_row(
    mass: MusicAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stored CR of a streaming-only album is replaced; the rest of the row stays."""
    mapping = ProviderMapping(item_id="t1", provider_domain="tidal", provider_instance="tidal")
    artist = Artist(item_id="a1", provider="tidal", name="Ihsahn", provider_mappings={mapping})
    stored = CriticalReception(
        amg_dr=7.0, sources=[ReviewSourceEntry(source="AMG", rating=3.5, review="old")]
    )
    added = await mass.music.albums.add_item_to_library(
        Album(
            item_id="t1",
            provider="tidal",
            name="Pharos",
            year=2020,
            artists=UniqueList([artist]),
            metadata=MediaItemMetadata(description="bio", critical_reception=stored),
            provider_mappings={mapping},
        )
    )
    manifest = MagicMock()
    manifest.type = ProviderType.PLUGIN
    manifest.domain = "music_rater"
    config = MagicMock()
    values = {CONF_URL: "http://mr/api/v1", CONF_API_KEY: "k"}
    config.get_value = MagicMock(side_effect=lambda key, default=None: values.get(key, default))
    prov = MusicRaterProvider(mass, manifest, config, set())
    session = _Session({("Ihsahn", "Pharos", 2020): {"AMG_RATING": "4.0", "AMG_REVIEW": "new"}})
    monkeypatch.setattr(type(mass), "http_session", property(lambda _self: session))

    await prov.sync_review_data()

    row = await mass.music.albums.get_library_item(added.item_id)
    assert row.metadata.critical_reception == CriticalReception(
        amg_dr=7.0, sources=[ReviewSourceEntry(source="AMG", rating=4.0, review="new")]
    )
    assert row.metadata.description == "bio"
    assert row.name == "Pharos"
