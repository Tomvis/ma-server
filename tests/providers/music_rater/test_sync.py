"""Tests for the music-rater review-data refresh of streaming-only albums (MUSIC-79)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Self, cast
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from music_assistant_models.enums import ProviderType
from music_assistant_models.media_items import Album, Artist, ProviderMapping, UniqueList
from music_assistant_models.media_items.metadata import (
    CriticalReception,
    MediaItemMetadata,
    ReviewSourceEntry,
)

from music_assistant.providers.music_rater import (
    CONF_API_KEY,
    CONF_URL,
    MusicRaterProvider,
)


class _Response:
    def __init__(self, status: int, payload: Any = None) -> None:
        self.status = status
        self._payload = payload

    async def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise aiohttp.ClientResponseError(MagicMock(), (), status=self.status)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class _Session:
    """Answers GET /tags/for-album from a {(artist, album, year): fields} table."""

    def __init__(self, table: dict[tuple[str, str, int], dict[str, Any]], status: int = 0) -> None:
        self.table = table
        self.status = status
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append({"url": url, **kwargs})
        if self.status:
            return _Response(self.status)
        params = kwargs["params"]
        key = (params["artist"], params["album"], params["year"])
        if key in self.table:
            return _Response(200, {"fields": self.table[key]})
        return _Response(404, {"detail": "nope"})


def _album(
    item_id: int,
    name: str,
    year: int | None,
    domains: list[str],
    cr: CriticalReception | None = None,
) -> Album:
    return Album(
        item_id=str(item_id),
        provider="library",
        name=name,
        year=year,
        artists=UniqueList(
            [Artist(item_id="1", provider="library", name="Ihsahn", provider_mappings=set())]
        ),
        metadata=MediaItemMetadata(critical_reception=cr),
        provider_mappings={
            ProviderMapping(item_id=f"{d}-{item_id}", provider_domain=d, provider_instance=d)
            for d in domains
        },
    )


def _cr(rating: float) -> CriticalReception:
    return CriticalReception(sources=[ReviewSourceEntry(source="AMG", rating=rating)])


def _provider(session: _Session, albums: list[Album]) -> MusicRaterProvider:
    mass = MagicMock()
    mass.http_session = session

    async def _iter(**kwargs: Any) -> AsyncGenerator[Album]:
        # streaming mappings of these rows are in_library=False in production
        assert kwargs == {"library_items_only": False}
        for album in albums:
            yield album

    mass.music.albums.iter_library_items = _iter
    mass.music.albums.update_item_in_library = AsyncMock()
    manifest = MagicMock()
    manifest.type = ProviderType.PLUGIN
    manifest.domain = "music_rater"
    config = MagicMock()
    config.instance_id = "music_rater"
    values = {CONF_URL: "http://mr/api/v1/", CONF_API_KEY: "mr_key"}
    config.get_value = MagicMock(side_effect=lambda key, default=None: values.get(key, default))
    return MusicRaterProvider(mass, manifest, config, set())


def _updated(prov: MusicRaterProvider) -> dict[str, CriticalReception | None]:
    mock = cast("AsyncMock", prov.mass.music.albums.update_item_in_library)
    return {c.args[0]: c.args[1].metadata.critical_reception for c in mock.await_args_list}


async def test_changed_cr_is_written() -> None:
    """A streaming-only album whose music-rater data changed is updated."""
    session = _Session({("Ihsahn", "Pharos", 2020): {"AMG_RATING": "4.0"}})
    prov = _provider(session, [_album(7, "Pharos", 2020, ["tidal"], _cr(3.5))])
    await prov.sync_review_data()
    assert _updated(prov) == {"7": _cr(4.0)}
    call = session.calls[0]
    assert call["url"] == "http://mr/api/v1/tags/for-album"
    assert call["headers"] == {"Authorization": "Bearer mr_key"}


async def test_unchanged_cr_is_not_written() -> None:
    """No write (and no MEDIA_ITEM_UPDATED event) when nothing changed."""
    session = _Session({("Ihsahn", "Pharos", 2020): {"AMG_RATING": "4.0"}})
    prov = _provider(session, [_album(7, "Pharos", 2020, ["tidal"], _cr(4.0))])
    await prov.sync_review_data()
    assert _updated(prov) == {}


async def test_file_tagged_albums_are_skipped() -> None:
    """Navidrome and filesystem albums get their CR from the files' tags."""
    session = _Session({})
    albums = [
        _album(1, "A", 2020, ["opensubsonic", "tidal"]),
        _album(2, "B", 2020, ["filesystem_local"]),
    ]
    prov = _provider(session, albums)
    await prov.sync_review_data()
    assert session.calls == []


async def test_year_falls_back_one_either_side() -> None:
    """Streaming catalogs often date a release a year off music-rater's."""
    session = _Session({("Ihsahn", "Pharos", 2021): {"AMG_RATING": "4.0"}})
    prov = _provider(session, [_album(7, "Pharos", 2020, ["bandcamp"])])
    await prov.sync_review_data()
    assert [c["params"]["year"] for c in session.calls] == [2020, 2019, 2021]
    assert _updated(prov) == {"7": _cr(4.0)}


async def test_no_match_keeps_stored_cr() -> None:
    """A 404 on every year is not a report: nothing is written."""
    session = _Session({})
    prov = _provider(session, [_album(7, "Pharos", 2020, ["tidal"], _cr(3.5))])
    await prov.sync_review_data()
    assert len(session.calls) == 3
    assert _updated(prov) == {}


async def test_album_without_year_is_skipped() -> None:
    """The identity lookup needs a year."""
    session = _Session({})
    prov = _provider(session, [_album(7, "Pharos", None, ["tidal"])])
    await prov.sync_review_data()
    assert session.calls == []


async def test_stored_amg_dr_survives() -> None:
    """The update goes through critical_reception_for_update's amg_dr rule."""
    stored = CriticalReception(amg_dr=7.0, sources=[ReviewSourceEntry(source="AMG", rating=3.5)])
    session = _Session({("Ihsahn", "Pharos", 2020): {"AMG_RATING": "4.0"}})
    prov = _provider(session, [_album(7, "Pharos", 2020, ["tidal"], stored)])
    await prov.sync_review_data()
    assert _updated(prov) == {
        "7": CriticalReception(amg_dr=7.0, sources=[ReviewSourceEntry(source="AMG", rating=4.0)])
    }


@pytest.mark.parametrize("status", [401, 500])
async def test_server_error_aborts_the_run(status: int) -> None:
    """A dead key or server stops after the first album instead of hammering it."""
    session = _Session({}, status=status)
    albums = [_album(i, f"A{i}", 2020, ["tidal"]) for i in range(5)]
    prov = _provider(session, albums)
    with pytest.raises(aiohttp.ClientResponseError):
        await prov.sync_review_data()
    assert len(session.calls) == 1
    assert _updated(prov) == {}
