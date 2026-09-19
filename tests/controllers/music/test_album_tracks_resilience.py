"""Tests that one failing provider cannot empty an album's track listing."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import ClientResponseError
from aiohttp.client_reqrep import RequestInfo
from music_assistant_models.errors import MediaNotFoundError
from music_assistant_models.media_items import Album, Artist, ProviderMapping, Track
from yarl import URL

from music_assistant.controllers.music.media.albums import AlbumsController


def _mapping(item_id: str, provider: str) -> ProviderMapping:
    """Return a provider mapping for the given provider."""
    return ProviderMapping(item_id=item_id, provider_domain=provider, provider_instance=provider)


def _make_library_album(mappings: list[ProviderMapping]) -> Album:
    """Return a library album carrying the given provider mappings."""
    return Album(
        item_id="1",
        provider="library",
        name="Some Album",
        artists=[
            Artist(
                item_id="a1",
                provider="library",
                name="Some Artist",
                provider_mappings={_mapping("a1", "navidrome")},
            )
        ],
        provider_mappings=set(mappings),
    )


def _make_track(item_id: str, name: str, track_number: int) -> Track:
    """Return a library track as Navidrome would have contributed it."""
    return Track(
        item_id=item_id,
        provider="library",
        name=name,
        disc_number=1,
        track_number=track_number,
        provider_mappings={_mapping(f"nd-{item_id}", "navidrome")},
    )


def _content_type_error() -> ClientResponseError:
    """Return the raw aiohttp error a provider leaks when its API answers HTML."""
    url = URL("https://bandcamp.com/api/mobile/24/tralbum_details?tralbum_id=1306979829")
    return ClientResponseError(
        RequestInfo(url=url, method="GET", headers=None, real_url=url),  # type: ignore[arg-type]
        (),
        status=200,
        message="Attempt to decode JSON with unexpected mimetype: text/html; charset=utf-8",
    )


def _make_controller(album: Album, db_items: list[Track]) -> AlbumsController:
    """Return an albums controller serving the given library album and tracks."""
    controller = AlbumsController.__new__(AlbumsController)
    controller.mass = Mock()
    controller.logger = logging.getLogger(__name__)
    controller.get_library_item_by_prov_id = AsyncMock(return_value=album)  # type: ignore[method-assign]
    controller.get_library_album_tracks = AsyncMock(return_value=db_items)  # type: ignore[method-assign]
    controller._ensure_provider_filter = Mock(return_value=None)  # type: ignore[method-assign]
    return controller


@pytest.mark.parametrize(
    ("error", "label"),
    [
        (_content_type_error(), "raw aiohttp error the provider failed to wrap"),
        (MediaNotFoundError("gone"), "wrapped MusicAssistantError"),
    ],
)
@pytest.mark.asyncio
async def test_album_tracks_survive_a_failing_provider(error: Exception, label: str) -> None:
    """
    A failing provider mapping must cost only its own listing, not the whole album.

    Regression test: a library album mapped to both Navidrome and Bandcamp returned NO
    tracks at all when Bandcamp's API answered HTML, because aiohttp's ContentTypeError
    is not a MusicAssistantError and escaped the guard. The tracks the user actually owns
    live in the library and must still come back.
    """
    album = _make_library_album([_mapping("nd1", "navidrome"), _mapping("bc1", "bandcamp")])
    db_items = [_make_track("1", "Track One", 1), _make_track("2", "Track Two", 2)]
    controller = _make_controller(album, db_items)

    async def _provider_tracks(_item_id: str, provider_instance: str) -> list[Track]:
        if provider_instance == "bandcamp":
            raise error
        return []

    controller._get_provider_album_tracks = AsyncMock(side_effect=_provider_tracks)  # type: ignore[method-assign]

    result = await controller.tracks("1", "library")

    assert [t.name for t in result] == ["Track One", "Track Two"], label
