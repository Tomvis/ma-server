"""Tests that one failing provider cannot break the album 'other versions' lookup."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import ClientResponseError
from aiohttp.client_reqrep import RequestInfo
from music_assistant_models.enums import MediaType, ProviderFeature
from music_assistant_models.media_items import Album, Artist, ProviderMapping
from yarl import URL

from music_assistant.controllers.music.media.albums import AlbumsController
from music_assistant.models.music_provider import MusicProvider

if TYPE_CHECKING:
    from typing import Any


def _make_album(item_id: str, provider: str, name: str, artist: str) -> Album:
    """Return a minimal Album carrying one provider mapping."""
    return Album(
        item_id=item_id,
        provider=provider,
        name=name,
        artists=[
            Artist(
                item_id=f"{item_id}-a",
                provider=provider,
                name=artist,
                provider_mappings={
                    ProviderMapping(
                        item_id=f"{item_id}-a",
                        provider_domain=provider,
                        provider_instance=provider,
                    )
                },
            )
        ],
        provider_mappings={
            ProviderMapping(
                item_id=item_id,
                provider_domain=provider,
                provider_instance=provider,
            )
        },
    )


def _make_provider(instance_id: str) -> Mock:
    """Return a mocked music provider that supports album search."""
    prov = Mock(spec=MusicProvider)
    prov.instance_id = instance_id
    prov.domain = instance_id
    prov.name = instance_id
    prov.available = True
    prov.supported_media_types = [MediaType.ALBUM]
    prov.supported_features = {ProviderFeature.SEARCH}
    prov.is_streaming_provider = True
    return prov


def _content_type_error(url: str) -> ClientResponseError:
    """Return the error aiohttp raises when a provider answers HTML instead of JSON."""
    return ClientResponseError(
        RequestInfo(url=URL(url), method="GET", headers=None, real_url=URL(url)),  # type: ignore[arg-type]
        (),
        status=200,
        message="Attempt to decode JSON with unexpected mimetype: text/html; charset=utf-8",
    )


def _make_controller(providers: list[Mock], album: Album) -> AlbumsController:
    """Return an albums controller wired to the given mocked providers."""
    mass = Mock()
    mass.music.get_unique_providers = Mock(return_value=[p.instance_id for p in providers])
    mass.get_provider = Mock(
        side_effect=lambda instance_id, **_kwargs: next(
            (p for p in providers if p.instance_id == instance_id), None
        )
    )
    controller = AlbumsController.__new__(AlbumsController)
    controller.mass = mass
    controller.logger = logging.getLogger(__name__)
    controller.get_provider_item = AsyncMock(return_value=album)  # type: ignore[method-assign]
    return controller


@pytest.mark.asyncio
async def test_versions_survives_a_provider_that_raises() -> None:
    """
    A provider answering HTML instead of JSON must not fail the whole lookup.

    Regression test: Bandcamp's fuzzysearch endpoint intermittently returns an HTML
    page, so aiohttp raises ContentTypeError out of that provider's search. The album
    page calls music/albums/album_versions, so an unisolated raise there breaks the
    whole page rather than just dropping that one provider's contribution.
    """
    album = _make_album("lib1", "library", "Some Album", "Some Artist")
    broken = _make_provider("bandcamp")
    healthy = _make_provider("tidal")
    controller = _make_controller([broken, healthy], album)

    other_version = _make_album("t1", "tidal", "Some Album", "Some Artist")

    async def _search(_query: str, provider_id: str, **_kwargs: Any) -> list[Album]:
        if provider_id == "bandcamp":
            raise _content_type_error(
                "https://bandcamp.com/api/fuzzysearch/1/app_autocomplete?q=Some+Album"
            )
        return [other_version]

    controller.search = AsyncMock(side_effect=_search)  # type: ignore[method-assign]

    result = await controller.versions("lib1", "library")

    # the healthy provider's version is still returned; only bandcamp drops out
    assert [a.item_id for a in result] == ["t1"]


@pytest.mark.asyncio
async def test_versions_survives_a_failing_get_album_versions() -> None:
    """The specialized get_album_versions() call must be isolated the same way."""
    album = _make_album("lib1", "library", "Some Album", "Some Artist")
    broken = _make_provider("ytmusic")
    broken.supported_features = {ProviderFeature.SEARCH, ProviderFeature.ALBUM_VERSIONS}
    broken.get_album_versions = AsyncMock(side_effect=RuntimeError("provider exploded"))
    # the album must map to this provider, or get_album_versions is never reached
    album.provider_mappings.add(
        ProviderMapping(item_id="ytm1", provider_domain="ytmusic", provider_instance="ytmusic")
    )
    healthy = _make_provider("tidal")
    controller = _make_controller([broken, healthy], album)

    other_version = _make_album("t1", "tidal", "Some Album", "Some Artist")

    async def _search(_query: str, provider_id: str, **_kwargs: Any) -> list[Album]:
        return [other_version] if provider_id == "tidal" else []

    controller.search = AsyncMock(side_effect=_search)  # type: ignore[method-assign]

    result = await controller.versions("lib1", "library")

    assert [a.item_id for a in result] == ["t1"]
