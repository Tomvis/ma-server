"""Tests for the digarr HTTP client."""

from __future__ import annotations

import pytest

from music_assistant.providers.digarr.client import (
    DigarrAuthError,
    DigarrClient,
    DigarrError,
)

from .conftest import FakeResponse, FakeSession

BASE = "http://digarr:3000"

SAMPLE = {
    "total": 2,
    "items": [
        {
            "id": 11,
            "kind": "artist",
            "score": 1.0,
            "status": "pending",
            "aiReasoning": None,
            "recommendedReleaseGroupId": None,
            "recommendedReleaseGroupTitle": None,
            "artist": {
                "id": 101,
                "mbid": "c14b4180-dc87-481e-b17a-64e4150f90f6",
                "name": "Opeth",
                "genres": ["progressive death metal", "metal"],
                "imageUrl": "http://example/opeth.jpg",
            },
        },
        {
            "id": 12,
            "kind": "album",
            "score": 0.8,
            "status": "pending",
            "aiReasoning": "Sits between two bands you already own.",
            "recommendedReleaseGroupId": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "recommendedReleaseGroupTitle": "Some Album",
            "artist": {
                "id": 102,
                "mbid": "bfe00ef8-f6f2-4810-999e-65e46b5970ab",
                "name": "Ulcerate",
                "genres": [],
                "imageUrl": None,
            },
        },
    ],
}


async def test_get_pending_parses_the_real_response_shape(session: FakeSession) -> None:
    """Recommendations map onto the dataclass, including the nested artist."""
    session.queue(FakeResponse(200, SAMPLE))
    client = DigarrClient(BASE, "dgr_x_y", session)

    recs = await client.get_pending(limit=15)

    assert [r.id for r in recs] == [11, 12]
    assert recs[0].artist_name == "Opeth"
    assert recs[0].artist_id == 101
    assert recs[0].artist_mbid == "c14b4180-dc87-481e-b17a-64e4150f90f6"
    assert recs[0].genres == ("progressive death metal", "metal")
    assert recs[0].ai_reasoning is None
    assert recs[1].kind == "album"
    assert recs[1].release_group_mbid == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


async def test_get_pending_sends_the_api_key_as_a_bearer(session: FakeSession) -> None:
    """The key travels in the Authorization header, never in the query string."""
    session.queue(FakeResponse(200, {"items": [], "total": 0}))
    client = DigarrClient(BASE, "dgr_x_y", session)

    await client.get_pending(limit=15)

    _method, url, kwargs = session.calls[0]
    assert kwargs["headers"]["Authorization"] == "Bearer dgr_x_y"
    assert "dgr_x_y" not in url
    assert "dgr_x_y" not in str(kwargs.get("params", {}))


async def test_get_pending_requests_the_documented_query(session: FakeSession) -> None:
    """Highest-scoring pending first, limited to the caller's buffer."""
    session.queue(FakeResponse(200, {"items": [], "total": 0}))
    client = DigarrClient(BASE, "k", session)

    await client.get_pending(limit=15)

    _method, url, kwargs = session.calls[0]
    assert url == f"{BASE}/api/v1/recommendations"
    assert kwargs["params"] == {"status": "pending", "sort": "score_desc", "limit": "15"}


async def test_skips_records_with_no_mbid(session: FakeSession) -> None:
    """A recommendation with no MusicBrainz id cannot be resolved, so it is dropped."""
    session.queue(
        FakeResponse(
            200,
            {
                "total": 1,
                "items": [
                    {
                        "id": 1,
                        "kind": "artist",
                        "score": 1.0,
                        "status": "pending",
                        "artist": {"id": 1, "name": "X", "mbid": None},
                    }
                ],
            },
        )
    )
    client = DigarrClient(BASE, "k", session)

    assert await client.get_pending(limit=15) == []


@pytest.mark.parametrize("status", [401, 403])
async def test_rejected_key_raises_auth_error(session: FakeSession, status: int) -> None:
    """A revoked key, or one missing a scope, is distinguishable from a transport failure."""
    session.queue(FakeResponse(status, body="nope"))
    client = DigarrClient(BASE, "bad", session)

    with pytest.raises(DigarrAuthError):
        await client.get_pending(limit=15)


async def test_500_raises_digarr_error(session: FakeSession) -> None:
    """Server errors surface as the provider's own error type."""
    session.queue(FakeResponse(500, body="boom"))
    client = DigarrClient(BASE, "k", session)

    with pytest.raises(DigarrError):
        await client.get_pending(limit=15)


async def test_transport_failure_raises_digarr_error(session: FakeSession) -> None:
    """An unreachable digarr is the provider's error, not a bare OSError."""
    session.queue(OSError("connection refused"))
    client = DigarrClient(BASE, "k", session)

    with pytest.raises(DigarrError):
        await client.get_pending(limit=15)


async def test_set_status_patches_the_recommendation(session: FakeSession) -> None:
    """Approve sends the minimal payload so digarr picks the user's default target."""
    session.queue(FakeResponse(200, {"status": "approved"}))
    client = DigarrClient(BASE, "k", session)

    result = await client.set_status(11, "approved")

    assert result["status"] == "approved"
    method, url, kwargs = session.calls[0]
    assert method == "PATCH"
    assert url == f"{BASE}/api/v1/recommendations/11"
    assert kwargs["json"] == {"status": "approved"}


async def test_undo_reverts_to_pending_and_opts_into_removal(session: FakeSession) -> None:
    """Undo of an approve must opt in explicitly, or digarr leaves the artist."""
    session.queue(FakeResponse(200, {"status": "pending", "lidarrArtistRemoved": True}))
    client = DigarrClient(BASE, "k", session)

    result = await client.set_status(11, "pending", remove_lidarr_artist=True)

    assert result["lidarrArtistRemoved"] is True
    _method, _url, kwargs = session.calls[0]
    assert kwargs["json"] == {"status": "pending", "removeLidarrArtist": True}


async def test_a_plain_revert_never_asks_for_removal(session: FakeSession) -> None:
    """A revert that is not an undo-of-approve must not carry the flag."""
    session.queue(FakeResponse(200, {"status": "pending"}))
    client = DigarrClient(BASE, "k", session)

    await client.set_status(11, "pending")

    _method, _url, kwargs = session.calls[0]
    assert kwargs["json"] == {"status": "pending"}


async def test_block_artist_sends_the_internal_id(session: FakeSession) -> None:
    """Digarr's block endpoint keys on its own artist id, not the MusicBrainz id."""
    session.queue(FakeResponse(204))
    client = DigarrClient(BASE, "k", session)

    await client.block_artist(101)

    method, url, kwargs = session.calls[0]
    assert method == "POST"
    assert url == f"{BASE}/api/v1/artist-blocks"
    assert kwargs["json"] == {"artistId": 101}
