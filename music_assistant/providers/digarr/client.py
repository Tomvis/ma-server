"""Typed HTTP client for the digarr API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiohttp import ClientSession

_LOGGER = logging.getLogger(__name__)


class DigarrError(Exception):
    """Any failure talking to digarr."""


class DigarrAuthError(DigarrError):
    """The API key was rejected: revoked, expired, or wrong."""


@dataclass(frozen=True)
class DigarrRecommendation:
    """One pending recommendation, flattened from digarr's row + nested artist."""

    id: int
    kind: str
    score: float
    status: str
    artist_id: int
    artist_name: str
    artist_mbid: str
    image_url: str | None
    genres: tuple[str, ...]
    ai_reasoning: str | None
    release_group_mbid: str | None
    release_group_title: str | None


class DigarrClient:
    """Minimal client covering exactly what the provider needs."""

    def __init__(self, url: str, api_key: str, session: ClientSession) -> None:
        """Bind the client to one digarr instance and one API key."""
        self._base = url.rstrip("/")
        self._api_key = api_key
        self._session = session

    async def get_pending(self, limit: int) -> list[DigarrRecommendation]:
        """Fetch the highest-scoring pending recommendations."""
        payload = await self._request(
            "GET",
            "/api/v1/recommendations",
            params={"status": "pending", "sort": "score_desc", "limit": str(limit)},
        )
        results: list[DigarrRecommendation] = []
        for raw in payload.get("items", []):
            artist = raw.get("artist") or {}
            mbid = artist.get("mbid")
            # Without an MBID there is no reliable way to resolve the artist to a
            # real item, and a card that cannot be opened is worse than no card.
            if not mbid:
                continue
            # A missing/non-numeric id is a malformed record, not something a caller
            # catching DigarrError (e.g. _refresh) would expect as a bare KeyError --
            # but it must not be fatal to the whole page: one bad record discarding
            # every other, good, recommendation in the same response would be worse
            # than just dropping the one card that cannot be resolved.
            try:
                rec_id = int(raw["id"])
                artist_id = int(artist["id"])
            except (KeyError, TypeError, ValueError) as err:
                _LOGGER.warning(
                    "digarr: skipping malformed pending recommendation (%s): %s", err, raw
                )
                continue
            results.append(
                DigarrRecommendation(
                    id=rec_id,
                    kind=str(raw.get("kind", "artist")),
                    score=float(raw.get("score", 0.0)),
                    status=str(raw.get("status", "pending")),
                    artist_id=artist_id,
                    artist_name=str(artist.get("name") or "Unknown Artist"),
                    artist_mbid=str(mbid),
                    image_url=artist.get("imageUrl"),
                    genres=tuple(artist.get("genres") or ()),
                    ai_reasoning=raw.get("aiReasoning"),
                    release_group_mbid=raw.get("recommendedReleaseGroupId"),
                    release_group_title=raw.get("recommendedReleaseGroupTitle"),
                )
            )
        return results

    async def set_status(
        self, rec_id: int, status: str, remove_lidarr_artist: bool = False
    ) -> dict[str, Any]:
        """
        Change a recommendation's status.

        Deliberately sends no target options. digarr accepts lidarrTargetId,
        approvalMode, monitorOption, selectedAlbumIds and profile overrides, but
        omitting them lets digarr choose the calling user's own Lidarr target --
        which is the correct per-user behaviour and the whole point of per-user keys.

        `remove_lidarr_artist` maps to digarr's `removeLidarrArtist` flag, which
        DEFAULTS TO FALSE SERVER-SIDE. Reverting to "pending" without it unwinds
        the row and leaves Lidarr untouched; only undo-of-an-approve should set it.
        digarr made the removal opt-in deliberately, because `status: "pending"`
        is also how its own UI restores a rejected recommendation.

        :param rec_id: digarr's internal recommendation id.
        :param status: The new status, e.g. "approved", "rejected", or "pending".
        :param remove_lidarr_artist: Set only when reverting an approve that already
            added a Lidarr artist and the artist should be removed with it.
        """
        payload: dict[str, Any] = {"status": status}
        if remove_lidarr_artist:
            payload["removeLidarrArtist"] = True
        result = await self._request("PATCH", f"/api/v1/recommendations/{rec_id}", json=payload)
        return result or {}

    async def block_artist(self, artist_id: int) -> None:
        """
        Permanently block an artist so it is never recommended again.

        Takes digarr's INTERNAL artist id, not the MusicBrainz id --
        createBlockSchema in src/server/schemas/... declares
        `artistId: z.number().int().positive()`.

        :param artist_id: digarr's internal artist id (not an MBID).
        """
        await self._request("POST", "/api/v1/artist-blocks", json={"artistId": artist_id})

    async def whoami(self) -> tuple[str, bool]:
        """
        Return (username, is_admin) for the key's owning user.

        GET /api/v1/auth/me (routes/auth.ts:278) returns the user row, resolved
        from the context userId -- so it reflects whichever user minted the key.
        It does NOT report the key's scopes; digarr has no endpoint that does.
        Test connection therefore proves identity and reachability, and a missing
        scope surfaces as a 403 on first use.
        """
        payload = await self._request("GET", "/api/v1/auth/me")
        return str(payload.get("username", "?")), bool(payload.get("isAdmin"))

    @property
    def _headers(self) -> dict[str, str]:
        # Header auth only. digarr accepts ?token= on two SSE/audio routes, and a
        # long-lived credential has no business in a URL or an access log.
        return {"Authorization": f"Bearer {self._api_key}"}

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            async with self._session.request(
                method, f"{self._base}{path}", headers=self._headers, **kwargs
            ) as response:
                if response.status in (401, 403):
                    raise DigarrAuthError(
                        f"digarr rejected the API key ({response.status}). "
                        "It may be revoked, expired, or missing a required scope."
                    )
                if response.status >= 400:
                    body = await response.text()
                    raise DigarrError(f"digarr returned {response.status}: {body[:200]}")
                if response.status == 204:
                    return None
                return await response.json()
        except DigarrError:
            raise
        except Exception as err:
            raise DigarrError(f"Could not reach digarr at {self._base}: {err}") from err
