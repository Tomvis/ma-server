"""Thin async client for the music-rater album API.

We talk to music-rater (operator-run companion service), not Lidarr directly.
music-rater stamps the Music Assistant URI on every album it has synced to MA,
so resolving an album_id is normally an exact-match lookup. Falling back to a
text search covers the case where the user invokes the action on something
they added to MA by hand before music-rater could pick it up.

Endpoints used:
- GET /api/v1/albums?music_assistant_uri=<uri>   exact resolve
- GET /api/v1/albums?search=<artist+album>       best-effort fallback
- POST /api/v1/albums/{album_id}/lidarr/queue    set lidarr_manual_add + sync
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from music_assistant_models.errors import (
    InvalidDataError,
    MediaNotFoundError,
    ProviderUnavailableError,
)

if TYPE_CHECKING:
    from aiohttp import ClientSession


class MusicRaterError(Exception):
    """Raised for unexpected music-rater API responses."""


class MusicRaterClient:
    """Minimal async wrapper for the music-rater album endpoints we need."""

    def __init__(
        self,
        url: str,
        session: ClientSession,
        *,
        verify_ssl: bool = True,
    ) -> None:
        """Build a client bound to one music-rater instance."""
        self._base = url.rstrip("/")
        self._headers = {"Accept": "application/json"}
        self._session = session
        self._verify_ssl = verify_ssl

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> tuple[int, Any]:
        """Issue a request and return (status, parsed-body-or-text)."""
        url = f"{self._base}/api/v1/{path.lstrip('/')}"
        async with self._session.request(
            method,
            url,
            headers=self._headers,
            ssl=self._verify_ssl,
            **kwargs,
        ) as resp:
            status = resp.status
            if status == 204 or resp.content_length == 0:
                return status, None
            ctype = resp.headers.get("Content-Type", "")
            if "application/json" in ctype:
                return status, await resp.json()
            return status, await resp.text()

    async def ping(self) -> None:
        """Cheap connectivity probe: list one album and check the response shape."""
        status, body = await self._request("GET", "albums", params={"limit": "1"})
        if status >= 500:
            raise ProviderUnavailableError(f"music-rater returned {status} on connectivity probe")
        if status >= 400 or not isinstance(body, dict) or "items" not in body:
            snippet = body if isinstance(body, str) else str(body)[:200]
            raise MusicRaterError(
                f"Unexpected response from music-rater /albums?limit=1 "
                f"(status={status}): {snippet[:200]}"
            )

    async def resolve_by_uri(self, music_assistant_uri: str) -> int | None:
        """Look up an album_id by its MA URI. Returns None if no match."""
        return await self._first_album_id(
            params={"music_assistant_uri": music_assistant_uri, "limit": "1"}
        )

    async def resolve_by_search(self, query: str) -> int | None:
        """Free-text album search (artist + album). Returns the top hit's id."""
        return await self._first_album_id(params={"search": query, "limit": "1"})

    async def _first_album_id(self, *, params: dict[str, str]) -> int | None:
        status, body = await self._request("GET", "albums", params=params)
        if status >= 500:
            raise ProviderUnavailableError(f"music-rater returned {status} from /albums")
        if status >= 400 or not isinstance(body, dict):
            snippet = body if isinstance(body, str) else str(body)[:200]
            raise MusicRaterError(f"GET /albums failed (status={status}): {snippet[:200]}")
        items = body.get("items") or []
        if not items:
            return None
        first = items[0]
        if not isinstance(first, dict) or "id" not in first:
            raise MusicRaterError(f"music-rater returned malformed album item: {first!r}")
        return int(first["id"])

    async def queue_lidarr(self, album_id: int) -> dict[str, Any]:
        """POST /albums/{id}/lidarr/queue — flips lidarr_manual_add + inline sync.

        Maps documented status codes to MA error types so the global error
        toast carries a useful message.
        """
        status, body = await self._request("POST", f"albums/{int(album_id)}/lidarr/queue")
        if status == 200 and isinstance(body, dict):
            return body
        # Per music-rater's documented error surface.
        snippet = body if isinstance(body, str) else str(body)[:200]
        if status == 404:
            raise MediaNotFoundError(f"music-rater no longer has album_id={album_id} (deleted?)")
        if status == 409:
            raise InvalidDataError(
                "music-rater has lidarr_manual_skip=True on this album — operator "
                "permanently excluded it from Lidarr"
            )
        if status == 502:
            raise ProviderUnavailableError("Lidarr is unreachable from music-rater (502)")
        if status == 503:
            raise ProviderUnavailableError(
                "music-rater's Lidarr config is incomplete (URL / API key / "
                "profile / root folder missing)"
            )
        raise MusicRaterError(
            f"POST /albums/{album_id}/lidarr/queue failed (status={status}): {snippet[:200]}"
        )
