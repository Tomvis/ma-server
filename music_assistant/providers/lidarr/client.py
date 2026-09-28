"""
Thin async client for the music-rater album API.

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

from aiohttp import ClientResponseError, ClientTimeout
from music_assistant_models.errors import (
    InvalidDataError,
    MediaNotFoundError,
    ProviderUnavailableError,
)

if TYPE_CHECKING:
    from aiohttp import ClientSession


# Cap each music-rater call. Resolve / ping / list are quick reads.
_REQUEST_TIMEOUT = ClientTimeout(total=30)

# The /lidarr/queue endpoint runs an inline single-album sync against Lidarr.
# For a brand-new artist that means waiting out Lidarr's async metadata
# refresh — music-rater polls the discography until the target album hydrates
# (backoff to ~120s) and then drains the post-add RefreshArtist before
# re-verifying the monitored flag (backoff to ~60s). Give the POST a much
# larger ceiling so a legitimate new-artist add isn't cut off mid-sync.
# Lidarr's absolute worst case can still edge past this; music-rater finishes
# server-side and the next add/sync reconciles, so we don't size for it.
_QUEUE_REQUEST_TIMEOUT = ClientTimeout(total=180)


def _snippet(body: Any) -> str:
    """Render a response body as a short, log/error-safe excerpt."""
    return (body if isinstance(body, str) else str(body))[:200]


class MusicRaterError(InvalidDataError):
    """
    Raised for unexpected music-rater API responses.

    Inherits from InvalidDataError so the global error toast on the frontend
    renders the message (it filters on MusicAssistantError subclasses).
    """


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

    async def ping(self) -> None:
        """Cheap connectivity probe: list one album and check the response shape."""
        status, body = await self._request("GET", "albums", params={"limit": "1"})
        if status >= 500:
            raise ProviderUnavailableError(f"music-rater returned {status} on connectivity probe")
        if status >= 400 or not isinstance(body, dict) or "items" not in body:
            raise MusicRaterError(
                f"Unexpected response from music-rater /albums?limit=1 "
                f"(status={status}): {_snippet(body)}"
            )

    async def resolve_by_uri(self, music_assistant_uri: str) -> int | None:
        """Look up an album_id by its MA URI. Returns None if no match."""
        return await self._first_album_id(
            params={"music_assistant_uri": music_assistant_uri, "limit": "1"}
        )

    async def resolve_by_search(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """
        Free-text album search (artist + album). Return up to `limit` candidate dicts.

        Unlike the exact URI resolve, a text search can surface a *different* album
        (a remaster, a live version, a same-titled record), so the caller must verify
        a candidate's artist/title against what was requested before queueing it.
        """
        return await self._get_album_items(params={"search": query, "limit": str(int(limit))})

    async def queue_lidarr(self, album_id: int) -> dict[str, Any]:
        """
        POST /albums/{id}/lidarr/queue — flips lidarr_manual_add + inline sync.

        Maps documented status codes to MA error types so the global error
        toast carries a useful message.
        """
        status, body = await self._request(
            "POST",
            f"albums/{int(album_id)}/lidarr/queue",
            timeout=_QUEUE_REQUEST_TIMEOUT,
        )
        if 200 <= status < 300:
            # The queue endpoint is documented to return a LidarrQueueResponse
            # JSON object with counters. Accept any 2xx (e.g. a 201 Created on a
            # POST that materialises a queue entry) so a successful sync isn't
            # reported as a failure. An empty body / non-dict shape means
            # music-rater violated its own contract; surface that to the user
            # rather than silently treating it as a benign no-op (which
            # _build_result would then re-raise as a misleading error anyway).
            if not isinstance(body, dict):
                raise MusicRaterError(
                    f"music-rater POST /albums/{album_id}/lidarr/queue returned "
                    f"status={status} with no JSON body: {_snippet(body) or '(empty)'}"
                )
            return body
        # Per music-rater's documented error surface.
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
        # Any other 5xx (500, 504, etc.) means music-rater itself is in trouble —
        # surface as provider-unavailable so MA's framework-level retry can kick
        # in. Without this, a generic 500 looks like a per-request app error
        # (MusicRaterError) and the retry never fires, mismatching the GET path
        # which raises ProviderUnavailableError on any 5xx.
        if status >= 500:
            raise ProviderUnavailableError(
                f"music-rater POST /albums/{album_id}/lidarr/queue failed "
                f"(status={status}): {_snippet(body)}"
            )
        raise MusicRaterError(
            f"POST /albums/{album_id}/lidarr/queue failed (status={status}): {_snippet(body)}"
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: ClientTimeout = _REQUEST_TIMEOUT,
        **kwargs: Any,
    ) -> tuple[int, Any]:
        """
        Issue a request and return (status, parsed-body-or-text).

        A malformed JSON body — Content-Type claims JSON but the bytes don't
        parse — is treated like a text response; callers already handle the
        "body isn't a dict" case and produce a useful error message.

        ``timeout`` defaults to the quick-read cap; the inline /lidarr/queue
        sync passes the larger ``_QUEUE_REQUEST_TIMEOUT``.
        """
        url = f"{self._base}/api/v1/{path.lstrip('/')}"
        async with self._session.request(
            method,
            url,
            headers=self._headers,
            ssl=self._verify_ssl,
            timeout=timeout,
            **kwargs,
        ) as resp:
            status = resp.status
            if status == 204 or resp.content_length == 0:
                return status, None
            ctype = resp.headers.get("Content-Type", "")
            if "application/json" in ctype:
                try:
                    return status, await resp.json()
                except ClientResponseError, ValueError:
                    return status, await resp.text()
            return status, await resp.text()

    async def _first_album_id(self, *, params: dict[str, str]) -> int | None:
        items = await self._get_album_items(params=params)
        if not items:
            return None
        first = items[0]
        if "id" not in first:
            raise MusicRaterError(f"music-rater returned malformed album item: {first!r}")
        try:
            return int(first["id"])
        except (TypeError, ValueError) as err:
            raise MusicRaterError(
                f"music-rater returned non-numeric album id: {first['id']!r}"
            ) from err

    async def _get_album_items(self, *, params: dict[str, str]) -> list[dict[str, Any]]:
        """GET /albums and return the (dict) items list, mapping errors to MA error types."""
        status, body = await self._request("GET", "albums", params=params)
        if status >= 500:
            raise ProviderUnavailableError(f"music-rater returned {status} from /albums")
        if status >= 400 or not isinstance(body, dict):
            raise MusicRaterError(f"GET /albums failed (status={status}): {_snippet(body)}")
        items = body.get("items") or []
        return [item for item in items if isinstance(item, dict)]
