"""
Thin async client for the Lidarr v1 REST API.

Auth is the X-Api-Key header. Only the endpoints the "Add to Lidarr" flow needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from aiohttp import ClientTimeout
from music_assistant_models.errors import (
    InvalidDataError,
    LoginFailed,
    ProviderUnavailableError,
)

if TYPE_CHECKING:
    from aiohttp import ClientSession

# Every call is a quick read or a single write; Lidarr's slow work (metadata refresh,
# searches) runs as queued commands, never inside a request.
_REQUEST_TIMEOUT = ClientTimeout(total=30)


class LidarrError(InvalidDataError):
    """
    Raised for unexpected or refused Lidarr API responses.

    An InvalidDataError so the frontend's error toast renders the message (it filters on
    MusicAssistantError subclasses) instead of a generic failure.
    """


class LidarrClient:
    """Minimal async wrapper for the Lidarr endpoints we need."""

    def __init__(
        self,
        url: str,
        api_key: str,
        session: ClientSession,
        *,
        verify_ssl: bool = True,
    ) -> None:
        """Build a client bound to one Lidarr instance."""
        self._base = url.rstrip("/")
        self._headers = {"X-Api-Key": api_key, "Accept": "application/json"}
        self._session = session
        self._verify_ssl = verify_ssl

    async def system_status(self) -> dict[str, Any]:
        """Return system status -- the connectivity / auth probe."""
        return cast("dict[str, Any]", await self._request("GET", "system/status"))

    async def list_root_folders(self) -> list[dict[str, Any]]:
        """List root folders, each carrying its default quality/metadata profile ids."""
        return cast("list[dict[str, Any]]", await self._request("GET", "rootfolder"))

    async def list_metadata_profiles(self) -> list[dict[str, Any]]:
        """List metadata profiles (only used to name one in an error)."""
        return cast("list[dict[str, Any]]", await self._request("GET", "metadataprofile"))

    async def list_artists(self) -> list[dict[str, Any]]:
        """List artists currently in the Lidarr library."""
        return cast("list[dict[str, Any]]", await self._request("GET", "artist"))

    async def get_artist(self, artist_id: int) -> dict[str, Any]:
        """Fetch one artist by Lidarr id."""
        return cast("dict[str, Any]", await self._request("GET", f"artist/{int(artist_id)}"))

    async def lookup_artist(self, term: str) -> list[dict[str, Any]]:
        """Look up artists by name or ``lidarr:<MBID>``; returns remote (not yet added) results."""
        return cast(
            "list[dict[str, Any]]",
            await self._request("GET", "artist/lookup", params={"term": term}),
        )

    async def add_artist(self, body: dict[str, Any]) -> dict[str, Any]:
        """Add an artist (an ArtistResource with addOptions)."""
        return cast("dict[str, Any]", await self._request("POST", "artist", json=body))

    async def update_artist(self, body: dict[str, Any]) -> dict[str, Any]:
        """Re-save an existing ArtistResource."""
        return cast(
            "dict[str, Any]", await self._request("PUT", f"artist/{int(body['id'])}", json=body)
        )

    async def list_albums(self, artist_id: int) -> list[dict[str, Any]]:
        """List the albums Lidarr has loaded for an artist."""
        return cast(
            "list[dict[str, Any]]",
            await self._request("GET", "album", params={"artistId": str(int(artist_id))}),
        )

    async def get_album(self, album_id: int) -> dict[str, Any]:
        """Fetch one album by Lidarr id."""
        return cast("dict[str, Any]", await self._request("GET", f"album/{int(album_id)}"))

    async def set_albums_monitored(self, album_ids: list[int], monitored: bool = True) -> None:
        """Bulk-toggle the monitored flag."""
        await self._request(
            "PUT", "album/monitor", json={"albumIds": album_ids, "monitored": monitored}
        )

    async def update_album(self, body: dict[str, Any]) -> dict[str, Any]:
        """Re-save a full AlbumResource (the UI's own route for the monitor toggle)."""
        return cast(
            "dict[str, Any]", await self._request("PUT", f"album/{int(body['id'])}", json=body)
        )

    async def queue_command(self, name: str, **fields: Any) -> dict[str, Any]:
        """Queue a Lidarr command (RefreshArtist, AlbumSearch, ...)."""
        return cast(
            "dict[str, Any]", await self._request("POST", "command", json={"name": name, **fields})
        )

    async def get_command(self, command_id: int) -> dict[str, Any]:
        """Poll a queued command's status."""
        return cast("dict[str, Any]", await self._request("GET", f"command/{int(command_id)}"))

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self._base}/api/v1/{path.lstrip('/')}"
        async with self._session.request(
            method,
            url,
            headers=self._headers,
            ssl=self._verify_ssl,
            timeout=_REQUEST_TIMEOUT,
            **kwargs,
        ) as resp:
            if resp.status == 401:
                raise LoginFailed("Lidarr rejected the API key")
            if resp.status >= 500:
                raise ProviderUnavailableError(f"Lidarr returned {resp.status} for {method} {path}")
            if resp.status >= 400:
                body = await resp.text()
                raise LidarrError(f"Lidarr {method} {path} -> {resp.status}: {body[:300]}")
            if resp.status == 204 or resp.content_length == 0:
                return None
            return await resp.json()
