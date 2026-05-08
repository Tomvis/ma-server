"""Thin async client for the Lidarr v1 REST API.

Auth is X-Api-Key header. Endpoints used:
- GET /api/v1/rootfolder
- GET /api/v1/qualityprofile
- GET /api/v1/metadataprofile
- GET /api/v1/system/status     (cheap connectivity probe)
- GET /api/v1/artist            (list existing artists)
- GET /api/v1/artist/lookup     (search MB or by name)
- POST /api/v1/artist           (add artist, optionally with addOptions)
- GET /api/v1/album?artistId=N  (list albums for an artist)
- PUT /api/v1/album/monitor     (set monitored flag for a list of album ids)
- POST /api/v1/command          (queue a command — RefreshArtist, AlbumSearch, …)
- GET /api/v1/command/{id}      (poll command status)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from music_assistant_models.errors import LoginFailed, ProviderUnavailableError

if TYPE_CHECKING:
    from aiohttp import ClientSession


class LidarrError(Exception):
    """Raised for unexpected Lidarr API responses."""


class LidarrClient:
    """Minimal async wrapper for the bits of Lidarr we need."""

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

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self._base}/api/v1/{path.lstrip('/')}"
        async with self._session.request(
            method,
            url,
            headers=self._headers,
            ssl=self._verify_ssl,
            **kwargs,
        ) as resp:
            if resp.status == 401:
                raise LoginFailed("Lidarr rejected the API key")
            if resp.status >= 500:
                raise ProviderUnavailableError(f"Lidarr returned {resp.status} for {method} {path}")
            if resp.status >= 400:
                body = await resp.text()
                raise LidarrError(f"{method} {path} -> {resp.status}: {body[:200]}")
            if resp.status == 204 or resp.content_length == 0:
                return None
            return await resp.json()

    async def system_status(self) -> dict[str, Any]:
        """Return system status — used as a connectivity / auth probe."""
        return cast("dict[str, Any]", await self._request("GET", "system/status"))

    async def list_root_folders(self) -> list[dict[str, Any]]:
        """List configured Lidarr root folders."""
        return cast("list[dict[str, Any]]", await self._request("GET", "rootfolder"))

    async def list_quality_profiles(self) -> list[dict[str, Any]]:
        """List configured Lidarr quality profiles."""
        return cast("list[dict[str, Any]]", await self._request("GET", "qualityprofile"))

    async def list_metadata_profiles(self) -> list[dict[str, Any]]:
        """List configured Lidarr metadata profiles."""
        return cast("list[dict[str, Any]]", await self._request("GET", "metadataprofile"))

    async def list_artists(self) -> list[dict[str, Any]]:
        """List artists currently in the Lidarr library."""
        return cast("list[dict[str, Any]]", await self._request("GET", "artist"))

    async def lookup_artist(self, term: str) -> list[dict[str, Any]]:
        """Lookup artists by name or `lidarr:<MBID>` shorthand. Returns raw remote results."""
        return cast(
            "list[dict[str, Any]]",
            await self._request("GET", "artist/lookup", params={"term": term}),
        )

    async def add_artist(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /artist — body is a Lidarr ArtistResource with addOptions."""
        return cast("dict[str, Any]", await self._request("POST", "artist", json=body))

    async def update_artist(self, body: dict[str, Any]) -> dict[str, Any]:
        """PUT /artist/{id} — re-save an existing ArtistResource."""
        return cast(
            "dict[str, Any]",
            await self._request("PUT", f"artist/{int(body['id'])}", json=body),
        )

    async def get_artist(self, artist_id: int) -> dict[str, Any]:
        """GET /artist/{id} — fetch a fresh artist record by Lidarr id."""
        return cast(
            "dict[str, Any]",
            await self._request("GET", f"artist/{int(artist_id)}"),
        )

    async def list_albums(self, artist_id: int) -> list[dict[str, Any]]:
        """List albums for an artist already known to Lidarr."""
        return cast(
            "list[dict[str, Any]]",
            await self._request("GET", "album", params={"artistId": str(artist_id)}),
        )

    async def set_albums_monitored(self, album_ids: list[int], monitored: bool = True) -> None:
        """Bulk-toggle the monitored flag via /album/monitor."""
        await self._request(
            "PUT",
            "album/monitor",
            json={"albumIds": album_ids, "monitored": monitored},
        )

    async def update_album(self, body: dict[str, Any]) -> dict[str, Any]:
        """PUT /album/{id} — re-save a full AlbumResource."""
        return cast(
            "dict[str, Any]",
            await self._request("PUT", f"album/{int(body['id'])}", json=body),
        )

    async def get_album(self, album_id: int) -> dict[str, Any]:
        """GET /album/{id} — fresh fetch of a single album."""
        return cast(
            "dict[str, Any]",
            await self._request("GET", f"album/{int(album_id)}"),
        )

    async def queue_command(self, name: str, **fields: Any) -> dict[str, Any]:
        """POST /command — queue a Lidarr command (RefreshArtist, AlbumSearch, …)."""
        return cast(
            "dict[str, Any]",
            await self._request("POST", "command", json={"name": name, **fields}),
        )

    async def get_command(self, command_id: int) -> dict[str, Any]:
        """GET /command/{id} — used to poll a queued command's status."""
        return cast(
            "dict[str, Any]",
            await self._request("GET", f"command/{command_id}"),
        )
