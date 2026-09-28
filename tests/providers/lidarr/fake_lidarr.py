"""In-memory stand-in for LidarrClient, modelling the Lidarr behaviours the add flow depends on."""

from __future__ import annotations

import copy
from typing import Any

from music_assistant.providers.lidarr.client import LidarrError

ROOTS = [
    {
        "id": 2,
        "path": "/data/media/music/tom/artists",
        "defaultQualityProfileId": 1,
        "defaultMetadataProfileId": 1,
    },
    {
        "id": 3,
        "path": "/data/media/music/lera/artists",
        "defaultQualityProfileId": 1,
        "defaultMetadataProfileId": 3,
    },
]
METADATA_PROFILES = [{"id": 1, "name": "Standard"}, {"id": 3, "name": "Fanatic"}]


class FakeLidarr:
    """
    Fake Lidarr.

    - ``catalog`` maps an artist MBID to the albums Lidarr would load for it on refresh.
    - A newly added artist comes back ``monitored=False`` (Lidarr zeroes it when
      ``addOptions.monitor == "none"``) and has no albums until ``hydrate_after``
      album listings have happened, mimicking the async metadata refresh.
    """

    def __init__(self, *, hydrate_after: int = 1) -> None:
        """Start with no artists; tests add them via ``catalog`` / ``existing``."""
        self.roots = copy.deepcopy(ROOTS)
        self.catalog: dict[str, list[dict[str, Any]]] = {}
        self.lookup_names: dict[str, str] = {}
        self.artists: list[dict[str, Any]] = []
        self.albums: dict[int, list[dict[str, Any]]] = {}
        self.commands: list[dict[str, Any]] = []
        self.added: list[dict[str, Any]] = []
        self.hydrate_after = hydrate_after
        self._list_calls: dict[int, int] = {}
        self._next_id = 100

    # ----- test setup helpers -----

    def existing(self, mbid: str, name: str, root: str, *, monitored: bool = True) -> int:
        """Put an artist in Lidarr already, with its catalog loaded."""
        artist_id = self._new_id()
        self.artists.append(
            {
                "id": artist_id,
                "foreignArtistId": mbid,
                "artistName": name,
                "rootFolderPath": root,
                "monitored": monitored,
                "metadataProfileId": 1,
            }
        )
        self.albums[artist_id] = self._load(mbid, artist_id)
        self._list_calls[artist_id] = self.hydrate_after
        return artist_id

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _load(self, mbid: str, artist_id: int) -> list[dict[str, Any]]:
        return [
            {**copy.deepcopy(a), "id": self._new_id(), "artistId": artist_id, "monitored": False}
            for a in self.catalog.get(mbid, [])
        ]

    def album(self, album_id: int) -> dict[str, Any]:
        """Return the stored album record by id."""
        for albums in self.albums.values():
            for a in albums:
                if a["id"] == album_id:
                    return a
        raise LidarrError(f"no album {album_id}")

    # ----- LidarrClient surface -----

    async def system_status(self) -> dict[str, Any]:
        """Report a running Lidarr."""
        return {"version": "3.1.5"}

    async def list_root_folders(self) -> list[dict[str, Any]]:
        """Return the root folders."""
        return copy.deepcopy(self.roots)

    async def list_metadata_profiles(self) -> list[dict[str, Any]]:
        """Return the metadata profiles."""
        return copy.deepcopy(METADATA_PROFILES)

    async def list_artists(self) -> list[dict[str, Any]]:
        """Return the artists in the library."""
        return copy.deepcopy(self.artists)

    async def get_artist(self, artist_id: int) -> dict[str, Any]:
        """Return one artist."""
        return copy.deepcopy(next(a for a in self.artists if a["id"] == artist_id))

    async def update_artist(self, body: dict[str, Any]) -> dict[str, Any]:
        """Replace an artist record."""
        for i, a in enumerate(self.artists):
            if a["id"] == body["id"]:
                self.artists[i] = copy.deepcopy(body)
                return copy.deepcopy(body)
        raise LidarrError("no such artist")

    async def lookup_artist(self, term: str) -> list[dict[str, Any]]:
        """Resolve by ``lidarr:<mbid>`` or exact name, from ``catalog``/``lookup_names``."""
        if term.startswith("lidarr:"):
            mbid = term.removeprefix("lidarr:")
            if mbid in self.catalog:
                return [{"foreignArtistId": mbid, "artistName": self.lookup_names.get(mbid, "?")}]
            return []
        return [
            {"foreignArtistId": mbid, "artistName": name}
            for mbid, name in self.lookup_names.items()
            if name.casefold() == term.casefold()
        ]

    async def add_artist(self, body: dict[str, Any]) -> dict[str, Any]:
        """Add an artist; it comes back unmonitored with no albums loaded yet."""
        self.added.append(copy.deepcopy(body))
        artist_id = self._new_id()
        record = {**copy.deepcopy(body), "id": artist_id, "monitored": False}
        record.pop("addOptions", None)
        self.artists.append(record)
        self.albums[artist_id] = self._load(body["foreignArtistId"], artist_id)
        self._list_calls[artist_id] = 0
        return copy.deepcopy(record)

    async def list_albums(self, artist_id: int) -> list[dict[str, Any]]:
        """Return the artist's albums once hydrated."""
        self._list_calls[artist_id] = self._list_calls.get(artist_id, 0) + 1
        if self._list_calls[artist_id] <= self.hydrate_after:
            return []
        return copy.deepcopy(self.albums.get(artist_id, []))

    async def get_album(self, album_id: int) -> dict[str, Any]:
        """Return one album."""
        return copy.deepcopy(self.album(album_id))

    async def set_albums_monitored(self, album_ids: list[int], monitored: bool = True) -> None:
        """Set the monitored flag."""
        for album_id in album_ids:
            self.album(album_id)["monitored"] = monitored

    async def update_album(self, body: dict[str, Any]) -> dict[str, Any]:
        """Update an album record."""
        self.album(body["id"]).update(copy.deepcopy(body))
        return copy.deepcopy(body)

    async def queue_command(self, name: str, **fields: Any) -> dict[str, Any]:
        """Record a command; RefreshArtist (re)loads the catalog immediately."""
        command = {"id": self._new_id(), "name": name, **fields}
        self.commands.append(command)
        if name == "RefreshArtist":
            for artist_id in fields.get("artistIds", []):
                self._list_calls[artist_id] = self.hydrate_after
                artist = next(a for a in self.artists if a["id"] == artist_id)
                self.albums[artist_id] = self._load(artist["foreignArtistId"], artist_id)
        return command

    async def get_command(self, command_id: int) -> dict[str, Any]:
        """Report every command as completed."""
        return {"id": command_id, "status": "completed"}
