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
    - A newly added artist comes back ``monitored=False`` with ``addOptions`` set and
      Lidarr's own new-artist RefreshArtist running. That refresh finishes after
      ``settle_after`` command listings: it loads the albums, then applies
      ``monitor=none`` -- unmonitoring EVERY album and writing back the artist as it was
      when the refresh started -- and clears ``addOptions``. Anything the flow monitors
      before that is undone, like the real race.
    """

    def __init__(self, *, settle_after: int = 2) -> None:
        """Start with no artists; tests add them via ``catalog`` / ``existing``."""
        self.roots = copy.deepcopy(ROOTS)
        self.catalog: dict[str, list[dict[str, Any]]] = {}
        self.lookup_names: dict[str, str] = {}
        self.artists: list[dict[str, Any]] = []
        self.albums: dict[int, list[dict[str, Any]]] = {}
        self.commands: list[dict[str, Any]] = []
        self.added: list[dict[str, Any]] = []
        self.settle_after = settle_after
        self._pending: dict[int, dict[str, Any]] = {}
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
        self._tick()
        return {"version": "3.1.5"}

    async def list_root_folders(self) -> list[dict[str, Any]]:
        """Return the root folders."""
        self._tick()
        return copy.deepcopy(self.roots)

    async def list_metadata_profiles(self) -> list[dict[str, Any]]:
        """Return the metadata profiles."""
        self._tick()
        return copy.deepcopy(METADATA_PROFILES)

    async def list_artists(self) -> list[dict[str, Any]]:
        """Return the artists in the library."""
        self._tick()
        return copy.deepcopy(self.artists)

    async def get_artist(self, artist_id: int) -> dict[str, Any]:
        """Return one artist."""
        self._tick()
        return copy.deepcopy(next(a for a in self.artists if a["id"] == artist_id))

    async def update_artist(self, body: dict[str, Any]) -> dict[str, Any]:
        """Replace an artist record."""
        self._tick()
        for i, a in enumerate(self.artists):
            if a["id"] == body["id"]:
                self.artists[i] = copy.deepcopy(body)
                return copy.deepcopy(body)
        raise LidarrError("no such artist")

    async def lookup_artist(self, term: str) -> list[dict[str, Any]]:
        """Resolve by ``lidarr:<mbid>`` or exact name, from ``catalog``/``lookup_names``."""
        self._tick()
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
        self._tick()
        self.added.append(copy.deepcopy(body))
        artist_id = self._new_id()
        record = {**copy.deepcopy(body), "id": artist_id, "monitored": False}
        self.artists.append(record)
        self.albums[artist_id] = []
        command = {
            "id": self._new_id(),
            "name": "RefreshArtist",
            "status": "started",
            "body": {"artistIds": [artist_id], "isNewArtist": True},
            "_snapshot": copy.deepcopy(record),
            "_polls": 0,
        }
        self.commands.append(command)
        self._pending[artist_id] = command
        return copy.deepcopy(record)

    def _tick(self) -> None:
        """Let time pass: advance Lidarr's own new-artist refreshes by one API call."""
        for artist_id, command in list(self._pending.items()):
            command["_polls"] += 1
            if command["_polls"] >= self.settle_after:
                self._settle(artist_id, command)

    async def list_commands(self) -> list[dict[str, Any]]:
        """List every command."""
        self._tick()
        return [
            {k: v for k, v in copy.deepcopy(c).items() if not k.startswith("_")}
            for c in self.commands
        ]

    def _settle(self, artist_id: int, command: dict[str, Any]) -> None:
        artist = next(a for a in self.artists if a["id"] == artist_id)
        loaded = self._load(artist["foreignArtistId"], artist_id)
        for album in loaded:
            album["monitored"] = False
        self.albums[artist_id] = loaded
        snapshot = {**command["_snapshot"]}
        snapshot.pop("addOptions", None)
        self.artists[self.artists.index(artist)] = snapshot
        command["status"] = "completed"
        del self._pending[artist_id]

    async def list_albums(self, artist_id: int) -> list[dict[str, Any]]:
        """Return the artist's albums once hydrated."""
        self._tick()
        return copy.deepcopy(self.albums.get(artist_id, []))

    async def get_album(self, album_id: int) -> dict[str, Any]:
        """Return one album."""
        self._tick()
        return copy.deepcopy(self.album(album_id))

    async def set_albums_monitored(self, album_ids: list[int], monitored: bool = True) -> None:
        """Set the monitored flag."""
        self._tick()
        for album_id in album_ids:
            self.album(album_id)["monitored"] = monitored

    async def update_album(self, body: dict[str, Any]) -> dict[str, Any]:
        """Update an album record."""
        self._tick()
        self.album(body["id"]).update(copy.deepcopy(body))
        return copy.deepcopy(body)

    async def queue_command(self, name: str, **fields: Any) -> dict[str, Any]:
        """Record a command; RefreshArtist (re)loads the catalog immediately."""
        self._tick()
        command = {"id": self._new_id(), "name": name, **fields}
        self.commands.append(command)
        if name == "RefreshArtist":
            # a plain (not new-artist) refresh: reloads the catalog, keeps monitoring
            for artist_id in fields.get("artistIds", []):
                artist = next(a for a in self.artists if a["id"] == artist_id)
                known = {a["foreignAlbumId"]: a for a in self.albums.get(artist_id, [])}
                self.albums[artist_id] = [
                    known.get(a["foreignAlbumId"], a)
                    for a in self._load(artist["foreignArtistId"], artist_id)
                ]
        return command

    async def get_command(self, command_id: int) -> dict[str, Any]:
        """Report every command as completed."""
        self._tick()
        return {"id": command_id, "status": "completed"}
