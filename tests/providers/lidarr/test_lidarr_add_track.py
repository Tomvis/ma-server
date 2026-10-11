"""Tests for wanting a single track in Lidarr (MUSIC-76)."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from music_assistant.providers.lidarr import lidarr_add
from music_assistant.providers.lidarr.client import LidarrError
from music_assistant.providers.lidarr.lidarr_add import AlbumRequest, TrackRequest, add_track

from .fake_lidarr import FakeLidarr

TOM = "/data/media/music/tom/artists"
LERA = "/data/media/music/lera/artists"
OPETH = "c14b4180-dc87-481e-b17a-64e4150f90f6"
RG = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
LOG = logging.getLogger("test")
BLACKWATER_PARK = [
    {
        "title": "The Leper Affinity",
        "mediumNumber": 1,
        "trackNumber": "1",
        "foreignRecordingId": "r1",
    },
    {"title": "Bleak", "mediumNumber": 1, "trackNumber": "2", "foreignRecordingId": "r2"},
    {"title": "Harvest", "mediumNumber": 1, "trackNumber": "3", "foreignRecordingId": "r3"},
    {"title": "Intro", "mediumNumber": 2, "trackNumber": "1", "foreignRecordingId": "r4"},
    {"title": "Intro", "mediumNumber": 2, "trackNumber": "2", "foreignRecordingId": "r5"},
]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lidarr_add, "POLL_INTERVAL", 0)


def _lidarr() -> FakeLidarr:
    lidarr = FakeLidarr(settle_after=2)
    lidarr.lookup_names[OPETH] = "Opeth"
    lidarr.catalog[OPETH] = [
        {
            "title": "Blackwater Park",
            "foreignAlbumId": RG,
            "releases": [],
            "tracks": BLACKWATER_PARK,
        },
        {"title": "Damnation", "foreignAlbumId": "rg-damnation", "releases": [], "tracks": []},
    ]
    return lidarr


def _request(track_name: str = "Bleak", **overrides: Any) -> TrackRequest:
    fields: dict[str, Any] = {
        "recording_mbid": None,
        "disc_number": None,
        "track_number": None,
    }
    fields.update(overrides)
    return TrackRequest(
        album=AlbumRequest(
            artist_mbid=OPETH,
            artist_name="Opeth",
            album_name="Blackwater Park",
            release_group_mbid=RG,
            release_mbid=None,
        ),
        track_name=track_name,
        **fields,
    )


def _album(lidarr: FakeLidarr, artist_id: int) -> dict[str, Any]:
    return next(a for a in lidarr.albums[artist_id] if a["title"] == "Blackwater Park")


def _wanted(album: dict[str, Any]) -> list[str]:
    return [t["foreignRecordingId"] for t in album["tracks"] if t["monitored"]]


def _searches(lidarr: FakeLidarr) -> list[dict[str, Any]]:
    return [c for c in lidarr.commands if c["name"] == "AlbumSearch"]


async def test_track_of_an_unwanted_album_is_wanted_alone_and_searched() -> None:
    """The album gets monitored for that one track only, and searched."""
    lidarr = _lidarr()
    artist_id = lidarr.existing(OPETH, "Opeth", TOM)

    result = await add_track(lidarr, _request(), root_folder=TOM, logger=LOG)  # type: ignore[arg-type]

    album = _album(lidarr, artist_id)
    assert album["monitored"] is True
    assert _wanted(album) == ["r2"]
    assert [s["albumIds"] for s in _searches(lidarr)] == [[album["id"]]]
    assert result == {
        "artist_name": "Opeth",
        "album_name": "Blackwater Park",
        "track_name": "Bleak",
        "artist_added": False,
        "already_wanted": False,
    }


async def test_new_artist_gets_only_the_track_wanted_after_lidarrs_refresh() -> None:
    """A new artist's add-time refresh must not undo the track's album being monitored."""
    lidarr = _lidarr()

    result = await add_track(lidarr, _request(), root_folder=LERA, logger=LOG)  # type: ignore[arg-type]

    [artist] = lidarr.artists
    assert artist["rootFolderPath"] == LERA
    assert artist["monitored"] is True
    album = _album(lidarr, artist["id"])
    assert album["monitored"] is True
    assert _wanted(album) == ["r2"]
    assert result["artist_added"] is True


async def test_recording_mbid_wins_over_the_title() -> None:
    """MusicBrainz's recording id is exact; a differently spelled title doesn't matter."""
    lidarr = _lidarr()
    artist_id = lidarr.existing(OPETH, "Opeth", TOM)

    await add_track(
        lidarr,  # type: ignore[arg-type]
        _request("Harvest (2021 Remaster)", recording_mbid="r3"),
        root_folder=TOM,
        logger=LOG,
    )

    assert _wanted(_album(lidarr, artist_id)) == ["r3"]


async def test_title_with_an_edition_tag_still_matches() -> None:
    """A streaming service's '(Remastered)' suffix is not a different song."""
    lidarr = _lidarr()
    artist_id = lidarr.existing(OPETH, "Opeth", TOM)

    await add_track(
        lidarr,  # type: ignore[arg-type]
        _request("Harvest (Remastered)"),
        root_folder=TOM,
        logger=LOG,
    )

    assert _wanted(_album(lidarr, artist_id)) == ["r3"]


async def test_repeated_title_is_told_apart_by_disc_and_number() -> None:
    """Two tracks share a title; the requested position picks the right one."""
    lidarr = _lidarr()
    artist_id = lidarr.existing(OPETH, "Opeth", TOM)

    await add_track(
        lidarr,  # type: ignore[arg-type]
        _request("Intro", disc_number=2, track_number=2),
        root_folder=TOM,
        logger=LOG,
    )

    assert _wanted(_album(lidarr, artist_id)) == ["r5"]


async def test_repeated_title_without_a_position_is_refused() -> None:
    """Guessing between two same-titled tracks would download the wrong one."""
    lidarr = _lidarr()
    artist_id = lidarr.existing(OPETH, "Opeth", TOM)

    with pytest.raises(LidarrError, match="Intro"):
        await add_track(lidarr, _request("Intro"), root_folder=TOM, logger=LOG)  # type: ignore[arg-type]
    assert _album(lidarr, artist_id)["monitored"] is False


async def test_position_alone_never_picks_a_different_song() -> None:
    """MA's edition may order tracks differently; a title mismatch at the position is refused."""
    lidarr = _lidarr()
    artist_id = lidarr.existing(OPETH, "Opeth", TOM)

    with pytest.raises(LidarrError, match="Patterns in the Ivy"):
        await add_track(
            lidarr,  # type: ignore[arg-type]
            _request("Patterns in the Ivy", disc_number=1, track_number=2),
            root_folder=TOM,
            logger=LOG,
        )
    assert _album(lidarr, artist_id)["monitored"] is False
    assert _searches(lidarr) == []


async def test_track_of_a_fully_wanted_album_is_already_wanted() -> None:
    """A monitored album wants every track by default: nothing changes, nothing is searched."""
    lidarr = _lidarr()
    artist_id = lidarr.existing(OPETH, "Opeth", TOM)
    _album(lidarr, artist_id)["monitored"] = True

    result = await add_track(lidarr, _request(), root_folder=TOM, logger=LOG)  # type: ignore[arg-type]

    assert result["already_wanted"] is True
    assert _wanted(_album(lidarr, artist_id)) == ["r1", "r2", "r3", "r4", "r5"]
    assert _searches(lidarr) == []


async def test_unwanted_track_of_a_monitored_album_is_added_to_the_wanted_ones() -> None:
    """A partially wanted album keeps its other wanted tracks."""
    lidarr = _lidarr()
    artist_id = lidarr.existing(OPETH, "Opeth", TOM)
    album = _album(lidarr, artist_id)
    album["monitored"] = True
    for track in album["tracks"]:
        track["monitored"] = track["foreignRecordingId"] == "r1"

    result = await add_track(lidarr, _request(), root_folder=TOM, logger=LOG)  # type: ignore[arg-type]

    assert result["already_wanted"] is False
    assert _wanted(_album(lidarr, artist_id)) == ["r1", "r2"]
    assert len(_searches(lidarr)) == 1


async def test_track_already_on_disk_is_wanted_but_not_searched() -> None:
    """Lidarr already has the file; wanting it is bookkeeping, not a download."""
    lidarr = _lidarr()
    artist_id = lidarr.existing(OPETH, "Opeth", TOM)
    next(t for t in _album(lidarr, artist_id)["tracks"] if t["title"] == "Bleak")["hasFile"] = True

    await add_track(lidarr, _request(), root_folder=TOM, logger=LOG)  # type: ignore[arg-type]

    assert _wanted(_album(lidarr, artist_id)) == ["r2"]
    assert _searches(lidarr) == []


async def test_track_missing_from_the_monitored_release_names_the_fix() -> None:
    """A bonus track absent from Lidarr's chosen release points at switching the release."""
    lidarr = _lidarr()
    artist_id = lidarr.existing(OPETH, "Opeth", TOM)

    with pytest.raises(LidarrError, match="release"):
        await add_track(
            lidarr,  # type: ignore[arg-type]
            _request("Still Day Beneath the Sun"),
            root_folder=TOM,
            logger=LOG,
        )
    assert _album(lidarr, artist_id)["monitored"] is False
