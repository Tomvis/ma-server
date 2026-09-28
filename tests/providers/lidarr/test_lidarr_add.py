"""Tests for the direct Lidarr add flow (MEDIA-1: music-rater no longer adds to Lidarr)."""

from __future__ import annotations

import logging

import pytest

from music_assistant.providers.lidarr import lidarr_add
from music_assistant.providers.lidarr.client import LidarrError
from music_assistant.providers.lidarr.lidarr_add import AlbumRequest, add_album

from .fake_lidarr import FakeLidarr

TOM = "/data/media/music/tom/artists"
LERA = "/data/media/music/lera/artists"
OPETH = "c14b4180-dc87-481e-b17a-64e4150f90f6"
RG = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
REL = "11111111-2222-3333-4444-555555555555"
LOG = logging.getLogger("test")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lidarr_add, "POLL_INTERVAL", 0)


def _lidarr() -> FakeLidarr:
    lidarr = FakeLidarr(settle_after=2)
    lidarr.lookup_names[OPETH] = "Opeth"
    lidarr.catalog[OPETH] = [
        {"title": "Blackwater Park", "foreignAlbumId": RG, "releases": [{"foreignReleaseId": REL}]},
        {"title": "Damnation", "foreignAlbumId": "rg-damnation", "releases": []},
    ]
    return lidarr


def _request(**overrides: str | None) -> AlbumRequest:
    fields: dict[str, str | None] = {
        "artist_mbid": OPETH,
        "artist_name": "Opeth",
        "album_name": "Blackwater Park",
        "release_group_mbid": RG,
        "release_mbid": None,
    }
    fields.update(overrides)
    return AlbumRequest(**fields)  # type: ignore[arg-type]


async def test_new_artist_is_added_to_the_given_root_with_its_default_profiles() -> None:
    """A new artist lands in the caller's root, monitoring nothing but the requested album."""
    lidarr = _lidarr()

    result = await add_album(lidarr, _request(), root_folder=LERA, logger=LOG)  # type: ignore[arg-type]

    [body] = lidarr.added
    assert body["rootFolderPath"] == LERA
    assert body["qualityProfileId"] == 1
    assert body["metadataProfileId"] == 3  # lera root's default, not tom's
    assert body["monitorNewItems"] == "none"
    assert body["addOptions"] == {"monitor": "none", "searchForMissingAlbums": False}
    [artist] = lidarr.artists
    assert artist["monitored"] is True  # flipped back after Lidarr zeroed it
    monitored = [a["title"] for a in lidarr.albums[artist["id"]] if a["monitored"]]
    assert monitored == ["Blackwater Park"]
    assert [c["name"] for c in lidarr.commands if c["name"] == "AlbumSearch"] == ["AlbumSearch"]
    assert result == {
        "artist_name": "Opeth",
        "album_name": "Blackwater Park",
        "artist_added": True,
        "album_monitored": True,
        "already_monitored": False,
    }


async def test_existing_artist_under_another_root_is_used_in_place() -> None:
    """An artist already in Lidarr is never re-added or moved; its album is monitored there."""
    lidarr = _lidarr()
    artist_id = lidarr.existing(OPETH, "Opeth", TOM)

    result = await add_album(lidarr, _request(), root_folder=LERA, logger=LOG)  # type: ignore[arg-type]

    assert lidarr.added == []
    assert lidarr.artists[0]["rootFolderPath"] == TOM
    assert [a["title"] for a in lidarr.albums[artist_id] if a["monitored"]] == ["Blackwater Park"]
    assert result["artist_added"] is False
    assert result["album_monitored"] is True


async def test_already_monitored_album_is_reported_and_not_searched_again() -> None:
    """A repeat add is idempotent: reported as already monitored, no second search."""
    lidarr = _lidarr()
    artist_id = lidarr.existing(OPETH, "Opeth", TOM)
    lidarr.albums[artist_id][0]["monitored"] = True

    result = await add_album(lidarr, _request(), root_folder=TOM, logger=LOG)  # type: ignore[arg-type]

    assert result["already_monitored"] is True
    assert result["album_monitored"] is True
    assert not any(c["name"] == "AlbumSearch" for c in lidarr.commands)


async def test_unmonitored_existing_artist_is_monitored() -> None:
    """Lidarr never searches albums of an unmonitored artist, so the artist is switched on."""
    lidarr = _lidarr()
    lidarr.existing(OPETH, "Opeth", TOM, monitored=False)

    await add_album(lidarr, _request(), root_folder=TOM, logger=LOG)  # type: ignore[arg-type]

    assert lidarr.artists[0]["monitored"] is True


async def test_album_matches_by_release_mbid_when_no_release_group_is_known() -> None:
    """MA's MB_ALBUM is a *release* id; Lidarr keys albums by release group."""
    lidarr = _lidarr()
    artist_id = lidarr.existing(OPETH, "Opeth", TOM)

    await add_album(
        lidarr,  # type: ignore[arg-type]
        _request(release_group_mbid=None, release_mbid=REL, album_name="Something Else"),
        root_folder=TOM,
        logger=LOG,
    )

    assert [a["title"] for a in lidarr.albums[artist_id] if a["monitored"]] == ["Blackwater Park"]


async def test_album_matches_by_normalized_title_without_any_mbid() -> None:
    """Edition parentheticals and punctuation don't defeat the title fallback."""
    lidarr = _lidarr()
    artist_id = lidarr.existing(OPETH, "Opeth", TOM)

    await add_album(
        lidarr,  # type: ignore[arg-type]
        _request(release_group_mbid=None, album_name="Damnation! (Deluxe Edition)"),
        root_folder=TOM,
        logger=LOG,
    )

    assert [a["title"] for a in lidarr.albums[artist_id] if a["monitored"]] == ["Damnation"]


async def test_existing_artist_is_refreshed_when_the_album_is_not_loaded_yet() -> None:
    """A release newer than Lidarr's last refresh is found after a RefreshArtist."""
    lidarr = _lidarr()
    artist_id = lidarr.existing(OPETH, "Opeth", TOM)
    lidarr.albums[artist_id] = []  # loaded before this album existed

    await add_album(lidarr, _request(), root_folder=TOM, logger=LOG)  # type: ignore[arg-type]

    assert any(c["name"] == "RefreshArtist" for c in lidarr.commands)
    assert [a["title"] for a in lidarr.albums[artist_id] if a["monitored"]] == ["Blackwater Park"]


@pytest.mark.parametrize("settle_after", range(1, 16))
async def test_nothing_is_monitored_until_lidarrs_own_new_artist_refresh_is_done(
    settle_after: int,
) -> None:
    """
    Lidarr's add-time refresh unmonitors every album when it finishes.

    Monitoring before it settles is silently undone while the toast says "monitored";
    parametrized over when that refresh lands relative to the flow's own calls.
    """
    lidarr = FakeLidarr(settle_after=settle_after)
    lidarr.lookup_names[OPETH] = "Opeth"
    lidarr.catalog[OPETH] = [{"title": "Blackwater Park", "foreignAlbumId": RG, "releases": []}]

    await add_album(lidarr, _request(), root_folder=TOM, logger=LOG)  # type: ignore[arg-type]

    [artist] = lidarr.artists
    assert artist["monitored"] is True
    assert [a["title"] for a in lidarr.albums[artist["id"]] if a["monitored"]] == [
        "Blackwater Park"
    ]


async def test_a_new_artist_that_never_settles_is_a_retryable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stuck add-time refresh fails with "try again", never with a false success."""
    monkeypatch.setattr(lidarr_add, "SETTLE_TIMEOUT", 0)
    lidarr = FakeLidarr(settle_after=1000)
    lidarr.lookup_names[OPETH] = "Opeth"
    lidarr.catalog[OPETH] = [{"title": "Blackwater Park", "foreignAlbumId": RG, "releases": []}]

    with pytest.raises(LidarrError, match="again"):
        await add_album(lidarr, _request(), root_folder=TOM, logger=LOG)  # type: ignore[arg-type]


async def test_known_mbid_is_not_title_matched_before_a_refresh() -> None:
    """
    A new album whose same-titled single is already loaded must not match the single.

    With an MBID known, the stale list is searched by MBID only; the refresh loads the
    album and the MBID then matches it.
    """
    lidarr = _lidarr()
    lidarr.catalog[OPETH].append({"title": "Ghost", "foreignAlbumId": "rg-ghost-lp"})
    artist_id = lidarr.existing(OPETH, "Opeth", TOM)
    lidarr.albums[artist_id] = [
        {
            "id": 900,
            "artistId": artist_id,
            "title": "Ghost",
            "foreignAlbumId": "rg-ghost-single",
            "monitored": False,
            "releases": [],
        }
    ]

    await add_album(
        lidarr,  # type: ignore[arg-type]
        _request(release_group_mbid="rg-ghost-lp", album_name="Ghost"),
        root_folder=TOM,
        logger=LOG,
    )

    monitored = [a["foreignAlbumId"] for a in lidarr.albums[artist_id] if a["monitored"]]
    assert monitored == ["rg-ghost-lp"]


async def test_non_latin_titles_match_themselves_not_the_first_album() -> None:
    """Titles in any script normalize to themselves, not to an empty string."""
    lidarr = _lidarr()
    lidarr.catalog[OPETH] = [
        {"title": "Альбом", "foreignAlbumId": "rg-ru", "releases": []},
        {"title": "אלבום", "foreignAlbumId": "rg-he", "releases": []},
    ]
    artist_id = lidarr.existing(OPETH, "Opeth", TOM)

    await add_album(
        lidarr,  # type: ignore[arg-type]
        _request(release_group_mbid=None, album_name="אלבום"),
        root_folder=TOM,
        logger=LOG,
    )

    assert [a["title"] for a in lidarr.albums[artist_id] if a["monitored"]] == ["אלבום"]


async def test_exact_title_wins_over_an_edition_stripped_match() -> None:
    """An exact title ("Fearless (Taylor's Version)") wins over the stripped one."""
    lidarr = _lidarr()
    lidarr.catalog[OPETH] = [
        {"title": "Fearless", "foreignAlbumId": "rg-2008", "releases": []},
        {"title": "Fearless (Taylor's Version)", "foreignAlbumId": "rg-2021", "releases": []},
    ]
    artist_id = lidarr.existing(OPETH, "Opeth", TOM)

    await add_album(
        lidarr,  # type: ignore[arg-type]
        _request(release_group_mbid=None, album_name="Fearless (Taylor's Version)"),
        root_folder=TOM,
        logger=LOG,
    )

    assert [a["title"] for a in lidarr.albums[artist_id] if a["monitored"]] == [
        "Fearless (Taylor's Version)"
    ]


async def test_album_missing_from_the_catalog_names_the_metadata_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A release the metadata profile filters out (e.g. a Single under Standard) is explained."""
    monkeypatch.setattr(lidarr_add, "DISCOGRAPHY_TIMEOUT", 0)
    lidarr = _lidarr()
    lidarr.existing(OPETH, "Opeth", TOM)

    with pytest.raises(LidarrError, match="Standard"):
        await add_album(
            lidarr,  # type: ignore[arg-type]
            _request(release_group_mbid="rg-single", album_name="Some Single"),
            root_folder=TOM,
            logger=LOG,
        )


async def test_unknown_root_folder_is_refused() -> None:
    """A mapped path Lidarr doesn't have must fail before anything is added."""
    lidarr = _lidarr()

    with pytest.raises(LidarrError, match="root folder"):
        await add_album(lidarr, _request(), root_folder="/nope", logger=LOG)  # type: ignore[arg-type]
    assert lidarr.added == []


async def test_artist_lidarr_cannot_resolve_is_refused() -> None:
    """An MBID Lidarr's metadata server doesn't know yields a clear error, not an add."""
    lidarr = _lidarr()

    with pytest.raises(LidarrError, match="couldn't find artist"):
        await add_album(
            lidarr,  # type: ignore[arg-type]
            _request(artist_mbid="unknown-mbid", artist_name="Nobody"),
            root_folder=TOM,
            logger=LOG,
        )
    assert lidarr.added == []
