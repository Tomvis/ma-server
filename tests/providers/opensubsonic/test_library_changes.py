"""Tests for library changes read from Navidrome's native API (MUSIC-119)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from music_assistant_models.enums import MediaType

from music_assistant.helpers.json import json_dumps, json_loads
from music_assistant.providers.opensubsonic import sonic_provider
from music_assistant.providers.opensubsonic.sonic_provider import OpenSonicProvider, _nd_time

from .test_nd_album_extras import _native, _Resp

T0 = "2026-10-17T10:00:00.1Z"
T1 = "2026-10-17T11:00:00.465031921Z"
T2 = "2026-10-17T12:00:00Z"


def _item(
    item_id: str, updated: str, missing: bool = False, album_id: str | None = None
) -> dict[str, Any]:
    return {"id": item_id, "updatedAt": updated, "missing": missing, "albumId": album_id}


def _album(album_id: str, imported: str, missing: bool = False) -> dict[str, Any]:
    # an album's updatedAt is its newest file mtime, which says nothing about a change
    return {"id": album_id, "importedAt": imported, "updatedAt": T0, "missing": missing}


class _Library:
    """Serve native API resources sorted newest first, and an album's songs."""

    def __init__(self, **resources: list[dict[str, Any]]) -> None:
        self.resources = {"artist": [], "album": [], "song": [], **resources}
        self.requests: list[tuple[str, str]] = []

    async def get(self, path: str, params: dict[str, str] | None = None) -> list[dict[str, Any]]:
        assert params is not None
        resource = path.removeprefix("/api/")
        start, end = int(params["_start"]), int(params["_end"])
        if album_id := params.get("album_id"):
            assert resource == "song"
            assert params["missing"] == "false"
            self.requests.append((f"song?album_id={album_id}", params["_start"]))
            songs = [
                x for x in self.resources["song"] if x["albumId"] == album_id and not x["missing"]
            ]
            return songs[start:end]
        assert params["_sort"] == ("importedAt" if resource == "album" else "updatedAt")
        assert params["_order"] == "DESC"
        self.requests.append((resource, params["_start"]))
        items = sorted(
            self.resources[resource], key=lambda x: _nd_time(x[params["_sort"]]), reverse=True
        )
        return items[start:end]


def _serve(provider: OpenSonicProvider, library: _Library) -> None:
    provider._nd_get = library.get  # type: ignore[method-assign]


def _cursor(**newest: str) -> str:
    return json_dumps(newest)


@pytest.fixture(autouse=True)
def _no_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read changes from the cursor exactly; test_recent_changes_are_read_again covers it."""
    monkeypatch.setattr(sonic_provider, "_ND_CHANGES_OVERLAP", timedelta(0))


def test_nd_time_keeps_nanoseconds() -> None:
    """Timestamps a nanosecond apart compare, which datetime alone would not tell apart."""
    assert _nd_time("2026-10-17T11:00:00.465031921Z") > _nd_time("2026-10-17T11:00:00.46503192Z")
    assert _nd_time("2026-10-17T11:00:00.5Z") == _nd_time("2026-10-17T11:00:00.500Z")
    assert _nd_time("2026-10-17T11:00:01Z") > _nd_time("2026-10-17T11:00:00.999999999Z")


@pytest.mark.asyncio
async def test_first_call_only_records_a_baseline(provider: OpenSonicProvider) -> None:
    """Without a cursor nothing tells what changed, so a full sync is asked for."""
    library = _Library(song=[_item("s1", T1), _item("s2", T0)], album=[_album("a1", T1)])
    _serve(provider, library)

    changes = await provider.get_library_changes(None)

    assert changes is not None
    assert not changes.complete
    assert json_loads(changes.cursor) == {"album_imported": T1, "song": T1}


@pytest.mark.asyncio
async def test_items_changed_since_the_cursor(provider: OpenSonicProvider) -> None:
    """Only items newer than the cursor are reported, missing ones as removed."""
    library = _Library(
        artist=[_item("ar1", T0)],
        album=[_album("a1", T2), _album("a2", T0)],
        song=[_item("s1", T2), _item("s2", T2, missing=True), _item("s3", T1), _item("s4", T0)],
    )
    _serve(provider, library)

    changes = await provider.get_library_changes(_cursor(artist=T0, album_imported=T0, song=T1))

    assert changes is not None
    assert changes.complete
    assert changes.changed == {MediaType.ALBUM: {"a1"}, MediaType.TRACK: {"s1"}}
    assert changes.removed == {MediaType.TRACK: {"s2"}}
    assert json_loads(changes.cursor) == {"artist": T0, "album_imported": T2, "song": T2}


@pytest.mark.asyncio
async def test_touched_album_reports_its_songs_as_restored(provider: OpenSonicProvider) -> None:
    """A file moved back keeps its old mtime, so only its album's importedAt tells."""
    library = _Library(
        album=[_album("a1", T2), _album("a2", T0)],
        song=[
            _item("s1", T0, album_id="a1"),
            _item("s2", T0, missing=True, album_id="a1"),
            _item("s3", T0, album_id="a2"),
        ],
    )
    _serve(provider, library)

    changes = await provider.get_library_changes(_cursor(artist=T0, album_imported=T1, song=T1))

    assert changes is not None
    assert changes.complete
    assert changes.changed == {MediaType.ALBUM: {"a1"}}
    assert changes.restored == {MediaType.TRACK: {"s1"}}
    assert changes.changed_at == {MediaType.TRACK: {"s1": int(_nd_time(T0)[0].timestamp())}}
    assert ("song?album_id=a2", "0") not in library.requests


@pytest.mark.asyncio
async def test_pre_imported_at_cursor_asks_for_a_full_sync(provider: OpenSonicProvider) -> None:
    """A cursor that tracked albums by updatedAt can't date album changes, so a full sync runs."""
    _serve(provider, _Library(album=[_album("a1", T1)], song=[_item("s1", T1)]))

    changes = await provider.get_library_changes(_cursor(artist=T0, album=T1, song=T1))

    assert changes is not None
    assert not changes.complete
    assert json_loads(changes.cursor) == {"artist": T0, "album_imported": T1, "song": T1}


@pytest.mark.asyncio
async def test_recent_changes_are_read_again(
    provider: OpenSonicProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A change scanned after a newer mtime was already seen still shows up."""
    monkeypatch.setattr(sonic_provider, "_ND_CHANGES_OVERLAP", timedelta(minutes=10))
    library = _Library(
        song=[
            _item("seen", "2026-10-17T11:00:00Z"),
            _item("late", "2026-10-17T10:55:00Z", missing=True),
            _item("old", "2026-10-17T10:49:59Z"),
        ]
    )
    _serve(provider, library)

    changes = await provider.get_library_changes(
        _cursor(artist=T0, album_imported=T0, song="2026-10-17T11:00:00Z")
    )

    assert changes is not None
    assert changes.changed == {MediaType.TRACK: {"seen"}}
    assert changes.removed == {MediaType.TRACK: {"late"}}
    assert json_loads(changes.cursor)["song"] == "2026-10-17T11:00:00Z"


@pytest.mark.asyncio
async def test_unchanged_library_keeps_its_cursor(provider: OpenSonicProvider) -> None:
    """No changes report nothing and keep the cursor, also for an empty resource."""
    library = _Library(song=[_item("s1", T1)])
    _serve(provider, library)
    cursor = _cursor(artist=T0, album_imported=T0, song=T1)

    changes = await provider.get_library_changes(cursor)

    assert changes is not None
    assert changes.complete
    assert not changes.changed
    assert not changes.removed
    assert json_loads(changes.cursor) == json_loads(cursor)


@pytest.mark.asyncio
async def test_changes_page_until_the_cursor(
    provider: OpenSonicProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pages are read until an item older than the cursor shows up."""
    monkeypatch.setattr(sonic_provider, "_ND_CHANGES_PAGE_SIZE", 2)
    songs = [_item(f"s{i}", f"2026-10-17T11:00:0{i}Z") for i in range(6)]
    library = _Library(song=songs)
    _serve(provider, library)

    changes = await provider.get_library_changes(
        _cursor(artist=T0, album_imported=T0, song="2026-10-17T11:00:01Z")
    )

    assert changes is not None
    assert changes.changed == {MediaType.TRACK: {"s2", "s3", "s4", "s5"}}
    assert [x for x in library.requests if x[0] == "song"] == [
        ("song", "0"),
        ("song", "2"),
        ("song", "4"),
    ]


@pytest.mark.asyncio
async def test_too_many_changes_ask_for_a_full_sync(
    provider: OpenSonicProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """More changes than the limit make a full sync cheaper, with the cursor still moved on."""
    monkeypatch.setattr(sonic_provider, "_ND_MAX_CHANGES", 2)
    library = _Library(
        album=[_album("a1", T2)],
        song=[_item("s1", T2), _item("s2", T2)],
    )
    _serve(provider, library)

    changes = await provider.get_library_changes(_cursor(artist=T0, album_imported=T0, song=T0))

    assert changes is not None
    assert not changes.complete
    assert json_loads(changes.cursor) == {"artist": T0, "album_imported": T2, "song": T2}


@pytest.mark.asyncio
async def test_server_without_native_api_reports_no_changes(provider: OpenSonicProvider) -> None:
    """A server that isn't Navidrome can't report changes, so it keeps full syncs."""

    async def _no_native(*_args: Any) -> None:
        return None

    provider._nd_get = _no_native  # type: ignore[method-assign,assignment]

    assert await provider.get_library_changes(_cursor(song=T0)) is None


@pytest.mark.asyncio
async def test_unreadable_cursor_records_a_new_baseline(provider: OpenSonicProvider) -> None:
    """A cursor that doesn't parse is treated like none."""
    _serve(provider, _Library(song=[_item("s1", T1)]))

    changes = await provider.get_library_changes("not json")

    assert changes is not None
    assert not changes.complete
    assert json_loads(changes.cursor) == {"song": T1}


@pytest.mark.asyncio
async def test_first_native_call_requests_the_given_path(provider: OpenSonicProvider) -> None:
    """Building the base URL on the first call does not replace the requested API path."""
    setup = {"baseURL": "http://navidrome", "port": 4533, "path": "music"}
    provider.get_setup_value = setup.get  # type: ignore[method-assign,assignment]
    requested: list[str] = []

    class _Session:
        def post(self, _url: str, **_kwargs: Any) -> _Resp:
            return _Resp(200, {"token": "t"})

        def get(self, url: str, **_kwargs: Any) -> _Resp:
            requested.append(url)
            return _Resp(200, [])

    _native(provider, _Session())  # type: ignore[arg-type]
    provider._nd_base_url = None

    assert await provider._nd_get("/api/song") == []
    assert requested == ["http://navidrome:4533/music/api/song"]
