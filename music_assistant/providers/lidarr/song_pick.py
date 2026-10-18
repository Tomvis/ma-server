"""
Client for the beets-lidarr "song pick" service (MUSIC-115).

``lidarr/add_track`` hands the track to that service, which picks the song's best
edition and wants it in Lidarr; MA no longer matches tracks inside Lidarr itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import aiohttp

from music_assistant.providers.lidarr.client import LidarrError

if TYPE_CHECKING:
    from aiohttp import ClientSession

# Seconds. The service answers only once Lidarr finished: a brand-new artist (~180 s
# settle) plus a metadata-profile switch (refresh, album poll, settle wait) is ~480 s,
# before MusicBrainz and the live DRDB lookup.
REQUEST_TIMEOUT = 600


async def want_song(session: ClientSession, base_url: str, body: dict[str, Any]) -> dict[str, Any]:
    """
    POST ``body`` to the song-pick service and return its 200 reply.

    Any refusal or failure raises ``LidarrError`` carrying the service's ``error``
    message, so the UI toast says what went wrong.

    :param session: Shared aiohttp session.
    :param base_url: Service base URL (without ``/song``).
    :param body: The request JSON.
    """
    url = f"{base_url.rstrip('/')}/song"
    try:
        async with session.post(
            url, json=body, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        ) as resp:
            status = resp.status
            try:
                data = await resp.json(content_type=None)
            except aiohttp.ContentTypeError, ValueError:
                data = None
    except TimeoutError as err:
        raise LidarrError(f"Song pick service timed out after {REQUEST_TIMEOUT}s") from err
    except aiohttp.ClientError as err:
        raise LidarrError(f"Song pick service unreachable at {base_url}: {err}") from err
    if status != 200:
        message = data.get("error") if isinstance(data, dict) else None
        raise LidarrError(str(message or f"Song pick service returned HTTP {status}"))
    if not isinstance(data, dict):
        raise LidarrError("Song pick service returned an unreadable reply")
    return data
