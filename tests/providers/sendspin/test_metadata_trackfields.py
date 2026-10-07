"""
Tests for the album/queue track numbers pushed in Sendspin metadata.

album_track / queue_track / total_tracks only exist on the local aiosendspin
"trackfields" fork - vanilla aiosendspin exposes a single `track`. The existing
group-state tests replace send_current_media_metadata with an AsyncMock, so they
stay green even if these fields are dropped, renamed, or rejected by the
installed aiosendspin. These tests drive the real method and assert on the
Metadata object that actually reaches the metadata role.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from aiosendspin.server.roles.metadata.state import Metadata
from music_assistant_models.enums import MediaType, PlaybackState, RepeatMode
from music_assistant_models.player import PlayerMedia

from music_assistant.providers.sendspin.player import SendspinPlayer


def _player(
    *,
    current_index: int | None,
    queue_items: int,
    track_number: int | None = 7,
) -> tuple[SendspinPlayer, Mock]:
    """Build a SendspinPlayer stub whose metadata role records what it is sent."""
    player = Mock(spec=SendspinPlayer)
    player.available = True
    media = PlayerMedia(
        uri="track-1",
        media_type=MediaType.TRACK,
        title="Everything In Its Right Place",
        artist="Radiohead",
        album="Kid A",
        duration=250,
        source_id="queue-1",
        queue_item_id="qi-1",
        elapsed_time=10,
        elapsed_time_last_updated=1.0,
    )
    player.state = SimpleNamespace(current_media=media, playback_state=PlaybackState.PLAYING)
    queue = SimpleNamespace(
        repeat_mode=RepeatMode.OFF,
        shuffle_enabled=False,
        current_index=current_index,
        items=queue_items,
    )
    queue_item = SimpleNamespace(
        media_item=SimpleNamespace(
            media_type=MediaType.TRACK,
            track_number=track_number,
            album=None,
            artists=[],
        )
    )
    player.mass = Mock(
        player_queues=Mock(get=Mock(return_value=queue), get_item=Mock(return_value=queue_item))
    )
    player._send_album_artwork = AsyncMock()
    player._send_artist_artwork = AsyncMock()
    player._clear_current_media_metadata = AsyncMock()
    player._compute_track_progress_ms = Mock(return_value=10_000)
    # Upstream's generation guard + metadata builder: run the real builder, own the snapshot.
    player._content_takeover_pending = False
    player._metadata_generation = 0
    player._metadata_publish_allowed = Mock(return_value=True)
    player._metadata_lock = asyncio.Lock()
    player._controller_role = None
    player._queue_repeat_shuffle = SendspinPlayer._queue_repeat_shuffle
    player._publish_repeat_shuffle = Mock()
    player._build_current_media_metadata = lambda *args, **kwargs: (
        SendspinPlayer._build_current_media_metadata(player, *args, **kwargs)
    )
    metadata_role = Mock()
    player._metadata_role = metadata_role
    return player, metadata_role


def _sent_metadata(player: SendspinPlayer, metadata_role: Mock) -> Metadata:
    asyncio.run(SendspinPlayer.send_current_media_metadata(player))
    metadata_role.set_metadata.assert_called_once()
    metadata: Metadata = metadata_role.set_metadata.call_args.args[0]
    return metadata


def test_metadata_carries_album_queue_and_total_track_numbers() -> None:
    """The pushed Metadata exposes all three trackfields, not vanilla's single `track`."""
    player, metadata_role = _player(current_index=3, queue_items=17, track_number=7)

    metadata = _sent_metadata(player, metadata_role)

    # album_track comes from the media item; queue_track is 1-based over the queue
    assert metadata.album_track == 7
    assert metadata.queue_track == 4
    assert metadata.total_tracks == 17


def test_metadata_omits_queue_position_when_queue_has_no_current_index() -> None:
    """A queue that has not started yet reports no queue position."""
    player, metadata_role = _player(current_index=None, queue_items=17)

    metadata = _sent_metadata(player, metadata_role)

    assert metadata.queue_track is None
    assert metadata.total_tracks == 17


def test_metadata_omits_total_tracks_for_an_empty_queue() -> None:
    """An empty queue reports no total rather than 0."""
    player, metadata_role = _player(current_index=0, queue_items=0)

    metadata = _sent_metadata(player, metadata_role)

    assert metadata.queue_track == 1
    assert metadata.total_tracks is None


def test_installed_aiosendspin_supports_the_trackfields_api() -> None:
    """
    Guard the dependency itself, not just our call site.

    manifest.json pins `aiosendspin[server]==6.0.5`, which PEP 440 also considers
    satisfied by the local 6.0.5+trackfields fork - so a clean environment can
    silently resolve vanilla aiosendspin, where these kwargs raise TypeError.
    """
    metadata = Metadata(title="t", album_track=1, queue_track=2, total_tracks=3)

    assert (metadata.album_track, metadata.queue_track, metadata.total_tracks) == (1, 2, 3)
