"""The provider class for Open Subsonic."""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from aiohttp import ClientResponseError
from libopensonic import AsyncConnection as SonicConnection
from libopensonic import Extensions as OpenSubsonicExtensions
from libopensonic.errors import (
    AuthError,
    CredentialError,
    DataNotFoundError,
    ParameterError,
    SonicError,
)
from libopensonic.media import PodcastChannel
from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType, ContentType, MediaType, StreamType
from music_assistant_models.errors import (
    ActionUnavailable,
    InvalidDataError,
    LoginFailed,
    MediaNotFoundError,
    ProviderPermissionDenied,
    UnsupportedFeaturedException,
)
from music_assistant_models.media_items import (
    Album,
    Artist,
    AudioFormat,
    BrowseFolder,
    ItemMapping,
    MediaItemType,
    Playlist,
    Podcast,
    PodcastEpisode,
    ProviderMapping,
    Radio,
    RecommendationFolder,
    SearchResults,
    Track,
    UniqueList,
)
from music_assistant_models.media_items.metadata import CriticalReception
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.constants import (
    CONF_PASSWORD,
    CONF_PATH,
    CONF_PORT,
    CONF_USERNAME,
    UNKNOWN_ARTIST,
)
from music_assistant.controllers.cache import use_cache
from music_assistant.helpers.podcast_parsers import rank_episodes_by_date
from music_assistant.helpers.tags import async_parse_tags
from music_assistant.helpers.util import TaskManager, remove_file
from music_assistant.models.music_provider import MusicProvider

from .parsers import (
    EP_CHAN_SEP,
    NAVI_VARIOUS_PREFIX,
    UNKNOWN_ARTIST_ID,
    parse_album,
    parse_artist,
    parse_epsiode,
    parse_playlist,
    parse_podcast,
    parse_radio,
    parse_structured_lyrics,
    parse_track,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from libopensonic.media import AlbumID3 as SonicAlbum
    from libopensonic.media import ArtistWithAlbumsID3 as SonicArtist
    from libopensonic.media import Bookmark as SonicBookmark
    from libopensonic.media import Child as SonicItem
    from libopensonic.media import InternetRadioStation as SonicRadio
    from libopensonic.media import Lyrics as SonicLyrics
    from libopensonic.media import OpenSubsonicExtension, StructuredLyrics
    from libopensonic.media import Playlist as SonicPlaylist
    from libopensonic.media import PodcastEpisode as SonicEpisode

CONF_BASE_URL = "baseURL"
CONF_API_KEY = "api_key"
CONF_ENABLE_PODCASTS = "enable_podcasts"
CONF_ENABLE_RADIO_STATIONS = "enable_radio_stations"
CONF_ENABLE_LEGACY_AUTH = "enable_legacy_auth"
CONF_RECO_FAVES = "recommend_favorites"
CONF_NEW_ALBUMS = "recommend_new"
CONF_PLAYED_ALBUMS = "recommend_played"
CONF_RECO_SIZE = "recommendation_count"
CONF_PAGE_SIZE = "pagination_size"
CONF_RAW_FILE = "request_raw_file"

CACHE_CATEGORY_PODCAST_CHANNEL = 1
CACHE_CATEGORY_PODCAST_EPISODES = 2
CACHE_CATEGORY_CRITICAL_RECEPTION = 3

# Raw prefix kept when extracting custom AMG/TPS/DR tags from a format the tag-prefix
# reader doesn't walk (MP4 'moov' and the like sit near the start of the file).
# ID3v2-tagged and FLAC files are walked instead (see _TagPrefixReader), so a large
# embedded cover image can't push their tags past this cap.
CRITICAL_RECEPTION_PROBE_BYTES = 512 * 1024
# Audio kept after a leading ID3v2 tag: ffprobe needs a few MPEG frames to accept it.
_CR_PROBE_AUDIO_TAIL_BYTES = 128 * 1024
# Largest leading ID3v2 tag kept in memory; bigger tags fall back to the raw prefix.
_CR_PROBE_MAX_ID3_BYTES = 16 * 1024 * 1024
# Total bytes read per probe, including cover art read past and not kept.
_CR_PROBE_MAX_READ_BYTES = 32 * 1024 * 1024
# CR/DR tags are written to the file once by the offline tagger and rarely change, so
# the extraction result is cached for a month rather than a day: at 24h every album in
# the library expired and re-probed itself daily (a 5000-album library paid 5000 probes
# a day) just to rediscover unchanged tags. A retag is still picked up before the TTL
# runs out — a force refresh (music/refresh_item) sets BYPASS_CACHE, which skips this
# entry because it is stored non-persistent, and the cache controller's "clear cache"
# action drops it outright.
CRITICAL_RECEPTION_CACHE_TTL = 86400 * 30  # 30 days
# Bumped when the CR tags read from files gain a field: an entry written by an older
# version that has review sources is re-probed once to pick it up (2 = <SRC>_REVIEW).
_CR_CACHE_VERSION = 2
# An album whose sampled files ffprobe cannot read fails the same way on every sync, and
# the Navidrome sync plugin starts a sync after every scan; a short negative entry stops
# it re-streaming those files all day (MUSIC-20) while still retrying after a retag.
_CR_UNPARSABLE_CACHE_TTL = 86400
# How many tracks to ffprobe before giving up. A first track that's a bonus /
# hidden track may have been written without the album's AMG/TPS/DR tags even
# when later tracks carry them; sampling a few covers this without blowing up
# sync cost.
_CR_PROBE_SONG_ATTEMPTS = 3
# Wall-clock budget for the whole per-album probe loop (all attempts together).
# The attempts run sequentially and each can burn the full PARSE_TAGS_TIMEOUT_SECONDS
# inside ffprobe, so without this the per-album worst case is ~90s — and get_album()
# awaits this path inline, so that is user-facing stall time. 30s keeps the bound no
# worse than the ~36s ceiling this code effectively had back when each probe carried
# its own 12s timeout, while still leaving room for two or three slow-but-working
# probes (stream fetch + ffprobe) to finish.
# This bounds CALLER LATENCY, not the work: cancelling cannot stop the asyncio.to_thread
# worker running ffprobe, so an orphaned probe keeps occupying a default-executor thread
# until its own PARSE_TAGS_TIMEOUT_SECONDS kills the subprocess.
_CR_PROBE_ALBUM_BUDGET_SECONDS = 30.0
# Max albums enriched concurrently per library-sync page. Bounds simultaneous
# conn.get_album round-trips and ffprobe subprocesses so a page of cache misses
# overlaps latency without saturating the network or the default thread pool.
_CR_ENRICH_CONCURRENCY = 4

Param = ParamSpec("Param")
RetType = TypeVar("RetType")


class _Unparsable:
    """A probe whose file ffprobe rejects outright, as opposed to a stream error."""


_PROBE_UNPARSABLE = _Unparsable()


class _TagPrefixReader:
    """
    Collect the smallest prefix of a streamed audio file that still carries its tags.

    A leading ID3v2 tag is kept whole, followed by a little audio. FLAC metadata is cut
    down to STREAMINFO plus VORBIS_COMMENT, read past any cover art or padding without
    keeping it, so the custom tags parse from a few kilobytes wherever the PICTURE block
    sits. Anything else keeps a raw prefix of CRITICAL_RECEPTION_PROBE_BYTES.
    """

    def __init__(self) -> None:
        """Start an empty prefix."""
        self._pending = bytearray()
        self._out = bytearray()
        self._phase = "start"
        self._skip = 0
        self._read = 0
        self._id3_size = 0
        self._tail_start = 0
        self._last_kept_header: int | None = None

    def feed(self, chunk: bytes) -> bool:
        """
        Add the next chunk of the stream.

        :param chunk: Bytes as they arrive from the stream.
        :return: True once the prefix is complete and reading can stop.
        """
        self._pending += chunk
        self._read += len(chunk)
        steps = {
            "start": self._step_start,
            "id3": self._step_id3,
            "flac": self._step_flac,
            "tail": self._step_tail,
        }
        while True:
            if self._skip:
                dropped = min(self._skip, len(self._pending))
                del self._pending[:dropped]
                self._skip -= dropped
            if self._phase == "done":
                return True
            if self._phase == "raw":
                return len(self._pending) >= CRITICAL_RECEPTION_PROBE_BYTES
            if self._skip or not steps[self._phase]():
                return self._read >= _CR_PROBE_MAX_READ_BYTES

    def result(self) -> bytes:
        """Return the prefix collected so far."""
        if self._phase in ("start", "raw"):
            return bytes(self._pending[:CRITICAL_RECEPTION_PROBE_BYTES])
        if self._phase == "id3":
            return bytes(self._pending)
        return bytes(self._out)

    def _step_start(self) -> bool:
        pending = self._pending
        if len(pending) < 10:
            return False
        if pending[:3] == b"ID3":
            # synchsafe size: 7 bits per byte, plus the 10-byte header (and footer)
            size = 10 + sum((pending[6 + i] & 0x7F) << (7 * (3 - i)) for i in range(4))
            if pending[5] & 0x10:
                size += 10
            if size > _CR_PROBE_MAX_ID3_BYTES:
                self._phase = "raw"
            else:
                self._id3_size = size
                self._phase = "id3"
        elif pending[:4] == b"fLaC":
            self._start_flac()
        else:
            self._phase = "raw"
        return True

    def _step_id3(self) -> bool:
        pending = self._pending
        size = self._id3_size
        if len(pending) < size + 4:
            return False
        if pending[size : size + 4] == b"fLaC":
            # FLAC keeps its tags in VORBIS_COMMENT; the stray ID3 tag adds nothing
            del pending[:size]
            self._start_flac()
        else:
            self._out += pending[:size]
            del pending[:size]
            self._tail_start = len(self._out)
            self._phase = "tail"
        return True

    def _step_flac(self) -> bool:
        pending = self._pending
        if len(pending) < 4:
            return False
        header = pending[0]
        block_type = header & 0x7F
        length = int.from_bytes(pending[1:4], "big")
        if block_type in (0, 4):  # STREAMINFO, VORBIS_COMMENT
            if len(pending) < 4 + length:
                return False
            self._last_kept_header = len(self._out)
            self._out += pending[: 4 + length]
            del pending[: 4 + length]
        else:
            del pending[:4]
            self._skip = length
        if block_type == 4 or header & 0x80:
            if self._last_kept_header is not None:
                self._out[self._last_kept_header] |= 0x80  # last-metadata-block flag
            self._phase = "done"
        return True

    def _step_tail(self) -> bool:
        self._out += self._pending
        self._pending.clear()
        if len(self._out) - self._tail_start < _CR_PROBE_AUDIO_TAIL_BYTES:
            return False
        del self._out[self._tail_start + _CR_PROBE_AUDIO_TAIL_BYTES :]
        self._phase = "done"
        return True

    def _start_flac(self) -> None:
        self._out += b"fLaC"
        del self._pending[:4]
        self._phase = "flac"


class OpenSonicProvider(MusicProvider):
    """Provider for Open Subsonic servers."""

    conn: SonicConnection
    _enable_podcasts: bool = True
    _enable_radio_stations: bool = True
    _show_faves: bool = True
    _show_new: bool = True
    _show_played: bool = True
    _reco_limit: int = 10
    _pagination_size: int = 200
    _id_lyrics: bool = False
    _direct_podcast_episode: bool = False
    _raw_file: bool = True

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return Config entries to setup this provider."""
        return (
            ConfigEntry(
                key=CONF_ENABLE_PODCASTS,
                type=ConfigEntryType.BOOLEAN,
                required=True,
                hidden=True,
                default_value=True,
            ),
            ConfigEntry(
                key=CONF_ENABLE_RADIO_STATIONS,
                type=ConfigEntryType.BOOLEAN,
                required=True,
                default_value=True,
            ),
            ConfigEntry(
                key=CONF_ENABLE_LEGACY_AUTH,
                type=ConfigEntryType.BOOLEAN,
                required=True,
                default_value=False,
            ),
            ConfigEntry(
                key=CONF_RECO_FAVES,
                type=ConfigEntryType.BOOLEAN,
                required=True,
                default_value=True,
            ),
            ConfigEntry(
                key=CONF_NEW_ALBUMS,
                type=ConfigEntryType.BOOLEAN,
                required=True,
                default_value=True,
            ),
            ConfigEntry(
                key=CONF_PLAYED_ALBUMS,
                type=ConfigEntryType.BOOLEAN,
                required=True,
                default_value=True,
            ),
            ConfigEntry(
                key=CONF_RECO_SIZE,
                type=ConfigEntryType.INTEGER,
                required=True,
                default_value=10,
            ),
            ConfigEntry(
                key=CONF_RAW_FILE,
                type=ConfigEntryType.BOOLEAN,
                required=False,
                default_value=True,
            ),
            ConfigEntry(
                key=CONF_PAGE_SIZE,
                type=ConfigEntryType.INTEGER,
                required=True,
                default_value=200,
                advanced=True,
            ),
        )

    async def handle_async_init(self) -> None:
        """Set up the music provider and test the connection."""
        port = self.get_setup_value(CONF_PORT)
        port = int(str(port)) if port is not None else 443
        path = self.get_setup_value(CONF_PATH)

        if path is None:
            path = ""

        api_key = self.get_setup_value(CONF_API_KEY)
        username = self.get_setup_value(CONF_USERNAME)
        password = self.get_setup_value(CONF_PASSWORD)

        if api_key:
            self.conn = SonicConnection(
                str(self.get_setup_value(CONF_BASE_URL)),
                api_key=str(api_key),
                port=port,
                server_path=str(path),
                use_get=True,
                app_name="Music Assistant",
            )
        elif username and password:
            self.conn = SonicConnection(
                str(self.get_setup_value(CONF_BASE_URL)),
                username=str(username),
                password=str(password),
                legacy_auth=bool(self.config.get_value(CONF_ENABLE_LEGACY_AUTH)),
                port=port,
                server_path=str(path),
                use_get=True,
                app_name="Music Assistant",
            )
        else:
            msg = f"No credentials for {self.get_setup_value(CONF_BASE_URL)}, provide an API key or username and password."
            raise LoginFailed(
                msg,
                translation_key="connect_failed",
                translation_owner=self.translation_owner,
                translation_args=[self.get_setup_value(CONF_BASE_URL)],
            )

        try:
            success = await self.conn.ping()
            if not success:
                raise CredentialError
        except (AuthError, CredentialError) as e:
            msg = (
                f"Failed to connect to {self.get_setup_value(CONF_BASE_URL)}, check your settings."
            )
            raise LoginFailed(
                msg,
                translation_key="connect_failed",
                translation_owner=self.translation_owner,
                translation_args=[self.get_setup_value(CONF_BASE_URL)],
            ) from e

        try:
            extensions: list[OpenSubsonicExtension] = await self.conn.get_open_subsonic_extensions()
            for entry in extensions:
                if entry.name == OpenSubsonicExtensions.SONG_LYRICS:
                    self._id_lyrics = True
                elif entry.name == OpenSubsonicExtensions.GET_PODCAST_EPISODE:
                    self._direct_podcast_episode = True
        except OSError:
            self.logger.info("Failed to query server for OpenSubsonic extensions")

        # Migration from the old subsonic config for enabling podcasts + the generic library sync option
        # to a only using the library sync (plus probing a podcast endpoint)
        # After this code has gone out in a release which marks the podcast option
        # as not visible, we will drop that config option entirely and set
        # _enable_podcasts to the result of the probe

        can_podcast = False
        if bool(self.config.get_value("library_sync_podcasts")):
            try:
                await self.conn.get_podcasts(inc_episodes=False)
                can_podcast = True
            except SonicError:
                self.logger.info("Server does not support podcasts, disabling")
            except ClientResponseError:
                self.logger.info("Server does not support podcasts, disabling")

        self._enable_podcasts = bool(self.config.get_value(CONF_ENABLE_PODCASTS)) and can_podcast
        self._enable_radio_stations = bool(self.config.get_value(CONF_ENABLE_RADIO_STATIONS))
        self._show_faves = bool(self.config.get_value(CONF_RECO_FAVES))
        self._show_new = bool(self.config.get_value(CONF_NEW_ALBUMS))
        self._show_played = bool(self.config.get_value(CONF_PLAYED_ALBUMS))
        self._reco_limit = int(str(self.config.get_value(CONF_RECO_SIZE)))
        self._pagination_size = int(str(self.config.get_value(CONF_PAGE_SIZE)))
        self._pagination_size = min(self._pagination_size, 500)
        self._raw_file = bool(self.config.get_value(CONF_RAW_FILE))

    async def unload(self, is_removed: bool = False) -> None:
        """Unload the provider."""
        await super().unload(is_removed)
        await self.conn.cleanup()

    @property
    def is_streaming_provider(self) -> bool:
        """
        Return True if the provider is a streaming provider.

        This literally means that the catalog is not the same as the library contents.
        For local based providers (files, plex), the catalog is the same as the library content.
        It also means that data is if this provider is NOT a streaming provider,
        data cross instances is unique, the catalog and library differs per instance.

        Setting this to True will only query one instance of the provider for search and lookups.
        Setting this to False will query all instances of this provider for search and lookups.
        """
        return False

    async def get_recommendations(self) -> list[RecommendationFolder]:
        """
        Get this provider's available recommendation rows, without items.

        These can be favorited items, recently added albums, newest podcast episodes,
        and most played albums.  What is included is configured with the provider.
        """
        recos: list[RecommendationFolder] = []
        if self._enable_podcasts:
            recos.append(
                RecommendationFolder(
                    item_id="subsonic_newest_podcasts",
                    provider=self.instance_id,
                    name="Newest Podcast Episodes",
                    translation_key="episodes_recently_added",
                )
            )
        if self._show_faves:
            recos.append(
                RecommendationFolder(
                    item_id="subsonic_starred_albums",
                    provider=self.instance_id,
                    name="Starred Items",
                    translation_key="starred_items",
                )
            )
        if self._show_new:
            recos.append(
                RecommendationFolder(
                    item_id="subsonic_new_albums",
                    provider=self.instance_id,
                    name="New Albums",
                    translation_key="recently_added_albums",
                )
            )
        if self._show_played:
            recos.append(
                RecommendationFolder(
                    item_id="subsonic_most_played",
                    provider=self.instance_id,
                    name="Most Played Albums",
                    translation_key="most_played_albums",
                )
            )
        return recos

    async def get_recommendation_items(
        self, item_id: str
    ) -> UniqueList[MediaItemType | ItemMapping | BrowseFolder]:
        """
        Get the items for a single recommendation row.

        :param item_id: The item_id of the row, as returned by get_recommendations.
        """
        folder: RecommendationFolder | None = None
        if item_id == "subsonic_newest_podcasts" and self._enable_podcasts:
            folder = await self._podcast_recommendations()
        elif item_id == "subsonic_starred_albums" and self._show_faves:
            folder = await self._favorites_recommendation()
        elif item_id == "subsonic_new_albums" and self._show_new:
            folder = await self._new_recommendations()
        elif item_id == "subsonic_most_played" and self._show_played:
            folder = await self._played_recommendations()
        if folder is None:
            return UniqueList()
        return folder.items

    async def resolve_image(self, path: str) -> bytes | Any:
        """Return the image."""
        self.logger.debug("Requesting cover art for '%s'", path)

        try:
            art = await self.conn.get_cover_art(path)
            return await art.content.read()
        except DataNotFoundError:
            self.logger.warning("Unable to locate a cover image for %s", path)
            return None

    @use_cache(3600 * 3)  # cache for 3 hours
    async def search(
        self, search_query: str, media_types: list[MediaType], limit: int = 20
    ) -> SearchResults:
        """Search the sonic library."""
        artists = limit if MediaType.ARTIST in media_types else 0
        albums = limit if MediaType.ALBUM in media_types else 0
        songs = limit if MediaType.TRACK in media_types else 0
        if not (artists or albums or songs):
            return SearchResults()
        answer = await self.conn.search3(
            query=search_query,
            artist_count=artists,
            artist_offset=0,
            album_count=albums,
            album_offset=0,
            song_count=songs,
            song_offset=0,
        )

        if answer.artist:
            ar = [
                parse_artist(self.instance_id, entry, logger=self.logger) for entry in answer.artist
            ]
        else:
            ar = []

        if answer.album:
            al = [parse_album(self.logger, self.instance_id, entry) for entry in answer.album]
        else:
            al = []

        if answer.song:
            tr = []
            for entry in answer.song:
                self._set_loudness(entry)
                tr.append(parse_track(self.logger, self.instance_id, entry))
        else:
            tr = []

        return SearchResults(artists=ar, albums=al, tracks=tr)

    async def set_favorite(
        self, prov_item_id: str, media_type: MediaType, favorite: bool | None
    ) -> None:
        """
        Set or clear favorite on the server.

        Subsonic only knows starred or not, so both a dislike and an unset unstar the item.
        """
        # The subsonic spec does not support favorite-ing anything but artists, albums, and tracks
        if media_type not in (MediaType.ARTIST, MediaType.ALBUM, MediaType.TRACK):
            return

        track_ids: list[str] = []
        album_ids: list[str] = []
        artist_ids: list[str] = []

        if media_type == MediaType.ARTIST:
            artist_ids.append(prov_item_id)
        elif media_type == MediaType.ALBUM:
            album_ids.append(prov_item_id)
        elif media_type == MediaType.TRACK:
            track_ids.append(prov_item_id)

        if favorite:
            await self.conn.star(sids=track_ids, album_ids=album_ids, artist_ids=artist_ids)
        else:
            await self.conn.unstar(sids=track_ids, album_ids=album_ids, artist_ids=artist_ids)

    async def get_library_artists(self) -> AsyncGenerator[Artist]:
        """Provide a generator for reading all artists."""
        artists = await self.conn.get_artists()

        if not artists.index:
            return

        for index in artists.index:
            if not index.artist:
                continue

            for artist in index.artist:
                yield parse_artist(self.instance_id, artist, logger=self.logger)

    async def get_library_albums(self) -> AsyncGenerator[Album]:
        """
        Provide a generator for reading all artists.

        Note the pagination, the open subsonic docs say that this method is limited to
        returning 500 items per invocation.
        """
        offset = 0
        size = self._pagination_size
        albums = await self.conn.get_album_list2(
            ltype="alphabeticalByArtist",
            size=size,
            offset=offset,
        )
        while albums:
            # Pull AMG/TPS/DR custom tags out of one track per album. These don't
            # ride on the OpenSubsonic schema, so we ffprobe a short prefix of the
            # audio. Enrich the whole page with bounded concurrency before yielding
            # so cache-miss albums overlap their network/ffprobe latency instead of
            # serializing. _enrich_album_with_critical_reception swallows its own
            # failures (non-fatal — the album still syncs, just without
            # critical_reception), so one album can't abort the page.
            parsed_page = [parse_album(self.logger, self.instance_id, album) for album in albums]
            async with TaskManager(self.mass, _CR_ENRICH_CONCURRENCY) as tm:
                for parsed, album in zip(parsed_page, albums, strict=True):
                    await tm.create_task_with_limit(
                        self._enrich_album_with_critical_reception(parsed, album.id)
                    )
            # Yield in the original page order after enrichment.
            for parsed in parsed_page:
                yield parsed
            offset += size
            albums = await self.conn.get_album_list2(
                ltype="alphabeticalByArtist",
                size=size,
                offset=offset,
            )

    async def get_library_playlists(self) -> AsyncGenerator[Playlist]:
        """Provide a generator for library playlists."""
        results = await self.conn.get_playlists()
        for entry in results:
            yield parse_playlist(self.instance_id, entry)

    async def get_library_radios(self) -> AsyncGenerator[Radio]:
        """Provide a generator for library radio stations."""
        if not self._enable_radio_stations:
            return
        stations: list[SonicRadio] = await self.conn.get_internet_radio_stations()
        for entry in stations:
            yield parse_radio(self.instance_id, entry)

    async def get_radio(self, prov_radio_id: str) -> Radio:
        """Return the requested radio station."""
        async for station in self.get_library_radios():
            if station.item_id == prov_radio_id:
                return station
        msg = f"Radio {prov_radio_id} not found"
        raise MediaNotFoundError(msg)

    async def get_library_tracks(self) -> AsyncGenerator[Track]:
        """
        Provide a generator for library tracks.

        Note the lack of item count on this method.
        """
        query = ""
        offset = 0
        count = self._pagination_size
        try:
            results = await self.conn.search3(
                query=query,
                artist_count=0,
                album_count=0,
                song_offset=offset,
                song_count=count,
            )
        except ParameterError:
            # Older Navidrome does not accept an empty string and requires the empty quotes
            query = '""'
            results = await self.conn.search3(
                query=query,
                artist_count=0,
                album_count=0,
                song_offset=offset,
                song_count=count,
            )
        while results.song:
            album: Album | None = None
            for entry in results.song:
                aid = entry.album_id or entry.parent
                if aid is not None and (album is None or album.item_id != aid):
                    album = await self.get_album(prov_album_id=aid)
                self._set_loudness(entry)
                lyrics: tuple[str, bool] | None = await self.get_track_lyrics(entry)
                yield parse_track(self.logger, self.instance_id, entry, album=album, lyrics=lyrics)
            offset += count
            results = await self.conn.search3(
                query=query,
                artist_count=0,
                album_count=0,
                song_offset=offset,
                song_count=count,
            )

    @use_cache(3600 * 3)  # cache for 3 hours
    async def get_album(self, prov_album_id: str) -> Album:
        """Return the requested Album."""
        try:
            sonic_album: SonicAlbum = await self.conn.get_album(prov_album_id)
            sonic_info = await self.conn.get_album_info2(aid=prov_album_id)
        except (ParameterError, DataNotFoundError) as e:
            msg = f"Album {prov_album_id} not found"
            raise MediaNotFoundError(msg) from e

        album = parse_album(self.logger, self.instance_id, sonic_album, sonic_info)
        # Route through the shared CR cache so library sync and direct get_album
        # don't both pay for ffprobe; the pre-fetched sonic_album skips a redundant
        # conn.get_album on cache miss.
        await self._enrich_album_with_critical_reception(
            album, prov_album_id, sonic_album=sonic_album
        )
        return album

    @use_cache(3600 * 3)  # cache for 3 hours
    async def get_album_tracks(self, prov_album_id: str) -> list[Track]:
        """Return a list of tracks on the specified Album."""
        try:
            sonic_album: SonicAlbum = await self.conn.get_album(prov_album_id)
        except (ParameterError, DataNotFoundError) as e:
            msg = f"Album {prov_album_id} not found"
            raise MediaNotFoundError(msg) from e
        tracks = []
        if sonic_album.song:
            for sonic_song in sonic_album.song:
                self._set_loudness(sonic_song)
                lyrics: tuple[str, bool] | None = await self.get_track_lyrics(sonic_song)
                tracks.append(parse_track(self.logger, self.instance_id, sonic_song, lyrics=lyrics))
        return tracks

    @use_cache(3600 * 3)  # cache for 3 hours
    async def get_artist(self, prov_artist_id: str) -> Artist:
        """Return the requested Artist."""
        if prov_artist_id == UNKNOWN_ARTIST_ID:
            return Artist(
                item_id=UNKNOWN_ARTIST_ID,
                name=UNKNOWN_ARTIST,
                provider=self.instance_id,
                provider_mappings={
                    ProviderMapping(
                        item_id=UNKNOWN_ARTIST_ID,
                        provider_domain=self.domain,
                        provider_instance=self.instance_id,
                    )
                },
            )
        if prov_artist_id.startswith(NAVI_VARIOUS_PREFIX):
            # Special case for handling track artists on various artists album for Navidrome.
            return Artist(
                item_id=prov_artist_id,
                name=prov_artist_id.removeprefix(NAVI_VARIOUS_PREFIX),
                provider=self.instance_id,
                provider_mappings={
                    ProviderMapping(
                        item_id=prov_artist_id,
                        provider_domain=self.domain,
                        provider_instance=self.instance_id,
                    )
                },
            )

        try:
            sonic_artist: SonicArtist = await self.conn.get_artist(artist_id=prov_artist_id)
            sonic_info = await self.conn.get_artist_info2(aid=prov_artist_id)
        except (ParameterError, DataNotFoundError) as e:
            msg = f"Artist {prov_artist_id} not found"
            raise MediaNotFoundError(msg) from e
        return parse_artist(self.instance_id, sonic_artist, sonic_info, logger=self.logger)

    @use_cache(3600 * 3)  # cache for 3 hours
    async def get_track(self, prov_track_id: str) -> Track:
        """Return the specified track."""
        try:
            sonic_song: SonicItem = await self.conn.get_song(prov_track_id)
        except (ParameterError, DataNotFoundError) as e:
            msg = f"Item {prov_track_id} not found"
            raise MediaNotFoundError(msg) from e
        aid = sonic_song.album_id or sonic_song.parent
        album: Album | None = None
        if not aid:
            self.logger.warning("Unable to find album id for track %s", sonic_song.id)
        else:
            album = await self.get_album(prov_album_id=aid)
        self._set_loudness(sonic_song)
        lyrics: tuple[str, bool] | None = await self.get_track_lyrics(sonic_song)
        return parse_track(self.logger, self.instance_id, sonic_song, album=album, lyrics=lyrics)

    @use_cache(3600 * 3)  # cache for 3 hours
    async def get_artist_albums(self, prov_artist_id: str) -> list[Album]:
        """Return a list of all Albums by specified Artist."""
        if prov_artist_id == UNKNOWN_ARTIST_ID or prov_artist_id.startswith(NAVI_VARIOUS_PREFIX):
            return []

        try:
            sonic_artist: SonicArtist = await self.conn.get_artist(prov_artist_id)
        except (ParameterError, DataNotFoundError) as e:
            msg = f"Album {prov_artist_id} not found"
            raise MediaNotFoundError(msg) from e
        albums = []
        if sonic_artist.album:
            for entry in sonic_artist.album:
                albums.append(parse_album(self.logger, self.instance_id, entry))
        return albums

    @use_cache(3600 * 3)  # cache for 3 hours
    async def get_playlist(self, prov_playlist_id: str) -> Playlist:
        """Return the specified Playlist."""
        try:
            sonic_playlist: SonicPlaylist = await self.conn.get_playlist(prov_playlist_id)
        except (ParameterError, DataNotFoundError) as e:
            msg = f"Playlist {prov_playlist_id} not found"
            raise MediaNotFoundError(msg) from e
        return parse_playlist(self.instance_id, sonic_playlist)

    @use_cache(3600 * 3)  # cache for 3 hours
    async def get_podcast_episode(self, prov_episode_id: str) -> PodcastEpisode:
        """Get (full) podcast episode details by id."""
        podcast_id, _ = prov_episode_id.split(EP_CHAN_SEP)
        async for episode in self.get_podcast_episodes(podcast_id):
            if episode.item_id == prov_episode_id:
                return episode
        msg = f"Episode {prov_episode_id} not found"
        raise MediaNotFoundError(msg)

    async def get_podcast_episodes(
        self,
        prov_podcast_id: str,
    ) -> AsyncGenerator[PodcastEpisode]:
        """Get all Episodes for given podcast id."""
        if not self._enable_podcasts:
            return
        channels = await self.conn.get_podcasts(inc_episodes=True, pid=prov_podcast_id)
        channel = channels[0]
        if not channel.episode:
            return

        # rank on the publish date, so the order the server returns the episodes in does
        # not decide the ordering
        positions = rank_episodes_by_date([ep.publish_date for ep in channel.episode])
        for position, episode in zip(positions, channel.episode, strict=True):
            self._set_loudness(episode)
            yield parse_epsiode(self.instance_id, episode, channel, position)

    @use_cache(3600 * 3)  # cache for 3 hours
    async def get_podcast(self, prov_podcast_id: str) -> Podcast:
        """Get full Podcast details by id."""
        if not self._enable_podcasts:
            msg = "Podcasts are currently disabled in the provider configuration"
            raise ActionUnavailable(msg)

        channels = await self.conn.get_podcasts(inc_episodes=True, pid=prov_podcast_id)

        return parse_podcast(self.instance_id, channels[0])

    async def get_library_podcasts(self) -> AsyncGenerator[Podcast]:
        """Retrieve library/subscribed podcasts from the provider."""
        if self._enable_podcasts:
            channels = await self.conn.get_podcasts(inc_episodes=True)

            for channel in channels:
                yield parse_podcast(self.instance_id, channel)

    @use_cache(3600 * 3)  # cache for 3 hours
    async def get_playlist_tracks(self, prov_playlist_id: str, page: int = 0) -> list[Track]:
        """Get playlist tracks."""
        result: list[Track] = []
        if page > 0:
            # paging not supported, we always return the whole list at once
            return result
        try:
            sonic_playlist: SonicPlaylist = await self.conn.get_playlist(prov_playlist_id)
        except (ParameterError, DataNotFoundError) as e:
            msg = f"Playlist {prov_playlist_id} not found"
            raise MediaNotFoundError(msg) from e

        if not sonic_playlist.entry:
            return result

        for index, sonic_song in enumerate(sonic_playlist.entry, 1):
            # A playlist can hold thousands of tracks, so we must not trigger a per-track
            # metadata fetch here: parse_track derives the album reference from the playlist
            # entry itself, and lyrics are fetched on demand when a track is played (get_track).
            # Fetching album + lyrics per entry turned a single getPlaylist call into thousands
            # of serial requests, making large playlists take minutes to start.
            self._set_loudness(sonic_song)
            track = parse_track(self.logger, self.instance_id, sonic_song)
            track.position = index
            result.append(track)
        return result

    @use_cache(3600 * 3)  # cache for 3 hours
    async def get_artist_toptracks(self, prov_artist_id: str) -> list[Track]:
        """Get the top listed tracks for a specified artist."""
        # We have seen top tracks requested for the UNKNOWN_ARTIST ID, protect against that
        if prov_artist_id == UNKNOWN_ARTIST_ID or prov_artist_id.startswith(NAVI_VARIOUS_PREFIX):
            return []

        try:
            sonic_artist: SonicArtist = await self.conn.get_artist(prov_artist_id)
        except DataNotFoundError as e:
            msg = f"Artist {prov_artist_id} not found"
            raise MediaNotFoundError(msg) from e
        songs: list[SonicItem] = await self.conn.get_top_songs(sonic_artist.name)
        tracks = []
        for entry in songs:
            self._set_loudness(entry)
            tracks.append(parse_track(self.logger, self.instance_id, entry))
        return tracks

    @use_cache(3600 * 3)  # cache for 3 hours
    async def get_similar_tracks(self, prov_track_id: str, limit: int = 25) -> list[Track]:
        """Get tracks similar to selected track."""
        try:
            songs: list[SonicItem] = await self.conn.get_similar_songs(
                iid=prov_track_id, count=limit
            )
        except DataNotFoundError as e:
            # Subsonic returns an error here instead of an empty list, I don't think this
            # should be an exception but there we are. Return an empty list because this
            # exception means we didn't find anything similar.
            self.logger.info(e)
            return []
        tracks = []
        for entry in songs:
            self._set_loudness(entry)
            lyrics: tuple[str, bool] | None = await self.get_track_lyrics(entry)
            tracks.append(parse_track(self.logger, self.instance_id, entry, lyrics=lyrics))
        return tracks

    async def create_playlist(self, name: str, media_types: set[MediaType]) -> Playlist:
        """Create a new empty playlist on the server."""
        if not await self.conn.create_playlist(name=name):
            raise ProviderPermissionDenied(
                "Please ensure you have permission to create playlists on your server"
            )
        pls: list[SonicPlaylist] = await self.conn.get_playlists()
        for pl in pls:
            if pl.name == name:
                return parse_playlist(self.instance_id, pl)
        raise MediaNotFoundError(
            f"Failed to create playlist with name '{name}'",
            translation_key="create_playlist_failed",
            translation_owner=self.translation_owner,
            translation_args=[name],
        )

    async def add_playlist_tracks(self, prov_playlist_id: str, prov_track_ids: list[str]) -> None:
        """
        Append the listed tracks to the selected playlist.

        Note that the configured user must own the playlist to edit this way.
        """
        try:
            await self.conn.update_playlist(
                lid=prov_playlist_id,
                song_ids_to_add=prov_track_ids,
            )
        except SonicError as ex:
            msg = f"Failed to add songs to {prov_playlist_id}, check your permissions."
            raise ProviderPermissionDenied(msg) from ex

    async def remove_playlist_tracks(
        self, prov_playlist_id: str, positions_to_remove: tuple[int, ...]
    ) -> None:
        """Remove selected positions from the playlist."""
        idx_to_remove = [pos - 1 for pos in positions_to_remove]
        try:
            await self.conn.update_playlist(
                lid=prov_playlist_id,
                song_indices_to_remove=idx_to_remove,
            )
        except SonicError as ex:
            msg = f"Failed to remove songs from {prov_playlist_id}, check your permissions."
            raise ProviderPermissionDenied(msg) from ex

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Get the details needed to process a specified track."""
        item: SonicItem | SonicEpisode
        if media_type == MediaType.TRACK:
            try:
                item = await self.conn.get_song(item_id)
            except (ParameterError, DataNotFoundError) as e:
                msg = f"Item {item_id} not found"
                raise MediaNotFoundError(msg) from e

            mime_type = item.transcoded_content_type or item.content_type

            self.logger.debug(
                "Fetching stream details for id %s '%s' with format '%s'",
                item.id,
                item.title,
                mime_type,
            )

        elif media_type == MediaType.PODCAST_EPISODE:
            item = await self._get_podcast_episode(item_id)

            mime_type = item.transcoded_content_type or item.content_type

            self.logger.debug(
                "Fetching stream details for podcast episode '%s' with format '%s'",
                item.id,
                item.content_type,
            )
        elif media_type == MediaType.RADIO:
            async for station in self.get_library_radios():
                if station.item_id == item_id:
                    return StreamDetails(
                        item_id=item_id,
                        provider=self.instance_id,
                        allow_seek=False,
                        can_seek=False,
                        media_type=MediaType.RADIO,
                        audio_format=AudioFormat(content_type=ContentType.UNKNOWN),
                        stream_type=StreamType.HTTP,
                        path=station.uri or "",
                    )
            msg = f"Radio {item_id} not found"
            raise MediaNotFoundError(msg)
        else:
            msg = f"Unsupported media type encountered '{media_type}'"
            raise UnsupportedFeaturedException(msg)

        fmat = "raw" if self._raw_file else None
        url, _ = self.conn.get_stream_url(item.id, tformat=fmat, estimate_length=True)

        return StreamDetails(
            item_id=item.id,
            provider=self.instance_id,
            allow_seek=True,
            can_seek=True,
            media_type=media_type,
            audio_format=AudioFormat(
                content_type=ContentType.try_parse(mime_type),
                sample_rate=item.sampling_rate or 44100,
                bit_depth=item.bit_depth or 16,
                channels=item.channel_count or 2,
            ),
            stream_type=StreamType.HTTP,
            path=url,
            duration=item.duration or 0,
        )

    async def on_played(
        self,
        media_type: MediaType,
        prov_item_id: str,
        fully_played: bool,
        position: int,
        media_item: MediaItemType,
        is_playing: bool = False,
    ) -> None:
        """
        Handle callback when a (playable) media item has been played.

        This is called by the Queue controller when;
            - a track has been fully played
            - a track has been stopped (or skipped) after being played
            - every 30s when a track is playing

        Fully played is True when the track has been played to the end.

        Position is the last known position of the track in seconds, to sync resume state.
        When fully_played is set to false and position is 0,
        the user marked the item as unplayed in the UI.

        is_playing is True when the track is currently playing.

        media_item is the full media item details of the played/playing track.
        """
        if media_type != MediaType.PODCAST_EPISODE:
            # We don't handle audio books in this provider so this is the only resummable media
            # type we should see.
            return

        _, ep_id = prov_item_id.split(EP_CHAN_SEP)

        if fully_played:
            # We completed the episode and should delete our bookmark
            try:
                await self.conn.delete_bookmark(mid=ep_id)
            except DataNotFoundError:
                # We probably raced with something else deleting this bookmark, not really a problem
                self.logger.info("Bookmark for item '%s' has already been deleted.", ep_id)
            return

        # Otherwise, create a new bookmark for this item or update the existing one
        # MA provides a position in seconds but expects it back in milliseconds
        await self.conn.create_bookmark(
            mid=ep_id,
            position=position * 1000,
            comment="Music Assistant Bookmark",
        )

    async def get_resume_position(
        self, item_id: str, media_type: MediaType
    ) -> tuple[bool, int, datetime | None]:
        """
        Get progress (resume point) details for the given Audiobook or Podcast episode.

        This is a separate call from the regular get_item call to ensure the resume position
        is always up-to-date and because a lot providers have this info present on a dedicated
        endpoint.

        Will be called right before playback starts to ensure the resume position is correct.

        Returns a boolean with the fully_played status
        and an integer with the resume position in ms.
        """
        if media_type != MediaType.PODCAST_EPISODE:
            raise NotImplementedError("AudioBooks are not supported by the Open Subsonic provider")

        _, ep_id = item_id.split(EP_CHAN_SEP)

        bookmarks: list[SonicBookmark] = await self.conn.get_bookmarks()

        for mark in bookmarks:
            if mark.entry.id == ep_id:
                return (
                    False,
                    mark.position,
                    datetime.fromisoformat(mark.created) if mark.created else None,
                )
        # If we get here, there is no bookmark
        return (False, 0, None)

    async def get_track_lyrics(self, track: SonicItem) -> tuple[str, bool] | None:
        """
        Get lyrics for a track.

        Fetches lyrics from Subsonic server. Returns the lyrics text in LRC format
        if the Lyrics are synced (have time stamp info) or raw text if not
        """
        return await self._fetch_track_lyrics(track.id, track.title, track.artist)

    # Library sync asks for every track's lyrics on every run, and Navidrome reads each file
    # to answer; caching per song (misses included) keeps a resync from re-reading the
    # whole library (MUSIC-20).
    @use_cache(3600 * 24)
    async def _fetch_track_lyrics(
        self, song_id: str, title: str | None, artist: str | None
    ) -> tuple[str, bool] | None:
        """Fetch lyrics for one song from the Subsonic server."""
        # Server doesn't support to newer lyrics retrieval, fall back to the old one
        if not self._id_lyrics:
            try:
                ly: SonicLyrics = await self.conn.get_lyrics(title, artist)
            except DataNotFoundError:
                self.logger.debug("Lyrics not found for '%s' by '%s'", title, artist)
                return None
            return (ly.value, False)

        try:
            lyrics: list[StructuredLyrics] = await self.conn.get_lyrics_by_song_id(song_id)
        except DataNotFoundError:
            self.logger.debug("Lyrics not found for '%s'", song_id)
            return None
        if not lyrics:
            return None
        return parse_structured_lyrics(lyrics[0])

    async def _get_podcast_episode(self, eid: str) -> SonicEpisode:
        chan_id, ep_id = eid.split(EP_CHAN_SEP)

        if self._direct_podcast_episode:
            try:
                return await self.conn.get_podcast_episode(ep_id)
            except DataNotFoundError as e:
                msg = f"Can't find episode {ep_id} in podcast {chan_id}"
                raise MediaNotFoundError(msg) from e

        chan = await self.conn.get_podcasts(inc_episodes=True, pid=chan_id)

        if not chan[0].episode:
            raise MediaNotFoundError(f"Missing episode list for podcast channel '{chan[0].id}'")

        for episode in chan[0].episode:
            if episode.id == ep_id:
                return episode

        msg = f"Can't find episode {ep_id} in podcast {chan_id}"
        raise MediaNotFoundError(msg)

    def _set_loudness(self, item: SonicItem) -> None:
        if item.replay_gain and item.replay_gain.track_gain is not None:
            # Convert ReplayGain values (gain in dB) to integrated loudness (LUFS)
            track_loudness = -18 - item.replay_gain.track_gain
            album_loudness = (
                -18 - item.replay_gain.album_gain
                if item.replay_gain.album_gain is not None
                else None
            )
            self.mass.create_task(
                self.mass.streams.audio_analysis.set_track_loudness(
                    item.id,
                    self.instance_id,
                    track_loudness,
                    album_loudness,
                )
            )

    async def _get_podcast_channel_async(self, chan_id: str) -> PodcastChannel | None:
        if cache := await self.mass.cache.get(
            key=chan_id,
            provider=self.instance_id,
            category=CACHE_CATEGORY_PODCAST_CHANNEL,
            base_class=PodcastChannel,
        ):
            return cache
        if channels := await self.conn.get_podcasts(inc_episodes=True, pid=chan_id):
            channel = channels[0]
            await self.mass.cache.set(
                key=chan_id,
                data=channel.to_dict(),
                provider=self.instance_id,
                expiration=600,
                category=CACHE_CATEGORY_PODCAST_CHANNEL,
            )
            return channel
        return None

    @use_cache(3600 * 3, cache_checksum="v2", base_class=RecommendationFolder)
    async def _podcast_recommendations(self) -> RecommendationFolder:
        podcasts: RecommendationFolder = RecommendationFolder(
            item_id="subsonic_newest_podcasts",
            provider=self.instance_id,
            name="Newest Podcast Episodes",
            translation_key="episodes_recently_added",
        )
        sonic_episodes = await self.conn.get_newest_podcasts(count=self._reco_limit)
        for ep in sonic_episodes:
            if channel_info := await self._get_podcast_channel_async(ep.channel_id):
                self._set_loudness(ep)
                podcasts.items.append(parse_epsiode(self.instance_id, ep, channel_info))
        return podcasts

    @use_cache(3600 * 3, cache_checksum="v2", base_class=RecommendationFolder)
    async def _favorites_recommendation(self) -> RecommendationFolder:
        faves: RecommendationFolder = RecommendationFolder(
            item_id="subsonic_starred_albums",
            provider=self.instance_id,
            name="Starred Items",
            translation_key="starred_items",
        )
        starred = await self.conn.get_starred2()
        if starred.album:
            for sonic_album in starred.album[: self._reco_limit]:
                faves.items.append(parse_album(self.logger, self.instance_id, sonic_album))
        if starred.artist:
            for sonic_artist in starred.artist[: self._reco_limit]:
                faves.items.append(parse_artist(self.instance_id, sonic_artist, logger=self.logger))
        if starred.song:
            for sonic_song in starred.song[: self._reco_limit]:
                self._set_loudness(sonic_song)
                lyrics: tuple[str, bool] | None = await self.get_track_lyrics(sonic_song)
                faves.items.append(
                    parse_track(self.logger, self.instance_id, sonic_song, lyrics=lyrics)
                )
        return faves

    @use_cache(3600 * 3, cache_checksum="v2", base_class=RecommendationFolder)
    async def _new_recommendations(self) -> RecommendationFolder:
        new_stuff: RecommendationFolder = RecommendationFolder(
            item_id="subsonic_new_albums",
            provider=self.instance_id,
            name="New Albums",
            translation_key="recently_added_albums",
        )
        new_albums = await self.conn.get_album_list2(ltype="newest", size=self._reco_limit)
        for sonic_album in new_albums:
            new_stuff.items.append(parse_album(self.logger, self.instance_id, sonic_album))
        return new_stuff

    @use_cache(3600 * 3, cache_checksum="v2", base_class=RecommendationFolder)
    async def _played_recommendations(self) -> RecommendationFolder:
        recent: RecommendationFolder = RecommendationFolder(
            item_id="subsonic_most_played",
            provider=self.instance_id,
            name="Most Played Albums",
            translation_key="most_played_albums",
        )
        albums = await self.conn.get_album_list2(ltype="frequent", size=self._reco_limit)
        for sonic_album in albums:
            recent.items.append(parse_album(self.logger, self.instance_id, sonic_album))
        return recent

    async def _enrich_album_with_critical_reception(
        self, album: Album, prov_album_id: str, sonic_album: SonicAlbum | None = None
    ) -> None:
        """
        Populate album CR + album-scope DR by ffprobing one track of the album.

        :param sonic_album: Pre-fetched album record forwarded to the CR fetch so callers
            that already paid for ``conn.get_album`` skip a redundant round-trip on cache miss.
        """
        try:
            cr, album_dr = await self._get_album_critical_reception(prov_album_id, sonic_album)
        except Exception as err:
            self.logger.debug(
                "critical_reception extraction failed for album %s: %s", prov_album_id, err
            )
            return
        if cr is not None:
            album.metadata.critical_reception = cr
        if album_dr is not None:
            album.metadata.dynamic_range = album_dr

    async def _get_album_critical_reception(
        self,
        prov_album_id: str,
        sonic_album: SonicAlbum | None = None,
    ) -> tuple[CriticalReception | None, float | None]:
        """
        Fetch one track of an album, ffprobe it, return (CR, album_dr).

        Cached per album_id for ``CRITICAL_RECEPTION_CACHE_TTL`` so bulk library sync
        doesn't re-ffprobe every album on each run. Only outcomes of a cleanly completed
        probe are cached (including the clean "no tags" negative); transient outcomes —
        album fetch failure, every probe erroring, or the whole-album probe budget of
        ``_CR_PROBE_ALBUM_BUDGET_SECONDS`` running out — return (possibly partial) results
        without writing this cache, so a later sync re-probes. Note that only *this* cache
        is skipped: ``get_album`` carries its own ``@use_cache``, so a transient result
        reached through it is still served from that cache for its own, shorter TTL.

        :param sonic_album: Pre-fetched album record; lets callers that already paid
            for ``conn.get_album`` skip the round-trip on cache miss.
        """
        # Bind the cache key to the configured server URL so a config edit that
        # repoints this provider at a different Subsonic server doesn't return
        # stale CR/DR for an album_id that happens to collide.
        cache_key = f"{self._cr_cache_namespace()}:{prov_album_id}"
        cached = await self.mass.cache.get(
            key=cache_key,
            provider=self.instance_id,
            category=CACHE_CATEGORY_CRITICAL_RECEPTION,
            default=None,
        )
        if cached is not None and not (cached.get("cr") and cached.get("v", 1) < _CR_CACHE_VERSION):
            # A stored entry is itself proof that a probe ran cleanly; legacy
            # {"ok": False} entries decode to (None, None), the same cached
            # clean negative they always meant. Clean negatives stay valid across
            # versions: an album without review tags gains no field to re-read.
            return (
                CriticalReception.from_dict(cr_data) if (cr_data := cached.get("cr")) else None,
                float(dr_data) if (dr_data := cached.get("dr")) is not None else None,
            )
        if sonic_album is None:
            try:
                sonic_album = await self.conn.get_album(prov_album_id)
            except ParameterError, DataNotFoundError:
                sonic_album = None
        if sonic_album is None or not sonic_album.song:
            # Don't cache "no songs" or "fetch failed" — those states can change
            # (user uploads tracks, server comes back) and a month-long negative
            # cache would block a follow-up sync from re-probing.
            return None, None
        # Try a handful of tracks and OR-merge their signals. A bonus / hidden
        # first track may carry CR tags but not ALBUM_DYNAMIC_RANGE, while a
        # later track carries DR but no CR — break out only once both have been
        # observed (or we run out of attempts or time) so neither signal is lost.
        # Cost note: these probes run SEQUENTIALLY and each one can burn the full
        # PARSE_TAGS_TIMEOUT_SECONDS inside ffprobe on top of its stream fetch, so the
        # loop as a whole — not each attempt — is bounded by _CR_PROBE_ALBUM_BUDGET_SECONDS,
        # independent of _CR_PROBE_SONG_ATTEMPTS. Scope is deliberately just this loop: the
        # cache round-trips and the conn.get_album fetch above sit outside it, so the budget
        # bounds the probing, not this method end-to-end. Nothing here can stop an ffprobe
        # already running in an executor thread; the budget bounds the caller's wait, not
        # the work, and a probe abandoned by the timeout runs on until PARSE_TAGS_TIMEOUT_SECONDS
        # kills it. Its temp file is still cleaned up: the `finally` in the probe helper is
        # entered by the CancelledError and gets to run its own awaits, because
        # asyncio.timeout cancels exactly once at the deadline — though those are two more
        # default-executor hops, so a saturated pool can stretch the unwind past the budget.
        cr: CriticalReception | None = None
        album_dr: float | None = None
        # Distinguish "probe ran cleanly and found nothing" from "every probe attempt
        # errored". A clean probe returns a (cr, dr) tuple (possibly (None, None));
        # a transient failure returns bare None and is skipped below.
        probed_clean = False
        unparsable = False
        started_at = asyncio.get_running_loop().time()
        try:
            async with asyncio.timeout(_CR_PROBE_ALBUM_BUDGET_SECONDS):
                for sonic_song in sonic_album.song[:_CR_PROBE_SONG_ATTEMPTS]:
                    probe = await self._extract_critical_reception_from_song(sonic_song.id)
                    if isinstance(probe, _Unparsable):
                        unparsable = True
                        continue
                    if probe is None:
                        continue
                    probed_clean = True
                    probe_cr, probe_dr = probe
                    if cr is None and probe_cr is not None:
                        cr = probe_cr
                    if album_dr is None and probe_dr is not None:
                        album_dr = probe_dr
                    if cr is not None and album_dr is not None:
                        break
        except TimeoutError:
            # Either the budget ran out, or a probe raised a TimeoutError of its own —
            # aiohttp's ServerTimeoutError subclasses it, and conn.stream errors are not
            # covered by the probe helper's `except Exception`. Both abort the loop, so
            # distinguish them by elapsed time rather than logging a budget we may not
            # have spent. Transient either way: hand back whatever earlier probes already
            # produced (possibly (None, None)) but do NOT write the CR cache, or a
            # partial/empty result gets pinned for CRITICAL_RECEPTION_CACHE_TTL. The next
            # sync re-probes and can complete the picture.
            elapsed = asyncio.get_running_loop().time() - started_at
            if elapsed >= _CR_PROBE_ALBUM_BUDGET_SECONDS:
                self.logger.debug(
                    "critical_reception probe budget of %ss exhausted for album %s",
                    _CR_PROBE_ALBUM_BUDGET_SECONDS,
                    prov_album_id,
                )
            else:
                self.logger.debug(
                    "critical_reception probe for album %s timed out after %.1fs "
                    "(inside the %ss budget — likely a stalled stream read)",
                    prov_album_id,
                    elapsed,
                    _CR_PROBE_ALBUM_BUDGET_SECONDS,
                )
            return cr, album_dr
        if not probed_clean:
            if unparsable:
                await self.mass.cache.set(
                    key=cache_key,
                    data={"cr": None, "dr": None},
                    provider=self.instance_id,
                    category=CACHE_CATEGORY_CRITICAL_RECEPTION,
                    expiration=_CR_UNPARSABLE_CACHE_TTL,
                )
                return None, None
            # Every probe attempt errored transiently (stream/ffprobe failure) rather
            # than cleanly finding no tags. Don't pin a month-long negative cache — mirror
            # the "fetch failed" path above so the next sync re-probes once it recovers.
            return None, None
        await self.mass.cache.set(
            key=cache_key,
            data={
                "cr": cr.to_dict() if cr is not None else None,
                "dr": album_dr,
                "v": _CR_CACHE_VERSION,
            },
            provider=self.instance_id,
            category=CACHE_CATEGORY_CRITICAL_RECEPTION,
            expiration=CRITICAL_RECEPTION_CACHE_TTL,
        )
        return cr, album_dr

    def _cr_cache_namespace(self) -> str:
        """
        Stable, URL-derived prefix for the CR cache key.

        Without this prefix the cache key is just ``prov_album_id`` — a config
        edit that swings this provider over to a different Subsonic server
        keeps the same MA provider instance_id but the album IDs no longer
        refer to the same albums, so the cache returns stale CR/DR for the
        new server. Hashing the URL bumps the namespace on any URL change.
        """
        url = str(self.config.get_value(CONF_BASE_URL) or "")
        if not url:
            return "default"
        # md5 is fine here — this is a cache namespace, not a security boundary.
        return hashlib.md5(url.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]

    async def _extract_critical_reception_from_song(
        self, song_id: str
    ) -> tuple[CriticalReception | None, float | None] | _Unparsable | None:
        """
        Stream a small prefix of the song, ffprobe, return (CR, album_dr) or None.

        Album-scope DR is the upstream tag writer's ALBUM_DYNAMIC_RANGE — same value
        on every track, so reading any one of them gives us the album-level number.
        """
        # Pull a fixed prefix of the file via the Subsonic stream endpoint, write it
        # to a temp file, then ffprobe that. Stdin-piping to ffprobe is unreliable
        # for some containers (M4A 'moov' atom can sit before mdat but ffprobe still
        # wants the file size to validate offsets); a temp file sidesteps all of it.
        try:
            resp = await self.conn.stream(song_id, tformat="raw", estimate_length=True)
        except ParameterError, DataNotFoundError:
            return None
        # mkstemp is synchronous (single syscall; not worth the executor hop) and
        # creates the temp file before any await, so the outer try/finally that
        # owns the cleanup is entered without a cancellation hole. We close the fd
        # right away — the prefix is buffered in memory and persisted in one write
        # below. The aiohttp ClientResponse must be released even if mkstemp raises
        # (OSError on tmpdir EACCES / ENFILE / disk full), so wrap the whole flow
        # in `async with resp:` so a mkstemp failure still triggers __aexit__.
        tmp_path: str | None = None
        try:
            async with resp:
                tmp_fd, tmp_path = tempfile.mkstemp(prefix="ma-cr-", suffix=".bin")
                os.close(tmp_fd)
                # Collect the tag-bearing prefix in memory, then persist it in a
                # single executor hop below.
                reader = _TagPrefixReader()
                async for chunk in resp.content.iter_chunked(64 * 1024):
                    if not chunk or reader.feed(chunk):
                        break
            probe_bytes = reader.result()
            if not probe_bytes:
                return None
            await asyncio.to_thread(Path(tmp_path).write_bytes, probe_bytes)
            # No outer asyncio.wait_for here: async_parse_tags runs parse_tags via
            # asyncio.to_thread, so cancelling the await would abandon — not stop —
            # the executor thread. The work is bounded from the inside instead:
            # the ffprobe subprocess is capped by PARSE_TAGS_TIMEOUT_SECONDS (a hung
            # ffprobe is killed there and surfaces here as an exception), and the
            # parse_tags_mutagen pass that follows — which always runs, since
            # tmp_path is a local existing file — is untimed but is a pure in-memory
            # parse of a prefix of at most CRITICAL_RECEPTION_PROBE_BYTES. The caller's
            # wait is bounded separately, per album rather than per probe, by
            # _CR_PROBE_ALBUM_BUDGET_SECONDS.
            try:
                tags = await async_parse_tags(tmp_path)
            except InvalidDataError:
                return _PROBE_UNPARSABLE
            except Exception:
                return None
            return tags.critical_reception, tags.album_dynamic_range
        finally:
            if tmp_path is not None:
                await remove_file(tmp_path)
