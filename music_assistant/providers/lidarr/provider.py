"""
Lidarr Plugin Provider implementation.

"Add to Lidarr" sends an album straight to Lidarr's API (see ``lidarr_add``). It used
to hand albums to music-rater, which retired its Lidarr subsystem (MEDIA-1).

Each add goes into the acting Music Assistant user's own Lidarr root folder, mapped
per user on the options page; the artist gets that root folder's default profiles.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

from music_assistant_models.auth import Scope, UserRole
from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption, ConfigValueType
from music_assistant_models.constants import SECURE_STRING_SUBSTITUTE
from music_assistant_models.enums import ConfigEntryType, ExternalID, MediaType
from music_assistant_models.errors import InvalidDataError

from music_assistant.constants import CONF_PROVIDERS
from music_assistant.controllers.webserver.helpers.auth_middleware import get_current_user
from music_assistant.helpers.compare import compare_strings
from music_assistant.models.plugin import PluginProvider
from music_assistant.providers.lidarr.client import LidarrClient, LidarrError
from music_assistant.providers.lidarr.constants import (
    CONF_ACTION_TEST,
    CONF_API_KEY,
    CONF_ROOT_FOLDER_PREFIX,
    CONF_URL,
    CONF_VERIFY_SSL,
)
from music_assistant.providers.lidarr.lidarr_add import AlbumRequest, add_album

if TYPE_CHECKING:
    from collections.abc import Callable

    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.media_items import Album
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant


class LidarrProvider(PluginProvider):
    """Plugin provider that sends Music Assistant albums to Lidarr."""

    _client: LidarrClient
    _unregister_handles: list[Callable[[], None]]

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
    ) -> None:
        """Initialize the provider with a bound Lidarr client."""
        super().__init__(mass, manifest, config)
        self._unregister_handles = []
        self._test_ok = False
        self._test_error: str | None = None
        # CONF_URL is collected by the setup flow into setup_data (a fresh instance's
        # values is {}), but an options-page edit lands in values and must win; see
        # _config_or_setup_value. CONF_API_KEY is read from setup_data ONLY: it is a
        # SECURE_STRING, and both config.get_value (through the plain-STRING passthrough
        # entry seeded before rehydrate) and get_setup_value's fallback to it would hand
        # back ciphertext. update_config() mirrors an options-page edit into setup_data.
        # Same scheme as providers/digarr/__init__.py.
        self._client = LidarrClient(
            url=cast("str", self._config_or_setup_value(CONF_URL)),
            api_key=cast("str", self._setup_data_only(CONF_API_KEY, "") or ""),
            session=mass.http_session,
            verify_ssl=bool(config.get_value(CONF_VERIFY_SSL, True)),
        )

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """
        Return the options entries shown for this (loaded) instance.

        CONF_URL and CONF_API_KEY are collected by the setup flow but must be declared
        here too: the framework re-parses the stored config against exactly these
        entries after construction, so a missing key reads back as None forever.

        CONF_URL's ``default_value`` is ``_setup_data_only(CONF_URL)`` and nothing that
        tracks the live value: ``Config.to_raw()`` persists an entry only when
        ``value != default_value``, so a default mirroring the value converges to equal
        it after one reload and the next save of any field silently drops it.

        CONF_API_KEY gets no ``default_value``: ``Config.__post_serialize__`` masks a
        SECURE_STRING's value but not its default, so a default would serve the
        plaintext key to every CONFIG_PROVIDERS_READ caller.

        One root-folder entry per Music Assistant user, offered Lidarr's live root
        folders (free text when Lidarr can't be reached). Optional: a user without a
        mapping gets a clear error from "Add to Lidarr" instead of a guessed folder.
        """
        root_options = await self._root_folder_options()
        user_entries = [
            ConfigEntry(
                key=f"{CONF_ROOT_FOLDER_PREFIX}{username}",
                type=ConfigEntryType.STRING,
                required=False,
                translation_key="root_folder_user",
                translation_params=[username],
                options=root_options,
            )
            for username in await self._usernames()
        ]
        return (
            ConfigEntry(key="intro", type=ConfigEntryType.LABEL),
            ConfigEntry(
                key=CONF_URL,
                type=ConfigEntryType.STRING,
                required=True,
                default_value=self._setup_data_only(CONF_URL),
            ),
            ConfigEntry(key=CONF_API_KEY, type=ConfigEntryType.SECURE_STRING, required=False),
            *user_entries,
            ConfigEntry(
                key=CONF_VERIFY_SSL,
                type=ConfigEntryType.BOOLEAN,
                required=False,
                advanced=True,
                default_value=True,
            ),
            ConfigEntry(key=CONF_ACTION_TEST, type=ConfigEntryType.ACTION, action=CONF_ACTION_TEST),
            ConfigEntry(
                key="test_ok_label",
                type=ConfigEntryType.LABEL,
                required=False,
                hidden=not self._test_ok,
            ),
            ConfigEntry(
                key="test_error_label",
                type=ConfigEntryType.ALERT,
                required=False,
                hidden=self._test_error is None,
                description=self._test_error,
            ),
        )

    async def handle_config_action(self, action: str) -> tuple[ConfigEntry, ...]:
        """Run the connectivity probe behind the 'Test connection' button."""
        if action != CONF_ACTION_TEST:
            return await super().handle_config_action(action)
        self._test_ok = False
        self._test_error = None
        try:
            await self._client.system_status()
            self._test_ok = True
        except Exception as err:
            self._test_error = f"{type(err).__name__}: {err}"
        return await self.get_config_entries()

    async def update_config(self, config: ProviderConfig, changed_keys: set[str]) -> None:
        """
        Mirror a genuine api_key edit into setup_data before the reload the base class schedules.

        setup_data is the only place the client reads the key from (see ``__init__``).
        A caller echoing ``SECURE_STRING_SUBSTITUTE`` back (the MCP config tool can) is
        flagged as changed by ``Config.update`` but is not a rotation, so it is skipped.
        Known gap shared with digarr: this hook isn't called when the instance is
        unavailable at save time; lidarr stays available when Lidarr is down
        (``loaded_in_mass`` only logs a failed probe), so that branch isn't reached here.

        :param config: The freshly saved config (values not yet encrypted).
        :param changed_keys: The dotted keys ("values/<key>") this save changed.
        """
        if f"values/{CONF_API_KEY}" in changed_keys:
            new_key = config.values[CONF_API_KEY].value
            if isinstance(new_key, str) and new_key and new_key != SECURE_STRING_SUBSTITUTE:
                self._update_setup_data(CONF_API_KEY, new_key)
        await super().update_config(config, changed_keys)

    async def loaded_in_mass(self) -> None:
        """
        Register the WebSocket command, probe Lidarr, and flag an ignored Reconfigure.

        The command is registered unconditionally so the action stays visible while
        Lidarr is down -- invocations then fail with a useful error. Raising here would
        leave the provider marked available (the framework swallows post-setup
        exceptions) but with the command silently missing.
        """
        self._unregister_handles.append(
            self.mass.register_api_command(
                "lidarr/add_album", self.add_album, required_scope=Scope.LIBRARY_MANAGE
            )
        )
        try:
            await self._client.system_status()
        except Exception as err:
            self.logger.warning(
                "Lidarr at %s unreachable on load: %s. The 'Add to Lidarr' action will "
                "surface this error on first use.",
                self._sanitized_url(),
                err,
            )

        # _finish_provider_reconfigure only ever writes setup_data, and an explicit
        # `values` url wins over it, so a Reconfigure on an instance with a url in
        # `values` reports success and changes nothing. Warn while the two disagree.
        setup_url = cast("str | None", self._setup_data_only(CONF_URL))
        if setup_url and setup_url.rstrip("/") != self._client._base:
            self.logger.warning(
                "lidarr: url %r was submitted via Reconfigure, but the options-page "
                "value %r is still active and takes priority over it -- edit (or "
                "clear) the url on this instance's options page directly; "
                "submitting Reconfigure again will not change it.",
                setup_url,
                self._client._base,
            )

    async def unload(self, is_removed: bool = False) -> None:
        """Drop the registered command handler."""
        for unregister in self._unregister_handles:
            unregister()
        self._unregister_handles.clear()
        await super().unload(is_removed)

    # ----- public API command -----

    async def add_album(self, item: str) -> dict[str, Any]:
        """
        Send an album (an MA URI) to Lidarr, into the acting user's root folder.

        Returns the frontend's LidarrAddAlbumResult shape.
        """
        root_folder = self._root_folder_for_current_user()
        media_item = await self.mass.music.get_item_by_uri(item)
        if media_item.media_type != MediaType.ALBUM:
            raise InvalidDataError(
                f"lidarr/add_album only accepts albums, got {media_item.media_type.value}"
            )
        album = cast("Album", media_item)
        if not album.name or not album.name.strip():
            raise InvalidDataError(f"Album {item!r} has no usable title")
        if not album.artists or not album.artists[0].name:
            raise InvalidDataError(f"Album {album.name!r} has no usable artist")
        artist_name = album.artists[0].name

        artist_mbid, release_group_mbid = await self._resolve_mbids(album)
        if not artist_mbid:
            artist_mbid = await self._artist_mbid_by_name(artist_name)
        if not artist_mbid:
            raise LidarrError(
                f"Couldn't determine a MusicBrainz artist ID for {artist_name!r}, which "
                "Lidarr needs to add the artist. Try refreshing the album's metadata first."
            )
        result = await add_album(
            self._client,
            AlbumRequest(
                artist_mbid=artist_mbid,
                artist_name=artist_name,
                album_name=album.name,
                release_group_mbid=release_group_mbid,
                release_mbid=album.get_external_id(ExternalID.MB_ALBUM),
            ),
            root_folder=root_folder,
            logger=self.logger,
        )
        return {**result, "lidarr_instance": self._host_port() or self.name}

    # ----- helpers -----

    def _root_folder_for_current_user(self) -> str:
        """Return the acting user's mapped Lidarr root folder, or raise a fixable error."""
        user = get_current_user()
        if user is None:
            raise LidarrError("'Add to Lidarr' needs a signed-in Music Assistant user")
        # Read the stored value directly: the per-user entries are declared only when the
        # user list could be fetched as the config was parsed, and an undeclared key
        # reads back as None through config.get_value.
        key = f"{CONF_ROOT_FOLDER_PREFIX}{user.username}"
        root = self.mass.config.get(f"{CONF_PROVIDERS}/{self.instance_id}/values/{key}")
        if not root:
            raise LidarrError(
                f"No Lidarr root folder is set for {user.username!r}; pick one on the "
                "Lidarr provider's options page."
            )
        return str(root)

    async def _usernames(self) -> list[str]:
        """List MA usernames; empty (no per-user entries) if the lookup is refused."""
        try:
            users = await self.mass.webserver.auth.list_users()
        except Exception as err:
            self.logger.warning("Could not list Music Assistant users: %s", err)
            return []
        # The Home Assistant system user is listed too; it never clicks "Add to Lidarr".
        return sorted(user.username for user in users if user.role != UserRole.SERVICE)

    async def _root_folder_options(self) -> list[ConfigValueOption]:
        """Offer Lidarr's root folders; an empty list renders free text when unreachable."""
        try:
            roots = await self._client.list_root_folders()
        except Exception as err:
            self.logger.debug("Could not list Lidarr root folders: %s", err)
            return []
        return [ConfigValueOption(str(r["path"]), str(r["path"])) for r in roots if r.get("path")]

    async def _resolve_mbids(self, album: Album) -> tuple[str | None, str | None]:
        """
        Best-effort resolve (artist MBID, release-group MBID) for the album.

        1. MBIDs already on the album / its artist mapping.
        2. The full Artist record (an ItemMapping can be sparse).
        3. The MusicBrainz provider, anchored on the album's release (group).
        4. The MusicBrainz provider, anchored on one of the album's tracks.
        """
        release_group = album.get_external_id(ExternalID.MB_RELEASEGROUP)
        artist_mapping = album.artists[0]
        if artist_mbid := artist_mapping.get_external_id(ExternalID.MB_ARTIST):
            return artist_mbid, release_group
        try:
            full_artist = await self.mass.music.artists.get(
                artist_mapping.item_id, artist_mapping.provider
            )
        except Exception as err:
            self.logger.debug("Artist fetch failed during MBID resolve: %s", err)
        else:
            if artist_mbid := full_artist.get_external_id(ExternalID.MB_ARTIST):
                return artist_mbid, release_group

        mb_provider: Any = self.mass.get_provider("musicbrainz")
        if mb_provider is None:
            return None, release_group
        if release_group or album.get_external_id(ExternalID.MB_ALBUM):
            try:
                mb_artist = await mb_provider.get_artist_details_by_album(
                    artist_mapping.name, album
                )
            except Exception as err:
                self.logger.debug("MB lookup-by-album failed: %s", err)
            else:
                if mb_artist and mb_artist.id:
                    return mb_artist.id, release_group
        try:
            tracks = await self.mass.music.albums.tracks(
                album.item_id, album.provider, in_library_only=False
            )
        except Exception as err:
            self.logger.debug("Could not fetch album tracks for MB search: %s", err)
            tracks = []
        for track in tracks[:3]:
            try:
                hit = await mb_provider.search(
                    artist_mapping.name, album.name, track.name, track.version or None
                )
            except Exception as err:
                self.logger.debug("MB track-search failed: %s", err)
                continue
            if hit:
                mb_artist, mb_release_group, _ = hit
                return mb_artist.id, release_group or (
                    mb_release_group.id if mb_release_group else None
                )
        return None, release_group

    async def _artist_mbid_by_name(self, artist_name: str) -> str | None:
        """
        Resolve an artist MBID from the name alone, accepting only an unambiguous match.

        Tried in order: artists already in Lidarr (the likely intent, and no lookup
        needed), a MusicBrainz artist search, then Lidarr's own lookup -- which a
        metadata-proxy plugin can break (Tubifarry's Discogs search 500s it here).
        """
        in_lidarr = {
            str(a["foreignArtistId"])
            for a in await self._client.list_artists()
            if a.get("foreignArtistId")
            and compare_strings(str(a.get("artistName", "")), artist_name, strict=True)
        }
        if len(in_lidarr) == 1:
            return in_lidarr.pop()
        if in_lidarr:
            return None
        if (mbid := await self._musicbrainz_artist_by_name(artist_name)) is not None:
            return mbid
        try:
            candidates = await self._client.lookup_artist(artist_name)
        except Exception as err:
            self.logger.warning("Lidarr lookup-by-name failed for %r: %s", artist_name, err)
            return None
        found = {
            str(c["foreignArtistId"])
            for c in candidates
            if c.get("foreignArtistId")
            and compare_strings(str(c.get("artistName", "")), artist_name, strict=True)
        }
        return found.pop() if len(found) == 1 else None

    async def _musicbrainz_artist_by_name(self, artist_name: str) -> str | None:
        """Return the MBID of the single MusicBrainz artist named exactly ``artist_name``."""
        mb_provider: Any = self.mass.get_provider("musicbrainz")
        if mb_provider is None:
            return None
        escaped = re.sub(r'([+\-&|!(){}\[\]^"~*?:\\/])', r"\\\1", artist_name)
        try:
            result = await mb_provider._api_client.get_data("artist", query=f'artist:"{escaped}"')
        except Exception as err:
            self.logger.debug("MusicBrainz artist search failed for %r: %s", artist_name, err)
            return None
        found = {
            str(a["id"])
            for a in (result or {}).get("artists", [])
            if a.get("id") and compare_strings(str(a.get("name", "")), artist_name, strict=True)
        }
        return found.pop() if len(found) == 1 else None

    def _host_port(self) -> str | None:
        """
        Return the configured URL's hostname[:port] with userinfo stripped, or None.

        urlparse(url).netloc keeps any ``user:pass@`` in front of the host, so this
        rebuilds from hostname/port only and is safe to log or toast.
        """
        parsed = urlparse(str(self._config_or_setup_value(CONF_URL) or ""))
        if not parsed.hostname:
            return None
        if parsed.port is not None:
            return f"{parsed.hostname}:{parsed.port}"
        return parsed.hostname

    def _sanitized_url(self) -> str:
        """Return the configured URL as scheme://host[:port], never with credentials."""
        host = self._host_port()
        url = str(self._config_or_setup_value(CONF_URL) or "")
        if host is None:
            return url
        return f"{urlparse(url).scheme or 'http'}://{host}"

    def _config_or_setup_value(self, key: str, default: ConfigValueType = None) -> ConfigValueType:
        """
        Resolve a setup-collected, options-editable key, preferring an explicit options edit.

        ``get_setup_value`` gives the setup-time value unconditional priority, so a later
        options-page edit (which lands in ``config.values``) would be ignored forever.
        Here an explicit options value wins; only a never-saved key falls back to setup.
        Safe only because ``get_config_entries()`` pins CONF_URL's ``default_value`` to
        ``_setup_data_only`` -- see that method's docstring. Mirrors digarr's helper.

        :param key: The config/setup key to resolve (CONF_URL).
        :param default: Fallback when neither an options edit nor a setup value exists.
        """
        if value := self.config.get_value(key):
            return value
        return self.get_setup_value(key, default)

    def _setup_data_only(self, key: str, default: ConfigValueType = None) -> ConfigValueType:
        """
        Return ``key``'s setup_data value, WITHOUT falling back to its live config value.

        Unlike ``get_setup_value``, which falls back to this field's own current value
        when the key is absent from setup_data. The deployed instance has its url in
        ``values`` only, so that fallback would reproduce the self-erasing default
        ``get_config_entries()`` avoids. Mirrors digarr's helper.

        :param key: The setup data key to look up.
        :param default: Value to return when the key is not present in setup_data.
        """
        return self.mass.config.get_provider_setup_value(self.instance_id, key, default)
