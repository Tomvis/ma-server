"""
Lidarr Plugin Provider implementation (music-rater bridge).

The "Add to Lidarr" action no longer talks to Lidarr directly; it hands the
album off to music-rater (operator-run companion service), which orchestrates
the actual Lidarr sync. This keeps Music Assistant out of the artist /
metadata-profile / root-folder business and lets music-rater own the policy.

Two-step flow per music-rater's documented API:

1. Resolve  GET /api/v1/albums?music_assistant_uri=<uri>
            Music-rater stamps the MA URI on every album it syncs, so this
            normally resolves unambiguously. If the album isn't synced
            (user added it to MA by hand), fall back to ?search=<artist+album>.

2. Queue    POST /api/v1/albums/{album_id}/lidarr/queue
            Idempotent — sets lidarr_manual_add=True and runs an inline single-
            album sync. Returns a LidarrQueueResponse we map back to the
            frontend's LidarrAddAlbumResult shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

from music_assistant_models.auth import Scope
from music_assistant_models.config_entries import ConfigEntry, ConfigValueType
from music_assistant_models.enums import ConfigEntryType, MediaType
from music_assistant_models.errors import InvalidDataError

from music_assistant.helpers.compare import compare_strings
from music_assistant.helpers.util import try_parse_int
from music_assistant.models.plugin import PluginProvider
from music_assistant.providers.lidarr.client import MusicRaterClient, MusicRaterError
from music_assistant.providers.lidarr.constants import (
    CONF_ACTION_TEST,
    CONF_URL,
    CONF_VERIFY_SSL,
)

# How many search candidates to pull when the exact MA-URI resolve misses. The
# top hit isn't trusted blindly (it can be a different edition / same-titled
# record); we scan candidates and accept only one whose artist+title match.
_SEARCH_CANDIDATE_LIMIT = 5

if TYPE_CHECKING:
    from collections.abc import Callable

    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.media_items import Album
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant


def _match_album_id(
    candidates: list[dict[str, Any]], artist_name: str, album_name: str
) -> int | None:
    """
    Return the id of the first candidate whose artist AND title match the request.

    music-rater's search items carry ``artist_name_raw`` / ``album_title_raw`` (always
    populated). We require both to match the requested artist/album (normalized,
    case-insensitive) so a fuzzy search hit for a different album is rejected rather
    than silently queued to Lidarr.
    """
    for item in candidates:
        cand_id = item.get("id")
        cand_artist = item.get("artist_name_raw")
        cand_album = item.get("album_title_raw")
        if not isinstance(cand_id, int):
            continue
        if not isinstance(cand_artist, str) or not isinstance(cand_album, str):
            continue
        if compare_strings(artist_name, cand_artist, strict=True) and compare_strings(
            album_name, cand_album, strict=True
        ):
            return cand_id
    return None


class LidarrProvider(PluginProvider):
    """Plugin provider that bridges Music Assistant albums into Lidarr via music-rater."""

    _client: MusicRaterClient
    _unregister_handles: list[Callable[[], None]]

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
    ) -> None:
        """Initialize the provider with a bound music-rater client."""
        super().__init__(mass, manifest, config)
        self._unregister_handles = []
        self._test_ok = False
        self._test_error: str | None = None
        # CONF_URL is collected by the setup flow into setup_data, not values -- a
        # freshly-created instance's config.values is {} (see controllers/config/
        # flows.py's _finish_provider_setup) -- but it is also a declared, editable
        # options entry, so an explicit options-page edit (which lands in `values`)
        # must be able to permanently override the setup-collected value.
        # _config_or_setup_value is what gives that edit priority; see its docstring
        # for how it tells "never edited" apart from "edited to this same value".
        # Reading plain config.get_value(CONF_URL) here (as an earlier version did)
        # missed setup_data entirely, so any instance added after the setup flow
        # started collecting the url there built a client from url=None and crashed
        # immediately on MusicRaterClient's `url.rstrip("/")`.
        self._client = MusicRaterClient(
            url=cast("str", self._config_or_setup_value(CONF_URL)),
            session=mass.http_session,
            verify_ssl=bool(config.get_value(CONF_VERIFY_SSL, True)),
        )

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """
        Return the options entries shown for this (loaded) instance.

        CONF_URL is collected by the setup flow but must be declared here as well:
        the framework re-parses the stored config against exactly these entries
        after construction, so a key that is missing from this tuple reads back as
        None for the rest of the instance's life.

        CONF_URL's ``default_value`` is deliberately ``self._setup_data_only(CONF_URL)``
        rather than mirroring ``self.config.get_value(CONF_URL)`` (as an earlier version
        of this method did): ``Config.to_raw()`` persists an entry only when ``value !=
        default_value``, so a default that tracks the field's own current value
        converges to equal it after one reload -- and the *next* save of *any* field at
        all then silently drops this one from storage. That was a live bug: the
        deployed instance has ``url`` in ``values`` and nothing in ``setup_data``, so
        losing it left ``required=True`` with neither a value nor a default, which
        fails ``Config.validate()`` and stops the instance loading -- and, on creation,
        makes ``_create_provider_instance`` delete the just-created config outright.
        ``_setup_data_only`` reads *only* setup_data (never falling back to this same
        field's own current value the way ``get_setup_value`` does), so this default can
        never converge to equal ``entry.value``: the deployed instance's setup_data has
        no ``url`` at all, so its default resolves to ``None`` forever, which will never
        equal a real configured URL. ``Config.parse`` already overlays the stored value
        as ``entry.value`` on every load -- that is what the options page actually
        renders -- so ``default_value`` only needs to cover the genuinely-unset case.
        Mirrors providers/digarr/__init__.py's identical CONF_URL/CONF_MA_USER hazard
        and its ``_setup_data_only`` fix.

        A bare ``None`` here (no ``default_value=`` at all, or a literal ``None``)
        would also be rejected by ``scripts/check_config_entries.py``: a required
        options entry must always have *something* to resolve to without user input,
        even if that something is a placeholder. ``_setup_data_only``'s call is opaque
        to that (deliberately static) check, but is honest about the actual runtime
        behaviour: this entry has no real default beyond "whatever the setup flow (or
        an options-page edit, via ``entry.value``) provided".

        Note the asymmetry this creates: for a fresh instance (url only in
        setup_data), that url becomes ``entry.default_value`` here, and
        ``Config.__post_serialize__`` masks a SECURE_STRING's ``value`` but never
        touches ``default_value`` of any type -- so a url containing embedded
        ``user:pass@`` credentials is served in plaintext to any caller with
        ``CONFIG_PROVIDERS_READ``, not just this instance's owner. Unavoidable given
        ``required=True`` needing *some* default (see above) -- but worth naming,
        since this provider otherwise strips userinfo everywhere a url reaches a log
        or a toast (``_sanitized_url``, ``_host_port``).
        """
        return (
            ConfigEntry(
                key="intro",
                type=ConfigEntryType.LABEL,
            ),
            ConfigEntry(
                key=CONF_URL,
                type=ConfigEntryType.STRING,
                required=True,
                default_value=self._setup_data_only(CONF_URL),
            ),
            ConfigEntry(
                key=CONF_VERIFY_SSL,
                type=ConfigEntryType.BOOLEAN,
                required=False,
                advanced=True,
                default_value=True,
            ),
            ConfigEntry(
                key=CONF_ACTION_TEST,
                type=ConfigEntryType.ACTION,
                action=CONF_ACTION_TEST,
            ),
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
            await self._client.ping()
            self._test_ok = True
        except MusicRaterError as err:
            self._test_error = str(err)
        except Exception as err:
            self._test_error = f"{type(err).__name__}: {err}"
        return await self.get_config_entries()

    async def loaded_in_mass(self) -> None:
        """
        Register the WebSocket command, probe music-rater, and flag an ignored Reconfigure.

        The command is registered unconditionally so users still see the action
        in the UI when music-rater is down — invocations will fail with a useful
        error from the client. Raising here would leave the provider marked
        available (the framework swallows post-setup exceptions) but with the
        command silently missing.
        """
        self._unregister_handles.append(
            self.mass.register_api_command(
                "lidarr/add_album", self.add_album, required_scope=Scope.LIBRARY_MANAGE
            )
        )
        try:
            await self._client.ping()
        except Exception as err:
            self.logger.warning(
                "music-rater at %s unreachable on load: %s. The 'Add to Lidarr' "
                "action will surface this error on first use.",
                self._sanitized_url(),
                err,
            )

        # _finish_provider_reconfigure (controllers/config/flows.py) only ever writes
        # setup_data, never `values` -- and _config_or_setup_value always prefers an
        # explicit `values` entry. So on an instance that already has a url in
        # `values` (the deployed one), a Reconfigure that submits a new url is
        # accepted, the reload reports success, last_error clears -- and the client
        # keeps using the old, options-page url. Nothing else would tell the admin
        # their Reconfigure silently did nothing, so warn once per load while the two
        # disagree. Compared with the *active* url (not `values` directly) so the
        # warning clears itself the moment either side is edited to match.
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
        Send an album to Lidarr via music-rater.

        - Resolves the album from MA (`item` is an MA URI) for the toast labels.
        - Looks up the music-rater album_id by URI; falls back to text search.
        - POSTs to music-rater's lidarr/queue endpoint and maps the response
          back to the frontend's LidarrAddAlbumResult shape.
        """
        media_item = await self.mass.music.get_item_by_uri(item)
        if media_item.media_type != MediaType.ALBUM:
            raise InvalidDataError(
                f"lidarr/add_album only accepts albums, got {media_item.media_type.value}"
            )
        album = cast("Album", media_item)
        if not album.name or not album.name.strip():
            # Falling through to the resolver with a blank name would build a
            # search query of just the artist and let music-rater hand back
            # whatever happens to lead that artist's catalog.
            raise InvalidDataError(f"Album {item!r} has no usable title")
        if not album.artists:
            raise InvalidDataError(f"Album {album.name!r} has no artist information")
        artist_name = album.artists[0].name
        if not artist_name:
            raise InvalidDataError(f"Album {album.name!r} has no usable artist name")

        album_id = await self._resolve_album_id(
            album_uri=item, artist_name=artist_name, album_name=album.name
        )
        response = await self._client.queue_lidarr(album_id)
        return self._build_result(response, artist_name=artist_name, album_name=album.name)

    # ----- helpers -----

    def _host_port(self) -> str | None:
        """
        Return the configured URL's hostname[:port] with userinfo stripped, or None.

        urlparse(url).netloc keeps the `user:pass@` userinfo in front of the host, so
        deriving host:port from it would leak embedded credentials. This rebuilds from
        hostname/port only, so callers can safely log or toast the result.

        Reads through ``_config_or_setup_value`` (not plain ``config.get_value``) for
        the same reason ``__init__`` does -- see that call site's comment.
        """
        parsed = urlparse(str(self._config_or_setup_value(CONF_URL) or ""))
        if not parsed.hostname:
            return None
        if parsed.port is not None:
            return f"{parsed.hostname}:{parsed.port}"
        return parsed.hostname

    def _sanitized_url(self) -> str:
        """
        Return the configured music-rater URL with any userinfo stripped.

        Reassembled as scheme://host[:port] so logging or toasting the result can
        never leak credentials embedded in the configured URL.
        """
        host = self._host_port()
        url = str(self._config_or_setup_value(CONF_URL) or "")
        if host is None:
            return url
        scheme = urlparse(url).scheme or "http"
        return f"{scheme}://{host}"

    async def _resolve_album_id(self, *, album_uri: str, artist_name: str, album_name: str) -> int:
        """Resolve a music-rater album_id with URI-then-search fallback."""
        album_id = await self._client.resolve_by_uri(album_uri)
        if album_id is not None:
            self.logger.debug("music-rater resolved URI=%s -> album_id=%d", album_uri, album_id)
            return album_id

        # URI didn't hit. Music-rater hasn't synced this album to MA yet (likely
        # added by hand). Best-effort text search — but a free-text search ranks
        # fuzzily and can return a different edition / same-titled record, so we
        # only accept a candidate whose artist AND title match the request rather
        # than blindly queueing the top hit (which would silently sync the wrong
        # album while reporting success under the requested names).
        query = f"{artist_name} {album_name}".strip()
        self.logger.info(
            "music-rater URI lookup empty for %r — falling back to search %r",
            album_uri,
            query,
        )
        candidates = await self._client.resolve_by_search(query, limit=_SEARCH_CANDIDATE_LIMIT)
        album_id = _match_album_id(candidates, artist_name, album_name)
        if album_id is None:
            raise InvalidDataError(
                f"music-rater doesn't know {artist_name!r} - {album_name!r}. "
                "Sync it from music-rater to MA first, or add it to music-rater."
            )
        self.logger.debug(
            "music-rater search %r matched album_id=%d for %r - %r",
            query,
            album_id,
            artist_name,
            album_name,
        )
        return album_id

    @staticmethod
    def _as_count(value: Any) -> int:
        """Coerce a music-rater counter field to int; default to 0 on garbage."""
        return try_parse_int(value) or 0

    def _build_result(
        self,
        response: dict[str, Any],
        *,
        artist_name: str,
        album_name: str,
    ) -> dict[str, Any]:
        """
        Map music-rater's LidarrQueueResponse to the frontend's LidarrAddAlbumResult.

        music-rater fields we read:
          artists_added, albums_monitored, skipped, errors, error_log, lidarr_synced
        """
        artists_added = self._as_count(response.get("artists_added"))
        albums_monitored = self._as_count(response.get("albums_monitored"))
        skipped = self._as_count(response.get("skipped"))
        errors = self._as_count(response.get("errors"))
        lidarr_synced = bool(response.get("lidarr_synced"))

        if errors > 0:
            err_log = response.get("error_log") or "(no error log)"
            # Upstream Lidarr per-album error (bad MBID / profile mismatch /
            # root-folder denial). Music-rater answered fine — this is an
            # application-level failure, not "provider unavailable", and must
            # not trip MA's framework-level provider-down retry path.
            raise MusicRaterError(f"music-rater reported {errors} Lidarr error(s): {err_log}")
        if skipped > 0:
            # Use the sanitized URL helper so any embedded credentials in CONF_URL
            # don't end up in the toast shown to admins.
            base = self._sanitized_url().rstrip("/")
            raise InvalidDataError(
                f"Lidarr couldn't match {artist_name!r} - {album_name!r}. "
                f"Resolve manually at {base}/lidarr/unmatched."
            )

        # albums_monitored == 1 → newly monitored now.
        # lidarr_synced=True with albums_monitored == 0 → idempotent re-call,
        # i.e. the album was already monitored before this request. The
        # artists_added == 0 guard matters: lidarr_synced only means an inline
        # sync ran, not that *this album* was already monitored. When a brand-
        # new artist is added the sync runs (lidarr_synced=True) yet
        # albums_monitored == 0, because Lidarr hasn't refreshed the new
        # artist's metadata and its albums don't exist in Lidarr's DB to
        # monitor yet. Without this guard that case is misread as "already
        # monitored" and reported as a false success.
        already_monitored = lidarr_synced and albums_monitored == 0 and artists_added == 0
        # New artist added but no album monitored: Lidarr accepted the artist
        # but its albums aren't available to monitor until the artist-metadata
        # refresh completes. This is the bug the user hits — the artist lands in
        # Lidarr while the album is silently never monitored, yet the call would
        # otherwise return success and the frontend would toast "added". Surface
        # it as a per-request error; a retry once Lidarr has refreshed the
        # artist will pick up the album.
        if artists_added > 0 and albums_monitored == 0:
            raise MusicRaterError(
                f"Lidarr added the artist {artist_name!r} but couldn't monitor "
                f"{album_name!r} yet — its catalog hasn't been refreshed. "
                "Try 'Add to Lidarr' again in a minute."
            )
        # No counters and no sync flag means music-rater accepted the POST but
        # neither monitored nor reported activity — surface as a malformed
        # backend response (per-request error, not provider-wide outage).
        if not (albums_monitored > 0 or already_monitored):
            raise MusicRaterError(
                f"music-rater returned no-op for {artist_name!r} - {album_name!r} "
                "(no errors, no monitors, no sync). Check music-rater logs."
            )
        return {
            "artist_name": artist_name,
            "album_name": album_name,
            "artist_added": artists_added > 0,
            "album_monitored": albums_monitored > 0 or already_monitored,
            "already_monitored": already_monitored,
            "lidarr_instance": self._music_rater_label(),
        }

    def _music_rater_label(self) -> str:
        """
        Human-readable identifier for the music-rater backend that handled this call.

        Frontend toasts use this to tell the operator which music-rater is
        acting when they've configured several. `self.name` is the MA-side
        display label (often just "Lidarr"), so we fall back to the configured
        URL's host:port — that's the only stable identity we have for the
        upstream service.
        """
        return self._host_port() or self.name

    def _config_or_setup_value(self, key: str, default: ConfigValueType = None) -> ConfigValueType:
        """
        Resolve a setup-collected, options-editable key, preferring an explicit options edit.

        ``get_setup_value`` gives the value collected once at setup time unconditional
        priority, so an edit made later on the options page -- which lands in
        ``config.values``, not ``setup_data`` -- would otherwise be silently ignored
        forever. This reverses that: an explicit options value wins, and only when the
        key has never actually been saved through the options page does the
        setup-collected value apply.

        CONF_URL here is a plain STRING, not a SECURE_STRING (digarr's CONF_API_KEY is
        the one that needs a dedicated read path for that reason) -- ``config.get_value``
        never needs to decrypt a STRING entry, so no ciphertext hazard applies to this
        method at all. It exists purely for the "never edited" vs "edited to this same
        value" distinction below.

        Telling "never edited" apart from "edited to a value that happens to match a
        default" differs before and after ``rehydrate_provider_config`` runs (right
        after construction, before validation and async init -- so most call sites,
        other than ``__init__`` itself, run after it):

        * Before it (in ``__init__``): a key the options page has never saved is not
          present in ``self.config.values`` at all yet (the pre-rehydrate config is
          seeded only with whatever is genuinely in stored ``values``), so
          ``config.get_value`` reads back falsy/None for it -- never a baked-in
          default -- so the setup value can never be shadowed there.
        * After it: every declared key IS present, with ``entry.value`` already
          resolved to either the explicit stored value or ``entry.default_value``.
          This still can't shadow the setup value -- but only because
          ``get_config_entries()``'s own ``default_value=`` for CONF_URL is pinned to
          ``_setup_data_only(CONF_URL)`` and NOTHING that varies with the live value
          (see that method's own docstring for why even ``get_setup_value`` is unsafe
          there). Computing that default from this method (or from anything else that
          mirrors the current value) would make ``entry.value`` converge to equal
          ``entry.default_value`` after one reload -- and since ``Config.to_raw``
          persists an entry only when they differ, the *next* save of any field at
          all would then silently erase this one from storage. Never do that.

        Mirrors providers/digarr/__init__.py's identical helper (there needed for
        CONF_URL and CONF_MA_USER; here only for CONF_URL).

        :param key: The config/setup key to resolve (CONF_URL).
        :param default: Value to fall back to when neither an options edit nor a
            setup value is present.
        """
        if value := self.config.get_value(key):
            return value
        return self.get_setup_value(key, default)

    def _setup_data_only(self, key: str, default: ConfigValueType = None) -> ConfigValueType:
        """
        Return ``key``'s setup_data value, WITHOUT ever falling back to its live config value.

        The only safe source for ``get_config_entries()``'s ``default_value=`` -- see
        that method's own docstring. Unlike ``self.get_setup_value``, which falls back
        to ``self.get_config_value`` (i.e. this same field's own current value) when the
        key is absent from setup_data, this falls back only to the static ``default``
        given here. That distinction is not academic: the deployed instance predates
        CONF_URL being collected at setup and has it in ``values`` alone, nothing in
        setup_data, so ``get_setup_value(CONF_URL)`` there resolves right back to the
        live ``values`` entry -- reproducing the exact self-erasing default this method
        exists to avoid. Mirrors providers/digarr/__init__.py's identical helper.

        :param key: The setup data key to look up.
        :param default: Value to return when the key is not present in setup_data.
        """
        return self.mass.config.get_provider_setup_value(self.instance_id, key, default)
