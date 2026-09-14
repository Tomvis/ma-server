"""
digarr plugin provider.

Surfaces digarr's pending recommendations as a Discover row. digarr scores
artists you do not own; this provider resolves each one to a real library or
streaming-provider item so it can be played, then hands approve/reject/block
back to digarr, which owns the Lidarr side.

One instance per digarr user. The row is gated on the viewing Music Assistant
user because plugin rows bypass the core provider filter.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from music_assistant_models.background_task import TaskSchedule
from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import ConfigEntryType, ExternalID, ProviderFeature
from music_assistant_models.media_items import RecommendationFolder, UniqueList

from music_assistant.controllers.webserver.helpers.auth_middleware import get_current_user
from music_assistant.models.plugin import PluginProvider
from music_assistant.providers.digarr.client import DigarrClient, DigarrError
from music_assistant.providers.digarr.constants import (
    CACHE_CATEGORY_RESOLVED_ITEMS,
    CONF_ACTION_CLEAR_CACHE,
    CONF_ACTION_TEST,
    CONF_API_KEY,
    CONF_MA_USER,
    CONF_MIN_SCORE,
    CONF_ROW_SIZE,
    CONF_URL,
    DEFAULT_URL,
    EVENT_RECOMMENDATIONS_UPDATED,
    REFRESH_TASK_ID,
    RESOLUTION_BUFFER,
    ROW_ID,
    ROW_ITEM_TARGET,
)
from music_assistant.providers.digarr.parsers import resolve_artist

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.media_items import (
        Artist,
        BrowseFolder,
        ItemMapping,
        MediaItemType,
    )
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

SUPPORTED_FEATURES: set[ProviderFeature] = {
    ProviderFeature.RECOMMENDATIONS,
}


def mbid_of(item: Artist | ItemMapping) -> str | None:
    """
    Return an item's MusicBrainz artist id, if it has one.

    Defined at module level (rather than as a method) purely so tests can patch
    it directly.

    :param item: The resolved item to read the identifier from.
    """
    mbid: str | None = item.get_external_id(ExternalID.MB_ARTIST)
    return mbid


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return DigarrProvider(mass, manifest, config, SUPPORTED_FEATURES)


class DigarrProvider(PluginProvider):
    """Contributes a Discover row of digarr's pending recommendations."""

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
        supported_features: set[ProviderFeature] | None = None,
    ) -> None:
        """Initialize the provider with a bound digarr client."""
        super().__init__(mass, manifest, config, supported_features)
        self._test_ok = False
        self._test_error: str | None = None
        # CONF_URL/CONF_API_KEY are collected by the setup flow into setup_data, not
        # values -- a freshly-created instance's config.values is {} (see
        # controllers/config/flows.py's _finish_provider_setup), so config.get_value
        # would silently resolve to DEFAULT_URL / "" on first load. get_setup_value
        # reads setup_data first and falls back to the active config value, so it
        # covers both first load and every load after the options page is re-saved.
        self._client = DigarrClient(
            url=cast("str", self.get_setup_value(CONF_URL, DEFAULT_URL)),
            api_key=cast("str", self.get_setup_value(CONF_API_KEY, "")),
            session=mass.http_session,
        )
        self._ma_user = cast("str", config.get_value(CONF_MA_USER, ""))
        self._row_size = int(cast("int", config.get_value(CONF_ROW_SIZE, ROW_ITEM_TARGET)))
        self._min_score = float(cast("float", config.get_value(CONF_MIN_SCORE, 0.0)))
        # The Discover row's current generation. Populated by _refresh; kept here (rather
        # than in handle_async_init) so a fresh instance always has a well-defined, empty
        # row instead of racing the first refresh with an AttributeError.
        self._items: list[Artist] = []
        self._rec_ids: dict[str, int] = {}
        self._mbid_rec_ids: dict[str, int] = {}

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """
        Return the options entries shown for this (loaded) instance.

        CONF_URL and CONF_API_KEY are collected by the setup flow but must be
        declared here as well: the framework re-parses the stored config against
        exactly these entries after construction, so a key missing from this tuple
        reads back as None for the rest of the instance's life.
        """
        users = await self._ma_usernames()
        return (
            ConfigEntry(
                key=CONF_URL,
                type=ConfigEntryType.STRING,
                required=True,
                default_value=self.get_setup_value(CONF_URL, DEFAULT_URL),
            ),
            ConfigEntry(
                key=CONF_API_KEY,
                type=ConfigEntryType.SECURE_STRING,
                required=True,
                default_value=self.get_setup_value(CONF_API_KEY),
            ),
            ConfigEntry(
                key=CONF_MA_USER,
                type=ConfigEntryType.STRING,
                # Left optional: on first load no user has been picked yet, and a
                # required entry with an unresolvable default (None) fails
                # Config.validate() and rolls the whole instance back before it ever
                # gets to load. The Discover row just doesn't render until one is set.
                required=False,
                # An empty list is the framework's own "no options" value: it's what makes
                # this render as free text instead of an unusable empty picker.
                options=users,
                default_value=self.config.get_value(CONF_MA_USER),
            ),
            ConfigEntry(
                key=CONF_ROW_SIZE,
                type=ConfigEntryType.INTEGER,
                required=False,
                advanced=True,
                default_value=ROW_ITEM_TARGET,
            ),
            ConfigEntry(
                key=CONF_MIN_SCORE,
                type=ConfigEntryType.FLOAT,
                required=False,
                advanced=True,
                default_value=0.0,
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
            ConfigEntry(
                key=CONF_ACTION_CLEAR_CACHE,
                type=ConfigEntryType.ACTION,
                action=CONF_ACTION_CLEAR_CACHE,
                advanced=True,
            ),
        )

    async def handle_config_action(self, action: str) -> tuple[ConfigEntry, ...]:
        """Run the connectivity probe behind the 'Test connection' button."""
        if action == CONF_ACTION_CLEAR_CACHE:
            # Signature is clear(key_filter, category_filter, provider_filter,
            # include_persistent) -- controllers/cache/controller.py:344.
            await self.mass.cache.clear(
                category_filter=CACHE_CATEGORY_RESOLVED_ITEMS,
                provider_filter=self.instance_id,
            )
            return await self.get_config_entries()
        if action != CONF_ACTION_TEST:
            return await super().handle_config_action(action)
        self._test_ok = False
        self._test_error = None
        try:
            username, is_admin = await self._client.whoami()
            self._test_ok = True
            # digarr exposes no endpoint reporting a key's scopes, so this proves
            # identity and reachability only. A missing scope surfaces as a 403
            # on first use, not here.
            self.logger.info("digarr key resolves to user %s (admin=%s)", username, is_admin)
        except DigarrError as err:
            self._test_error = str(err)
        except Exception as err:
            self._test_error = f"{type(err).__name__}: {err}"
        return await self.get_config_entries()

    async def handle_async_init(self) -> None:
        """
        Arm the recurring refresh and an initial delayed populate.

        Runs after ``get_config_entries`` was already resolved, so config-derived
        attributes (``self._ma_user``, ``self._row_size``, ``self._min_score``)
        are already set by the time this runs.
        """
        self._unregister_handles: list[Callable[[], None]] = []
        self.mass.tasks.register_scheduled_task(
            task_id=f"{REFRESH_TASK_ID}_{self.instance_id}",
            name="Refresh digarr recommendations",
            handler=self._refresh,
            schedule=TaskSchedule.hourly(every=6),
            translation_key="refresh_digarr_recommendations",
            translation_owner=self.translation_owner,
        )
        # Delayed so streaming providers have finished loading and can be searched.
        self.mass.call_later(
            20, self._refresh, task_id=f"{REFRESH_TASK_ID}_initial_{self.instance_id}"
        )

    async def unload(self, is_removed: bool = False) -> None:
        """
        Tear down the scheduled refresh and any other registered handles.

        :param is_removed: Whether the provider instance itself is being removed,
            as opposed to a reload; passed through so the task's persisted state
            (e.g. last-run bookkeeping) is dropped only on real removal.
        """
        self.mass.tasks.unregister_scheduled_task(
            f"{REFRESH_TASK_ID}_{self.instance_id}",
            clear_persisted_state=is_removed,
        )
        for unregister in self._unregister_handles:
            unregister()
        self._unregister_handles.clear()
        await super().unload(is_removed)

    async def get_recommendations(self) -> list[RecommendationFolder]:
        """
        Return this instance's single Discover row descriptor, gated to its bound user.

        ``_apply_user_provider_filter`` (controllers/music/controller.py:2796) only
        checks ``ProviderType.MUSIC`` providers, so a plugin row like this one is
        handed to every viewer unless the provider gates itself here. Must do no
        I/O: this runs inside the 5s ``RECOMMENDATIONS_ROWS_TIMEOUT``, so the row
        descriptor is built from state ``_refresh`` already computed.
        """
        if not self._viewer_is_bound_user():
            return []
        return [
            RecommendationFolder(
                item_id=ROW_ID,
                provider=self.instance_id,
                name="digarr — Up Next",
                translation_key=ROW_ID,
                icon="mdi-radar",
            )
        ]

    async def get_recommendation_items(
        self, item_id: str
    ) -> UniqueList[MediaItemType | ItemMapping | BrowseFolder]:
        """
        Return the items backing the Discover row.

        Gated the same way as get_recommendations: the recommendations controller's
        own re-check of ``_apply_user_provider_filter`` does not gate plugin
        providers, so without this a client that already knows the row id and this
        instance id could fetch another user's row items directly. That matters
        beyond row visibility -- a future approve/reject/block action resolves an
        item uri back to a recommendation id via this instance's own API key (i.e.
        as its bound user), so an ungated items call is the first half of one user
        acting as another.

        An empty result is otherwise a valid, deliberate response for the bound
        user too (an empty row still renders so a broken integration looks broken
        rather than absent); an unrecognised ``item_id`` returns the same empty list.

        :param item_id: The item_id of the row, as returned by get_recommendations.
        """
        if not self._viewer_is_bound_user() or item_id != ROW_ID:
            return UniqueList()
        return UniqueList(self._items)

    def recommendation_id_for(self, uri: str, item: Artist | ItemMapping) -> int | None:
        """
        Map a resolved item back to the digarr recommendation id it came from.

        Tries the uri recorded at the last successful refresh first, then falls
        back to the item's MusicBrainz artist id for a uri that has since
        rotated (e.g. a streaming provider reissuing an id). Never falls back to
        a name match: approving the wrong artist triggers a real Lidarr download.

        :param uri: The resolved item's uri.
        :param item: The resolved item itself, used for its MusicBrainz id as a
            fallback lookup key.
        """
        if (rec_id := self._rec_ids.get(uri)) is not None:
            return rec_id
        if (mbid := mbid_of(item)) is not None:
            return self._mbid_rec_ids.get(mbid)
        return None

    async def _ma_usernames(self) -> list[ConfigValueOption]:
        """List MA usernames so the bound user is a picker, not free text."""
        # controllers/webserver/auth.py:832. Note it requires the users.read
        # scope, so wrap it: a config page opened without that scope must fall
        # back to free text rather than failing to render at all.
        try:
            users = await self.mass.webserver.auth.list_users()
        except Exception as err:
            self.logger.warning("Could not list Music Assistant users: %s", err)
            return []
        return [ConfigValueOption(user.username, user.username) for user in users]

    async def _refresh(self) -> None:
        """
        Rebuild the Discover row from digarr's current pending recommendations.

        Candidates below ``self._min_score`` never reach resolution. The rest are
        resolved concurrently (the search semaphore lives in parsers.py), and the
        first ``self._row_size`` that resolve become the new row. The new item
        list and both id maps are built into local variables and assigned to
        instance state only at the end, so a slow or failing refresh keeps
        serving the previous generation instead of blanking the row. A
        ``DigarrError`` is logged and swallowed for the same reason: this runs
        from a background schedule with no caller to raise to.
        """
        try:
            candidates = [
                rec
                for rec in await self._client.get_pending(limit=RESOLUTION_BUFFER)
                if rec.score >= self._min_score
            ]
        except DigarrError as err:
            self.logger.warning("digarr: could not fetch pending recommendations: %s", err)
            return

        resolved = await asyncio.gather(
            *[resolve_artist(rec, self.mass, self.instance_id) for rec in candidates]
        )

        items: list[Artist] = []
        rec_ids: dict[str, int] = {}
        mbid_rec_ids: dict[str, int] = {}
        unresolved: list[str] = []
        for rec, artist in zip(candidates, resolved, strict=True):
            if artist is None:
                unresolved.append(rec.artist_name)
                continue
            if len(items) < self._row_size:
                items.append(artist)
                rec_ids[artist.uri] = rec.id
                mbid_rec_ids[rec.artist_mbid] = rec.id

        self._items = items
        self._rec_ids = rec_ids
        self._mbid_rec_ids = mbid_rec_ids

        if unresolved:
            self.logger.info(
                "digarr: %s of %s recommendations resolved to nothing on any provider: %s",
                len(unresolved),
                len(candidates),
                ", ".join(unresolved[:10]),
            )

        self.signal_provider_event({"event": EVENT_RECOMMENDATIONS_UPDATED})

    def _viewer_is_bound_user(self) -> bool:
        """
        Return whether the current viewer is the Music Assistant user this instance is bound to.

        The single check backing both get_recommendations and get_recommendation_items,
        so the two entry points can never drift apart. Plugin providers are not gated by
        ``_apply_user_provider_filter`` (controllers/music/controller.py:2796), so without
        this every viewer would see -- and could fetch -- every digarr instance's row.
        """
        user = get_current_user()
        return user is not None and user.username == self._ma_user
