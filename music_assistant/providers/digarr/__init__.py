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

from typing import TYPE_CHECKING, cast

from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.enums import ConfigEntryType, ProviderFeature

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
    ROW_ITEM_TARGET,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

SUPPORTED_FEATURES: set[ProviderFeature] = {
    ProviderFeature.RECOMMENDATIONS,
}


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
        self._client = DigarrClient(
            url=cast("str", config.get_value(CONF_URL, DEFAULT_URL)),
            api_key=cast("str", config.get_value(CONF_API_KEY, "")),
            session=mass.http_session,
        )
        self._ma_user = cast("str", config.get_value(CONF_MA_USER, ""))
        self._row_size = int(cast("int", config.get_value(CONF_ROW_SIZE, ROW_ITEM_TARGET)))
        self._min_score = float(cast("float", config.get_value(CONF_MIN_SCORE, 0.0)))

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
                default_value=self.config.get_value(CONF_URL, DEFAULT_URL),
            ),
            ConfigEntry(
                key=CONF_API_KEY,
                type=ConfigEntryType.SECURE_STRING,
                required=True,
                default_value=self.config.get_value(CONF_API_KEY),
            ),
            ConfigEntry(
                key=CONF_MA_USER,
                type=ConfigEntryType.STRING,
                required=True,
                # An empty list would render as a picker with nothing to pick; None
                # falls back to free text so the field stays usable either way.
                options=users or None,
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

    async def _ma_usernames(self) -> list[ConfigValueOption]:
        """List MA usernames so the bound user is a picker, not free text."""
        # controllers/webserver/auth.py:832. Note it requires the users.read
        # scope, so wrap it: a config page opened without that scope must fall
        # back to free text rather than failing to render at all.
        try:
            users = await self.mass.webserver.auth.list_users()
        except Exception:
            return []
        return [ConfigValueOption(user.username, user.username) for user in users]
