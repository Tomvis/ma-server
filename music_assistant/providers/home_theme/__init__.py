"""
Home theme plugin provider (HW-64, fork-only).

Each person picks a theme and light/dark once on the household's authentik
dashboard; home-monitoring's theme_sync pushes the result here as that user's
*claim*: ``{"theme": "<id>", "mode": "automatic|light|dark", "override": "follow" |
"<theme>/<mode>"}``, where ``override`` is the person's per-app choice for Music
Assistant in authentik.

A person can also pick a theme inside Music Assistant (the *choice*). It is kept
with the claim's override at that moment (its *basis*). The frontend shows the
choice while its basis matches the claim's override, else the claim. A claim with
a different override drops the choice here, so changing the per-app choice in
authentik (back to "follow" included) always wins, while a change of the global
choice never undoes an in-app one.

The home language (HW-190) is held the same way under ``home_language/<user_id>``:
claim ``{"language": "<frontend locale>", "override": "follow" | "<language id>"}``
and a choice ``{"language", "basis"}``, with the same precedence rule.

Kept in settings.json under ``home_theme/<user_id>`` rather than in the user's
preferences: clients write preferences back as one whole object, so a tab left
open would put a stale claim back.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from music_assistant_models.auth import Scope
from music_assistant_models.errors import InsufficientPermissions, InvalidDataError

from music_assistant.controllers.webserver.helpers.auth_middleware import (
    get_current_user,
    has_scope,
)
from music_assistant.models.plugin import PluginProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.enums import ProviderFeature
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

STORAGE_KEY = "home_theme"
LANGUAGE_STORAGE_KEY = "home_language"
FOLLOW = "follow"
MODES = ("automatic", "light", "dark")
THEME_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
LOCALE = re.compile(r"^[a-z]{2,3}([_-][A-Za-z0-9]{2,8})?$")


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    features: set[ProviderFeature] = set()
    return HomeThemeProvider(mass, manifest, config, features)


def theme_value(theme: str, mode: str) -> dict[str, str]:
    """Validate a theme id and mode, returned as a stored value."""
    if not isinstance(theme, str) or not THEME_ID.match(theme):
        raise InvalidDataError(f"Invalid theme id: {theme!r}")
    if mode not in MODES:
        raise InvalidDataError(f"Invalid mode: {mode!r}")
    return {"theme": theme, "mode": mode}


def override_value(override: str) -> str:
    """Validate a per-app override: "follow" or "<theme>/<mode>"."""
    if override != FOLLOW:
        theme, _, mode = str(override).partition("/")
        theme_value(theme, mode)
    return override


def language_value(language: str) -> str:
    """Validate a frontend locale (en, ru_RU, sr_Latn) or a home language id (ru, he)."""
    if not isinstance(language, str) or not LOCALE.match(language):
        raise InvalidDataError(f"Invalid language: {language!r}")
    return language


def language_override_value(override: str) -> str:
    """Validate a per-app language override: "follow" or a home language id."""
    return override if override == FOLLOW else language_value(override)


class HomeThemeProvider(PluginProvider):
    """Builtin provider holding each user's home theme claim and in-app choice."""

    async def handle_async_init(self) -> None:
        """Set up the handle list."""
        self._unregister_handles: list[Callable[[], None]] = []

    async def loaded_in_mass(self) -> None:
        """Register the API commands."""
        self._unregister_handles += [
            self.mass.register_api_command("home_theme/get", self.get_theme),
            self.mass.register_api_command("home_theme/choose", self.choose_theme),
            self.mass.register_api_command(
                "home_theme/claim", self.set_claim, required_scope=Scope.USERS_MANAGE
            ),
            self.mass.register_api_command("home_language/get", self.get_language),
            self.mass.register_api_command("home_language/choose", self.choose_language),
            self.mass.register_api_command(
                "home_language/claim", self.set_language_claim, required_scope=Scope.USERS_MANAGE
            ),
        ]

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload/close of the provider."""
        for unregister in self._unregister_handles:
            unregister()
        self._unregister_handles.clear()

    async def get_theme(self, username: str | None = None) -> dict[str, Any] | None:
        """
        Return a user's claim and choice: the caller's own, or (users.manage) another's.

        :param username: Another user's username; None is the signed-in user.
        :return: {"user_id", "username", "claim", "choice"}, or None for an unknown user.
        """
        user_id = await self._user_id(username)
        if user_id is None:
            return None
        return self._state(user_id)

    async def choose_theme(
        self, theme: str | None = None, mode: str | None = None
    ) -> dict[str, Any]:
        """
        Set the signed-in user's in-app choice, or clear it ("Follow home theme").

        :param theme: Theme id; None clears the choice.
        :param mode: automatic, light or dark.
        """
        user_id = await self._user_id(None)
        assert user_id is not None  # the signed-in user exists
        stored = self._stored(user_id)
        if theme is None:
            stored.pop("choice", None)
        else:
            claim = stored.get("claim") or {}
            stored["choice"] = {
                **theme_value(theme, mode or "automatic"),
                "basis": claim.get("override", FOLLOW),
            }
        return self._save(user_id, stored)

    async def set_claim(
        self, username: str, theme: str, mode: str, override: str = FOLLOW
    ) -> dict[str, Any] | None:
        """
        Store a user's home theme as pushed by theme_sync; a new override drops the choice.

        :param username: The user's username.
        :param theme: Effective theme id for Music Assistant.
        :param mode: Effective mode: automatic, light or dark.
        :param override: The person's per-app choice in authentik.
        :return: The new state, or None for an unknown user.
        """
        claim = {**theme_value(theme, mode), "override": override_value(override)}
        user_id = await self._user_id(username)
        if user_id is None:
            return None
        stored = self._stored(user_id)
        stored["claim"] = claim
        choice = stored.get("choice")
        if choice and choice.get("basis") != override:
            stored.pop("choice")
        return self._save(user_id, stored)

    async def get_language(self, username: str | None = None) -> dict[str, Any] | None:
        """
        Return a user's language claim and choice: the caller's own, or (users.manage) another's.

        :param username: Another user's username; None is the signed-in user.
        :return: {"user_id", "claim", "choice"}, or None for an unknown user.
        """
        user_id = await self._user_id(username)
        if user_id is None:
            return None
        return self._state(user_id, LANGUAGE_STORAGE_KEY)

    async def choose_language(self, language: str | None = None) -> dict[str, Any]:
        """
        Set the signed-in user's in-app language, or clear it ("Follow home language").

        :param language: Frontend locale; None clears the choice.
        """
        user_id = await self._user_id(None)
        assert user_id is not None  # the signed-in user exists
        stored = self._stored(user_id, LANGUAGE_STORAGE_KEY)
        if language is None:
            stored.pop("choice", None)
        else:
            claim = stored.get("claim") or {}
            stored["choice"] = {
                "language": language_value(language),
                "basis": claim.get("override", FOLLOW),
            }
        return self._save(user_id, stored, LANGUAGE_STORAGE_KEY)

    async def set_language_claim(
        self, username: str, language: str, override: str = FOLLOW
    ) -> dict[str, Any] | None:
        """
        Store a user's home language as pushed by theme_sync; a new override drops the choice.

        :param username: The user's username.
        :param language: Effective frontend locale for Music Assistant.
        :param override: The person's per-app language choice in authentik.
        :return: The new state, or None for an unknown user.
        """
        claim = {
            "language": language_value(language),
            "override": language_override_value(override),
        }
        user_id = await self._user_id(username)
        if user_id is None:
            return None
        stored = self._stored(user_id, LANGUAGE_STORAGE_KEY)
        stored["claim"] = claim
        choice = stored.get("choice")
        if choice and choice.get("basis") != override:
            stored.pop("choice")
        return self._save(user_id, stored, LANGUAGE_STORAGE_KEY)

    async def _user_id(self, username: str | None) -> str | None:
        user = get_current_user()
        if user is None:
            raise InsufficientPermissions("Not authenticated")
        if username is None or username == user.username:
            return str(user.user_id)
        if not has_scope(user, Scope.USERS_MANAGE):
            raise InsufficientPermissions("The users.manage scope is required for another user")
        other = await self.mass.webserver.auth.get_user_by_username(username)
        return str(other.user_id) if other else None

    def _stored(self, user_id: str, key: str = STORAGE_KEY) -> dict[str, Any]:
        return dict(self.mass.config.get(f"{key}/{user_id}") or {})

    def _state(self, user_id: str, key: str = STORAGE_KEY) -> dict[str, Any]:
        stored = self._stored(user_id, key)
        return {
            "user_id": user_id,
            "claim": stored.get("claim"),
            "choice": stored.get("choice"),
        }

    def _save(self, user_id: str, stored: dict[str, Any], key: str = STORAGE_KEY) -> dict[str, Any]:
        self.mass.config.set(f"{key}/{user_id}", stored)
        self.signal_provider_event({"user_id": user_id, "setting": key})
        return self._state(user_id, key)
