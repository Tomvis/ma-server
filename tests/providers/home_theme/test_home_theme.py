"""Tests for the home theme provider: claim, in-app choice and their precedence rule."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.auth import Scope, User, UserRole
from music_assistant_models.errors import InsufficientPermissions, InvalidDataError

from music_assistant.providers import home_theme
from music_assistant.providers.home_theme import HomeThemeProvider


class FakeConfig:
    """The slash-path get/set of the config controller, in memory."""

    def __init__(self) -> None:
        """Start empty."""
        self.data: dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Return a stored value."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Store a value."""
        self.data[key] = value


def _user(username: str, *, admin: bool = False) -> User:
    return User(
        user_id=f"id-{username}",
        username=username,
        role=UserRole.ADMIN if admin else UserRole.USER,
    )


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> HomeThemeProvider:
    """Build a provider on a stubbed mass, signed in as tom (admin)."""
    users = {name: _user(name, admin=name == "tom") for name in ("tom", "lera")}
    current = {"user": users["tom"]}
    monkeypatch.setattr(home_theme, "get_current_user", lambda: current["user"])
    monkeypatch.setattr(
        home_theme,
        "has_scope",
        lambda user, scope: scope == Scope.USERS_MANAGE and user.role == UserRole.ADMIN,
    )
    prov = HomeThemeProvider.__new__(HomeThemeProvider)
    prov.config = MagicMock(instance_id="home_theme")
    prov.mass = MagicMock()
    prov.mass.config = FakeConfig()
    prov.mass.webserver.auth.get_user_by_username = AsyncMock(side_effect=users.get)
    prov.current = current  # type: ignore[attr-defined]
    prov.users = users  # type: ignore[attr-defined]
    return prov


async def _claim(prov: HomeThemeProvider, *args: str) -> dict[str, Any]:
    state = await prov.set_claim(*args)
    assert state is not None
    return state


async def _get(prov: HomeThemeProvider, username: str | None = None) -> dict[str, Any]:
    state = await prov.get_theme(username)
    assert state is not None
    return state


def _sign_in(prov: HomeThemeProvider, username: str) -> None:
    prov.current["user"] = prov.users[username]  # type: ignore[attr-defined]


async def test_claim_is_stored_per_user(provider: HomeThemeProvider) -> None:
    """theme_sync's claim lands on the named user and announces the change."""
    state = await _claim(provider, "lera", "lagoon", "light")
    assert state == {
        "user_id": "id-lera",
        "claim": {"theme": "lagoon", "mode": "light", "override": "follow"},
        "choice": None,
    }
    assert await _get(provider, "lera") == state
    assert (await _get(provider))["claim"] is None
    signal = provider.mass.signal_event
    assert signal.call_args.kwargs["data"] == {"user_id": "id-lera"}  # type: ignore[attr-defined]


async def test_unknown_user_is_none(provider: HomeThemeProvider) -> None:
    """An authentik person without a Music Assistant account is reported, not created."""
    assert await provider.set_claim("nobody", "slate", "dark") is None
    assert await provider.get_theme("nobody") is None


async def test_choice_carries_the_claims_override_as_basis(provider: HomeThemeProvider) -> None:
    """An in-app choice remembers which per-app override it was made against."""
    await _claim(provider, "tom", "slate", "automatic", "slate/dark")
    state = await provider.choose_theme("rave", "dark")
    assert state["choice"] == {"theme": "rave", "mode": "dark", "basis": "slate/dark"}


async def test_global_change_keeps_the_choice(provider: HomeThemeProvider) -> None:
    """A new global theme (same override) never undoes an in-app choice."""
    await _claim(provider, "tom", "slate", "automatic")
    await provider.choose_theme("rave", "dark")
    state = await _claim(provider, "tom", "lagoon", "light")
    assert state["choice"] == {"theme": "rave", "mode": "dark", "basis": "follow"}
    assert state["claim"]["theme"] == "lagoon"


@pytest.mark.parametrize("override", ["mint/dark", "follow"])
async def test_override_change_drops_the_choice(provider: HomeThemeProvider, override: str) -> None:
    """Changing the per-app choice in authentik, back to follow included, wins."""
    await _claim(provider, "tom", "slate", "automatic", "slate/light")
    await provider.choose_theme("rave", "dark")
    state = await _claim(provider, "tom", "mint", "dark", override)
    assert state["choice"] is None


async def test_clearing_the_choice_follows_the_claim(provider: HomeThemeProvider) -> None:
    """Choosing "Follow home theme" in the app removes the choice."""
    await provider.choose_theme("rave", "dark")
    assert (await provider.choose_theme(None))["choice"] is None


async def test_other_users_need_users_manage(provider: HomeThemeProvider) -> None:
    """A regular user reads and chooses only for themself."""
    _sign_in(provider, "lera")
    await provider.choose_theme("lavender", "light")
    assert (await _get(provider))["choice"]["theme"] == "lavender"
    with pytest.raises(InsufficientPermissions):
        await _get(provider, "tom")


@pytest.mark.parametrize(
    ("theme", "mode", "override"),
    [("Slate", "dark", "follow"), ("slate", "dim", "follow"), ("slate", "dark", "x")],
)
async def test_invalid_values_are_refused(
    provider: HomeThemeProvider, theme: str, mode: str, override: str
) -> None:
    """Only theme ids, the three modes and well-formed overrides are stored."""
    with pytest.raises(InvalidDataError):
        await provider.set_claim("tom", theme, mode, override)
