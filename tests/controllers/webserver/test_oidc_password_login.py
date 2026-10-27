"""Fork (HW-250): OIDC-only mode hides and refuses built-in password sign-in."""

from __future__ import annotations

from typing import TYPE_CHECKING

from music_assistant.controllers.webserver.helpers.oidc_provider import (
    CONF_OIDC_DISABLE_PASSWORD_LOGIN,
    CONF_OIDC_ENABLED,
    OIDCLoginProvider,
)

from .test_oidc import auth_manager, client, idp, provider

__all__ = ["auth_manager", "client", "idp", "provider"]

if TYPE_CHECKING:
    from music_assistant.controllers.webserver.auth import AuthenticationManager

ADMIN_CREDENTIALS = {"username": "admin", "password": "local-test-password"}


def oidc_only(auth: AuthenticationManager, *, enabled: bool) -> None:
    """Set OIDC-only mode through the webserver configuration."""
    auth.webserver.config.values[CONF_OIDC_DISABLE_PASSWORD_LOGIN].value = enabled


async def provider_ids(auth: AuthenticationManager) -> list[str]:
    """Return the provider IDs offered to sign-in clients."""
    return [p["provider_id"] for p in await auth.get_login_providers()]


async def test_password_sign_in_works_by_default(provider: OIDCLoginProvider) -> None:
    """Without the option both providers are offered and passwords sign in (upstream)."""
    auth = provider.auth_manager
    assert await provider_ids(auth) == ["builtin", "oidc"]
    assert (await auth.authenticate_with_credentials("builtin", ADMIN_CREDENTIALS)).success


async def test_oidc_only_hides_and_refuses_password_sign_in(provider: OIDCLoginProvider) -> None:
    """The option drops builtin from the listing and refuses a correct password."""
    auth = provider.auth_manager
    oidc_only(auth, enabled=True)
    assert await provider_ids(auth) == ["oidc"]
    result = await auth.authenticate_with_credentials("builtin", ADMIN_CREDENTIALS)
    assert not result.success
    assert result.user is None
    assert result.error == "Password sign-in is disabled"


async def test_oidc_only_has_no_effect_without_oidc(provider: OIDCLoginProvider) -> None:
    """A disabled or broken OIDC setup never leaves the server without a way in."""
    auth = provider.auth_manager
    oidc_only(auth, enabled=True)
    auth.webserver.config.values[CONF_OIDC_ENABLED].value = False
    auth._setup_oidc_provider()
    assert await provider_ids(auth) == ["builtin"]
    assert (await auth.authenticate_with_credentials("builtin", ADMIN_CREDENTIALS)).success


async def test_oidc_only_has_no_effect_before_the_first_account(
    provider: OIDCLoginProvider,
) -> None:
    """Initial setup still offers password sign-in while no accounts exist."""
    auth = provider.auth_manager
    oidc_only(auth, enabled=True)
    auth._has_users = False
    assert "builtin" in await provider_ids(auth)


async def test_oidc_only_keeps_passwords_settable(provider: OIDCLoginProvider) -> None:
    """Passwords stay stored, so turning the option off restores sign-in unchanged."""
    auth = provider.auth_manager
    oidc_only(auth, enabled=True)
    assert not (await auth.authenticate_with_credentials("builtin", ADMIN_CREDENTIALS)).success
    oidc_only(auth, enabled=False)
    assert (await auth.authenticate_with_credentials("builtin", ADMIN_CREDENTIALS)).success
