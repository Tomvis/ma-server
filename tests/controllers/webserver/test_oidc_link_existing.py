"""Fork (HW-221): opt-in linking of a first OIDC sign-in to the existing account by username."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from music_assistant_models.auth import AuthProviderType, UserRole
from music_assistant_models.errors import AuthenticationFailed

from music_assistant.controllers.webserver.helpers.auth_providers import BuiltinLoginProvider
from music_assistant.controllers.webserver.helpers.oidc_deployment import (
    OIDC_CONFIG_FILE_ENV,
    load_oidc_deployment,
)
from music_assistant.controllers.webserver.helpers.oidc_provider import (
    CONF_OIDC_LINK_EXISTING_USERS,
    CONF_OIDC_ROLE_MAPPING,
    OIDCLoginProvider,
    oidc_identity_id,
)

from .test_oidc import CLIENT_ID, CLIENT_SECRET, MA_URL, auth_manager, client, idp, provider

__all__ = ["auth_manager", "client", "idp", "provider"]

if TYPE_CHECKING:
    from music_assistant_models.auth import User

    from .test_oidc import IdentityProvider


def configure(provider: OIDCLoginProvider, *, link: bool, mapping: str = "") -> OIDCLoginProvider:
    """Reload the provider through its normal configuration path."""
    auth = provider.auth_manager
    auth.webserver.config.values[CONF_OIDC_LINK_EXISTING_USERS].value = link
    auth.webserver.config.values[CONF_OIDC_ROLE_MAPPING].value = mapping
    auth._setup_oidc_provider()
    result = auth.login_providers["oidc"]
    assert isinstance(result, OIDCLoginProvider)
    return result


async def local_user(provider: OIDCLoginProvider, username: str, role: UserRole) -> User:
    """Create a password account with a player filter, like the prod family accounts."""
    builtin = provider.auth_manager.login_providers["builtin"]
    assert isinstance(builtin, BuiltinLoginProvider)
    return await builtin.create_user_with_password(
        username, f"{username}-password", role=role, player_filter=["kitchen"]
    )


async def oidc_links(provider: OIDCLoginProvider, user: User) -> list[str]:
    """Return the OIDC identities linked to an account."""
    rows = await provider.auth_manager.database.get_rows(
        "user_auth_providers",
        {"user_id": user.user_id, "provider_type": AuthProviderType.OIDC.value},
    )
    return [row["provider_user_id"] for row in rows]


async def sign_in(provider: OIDCLoginProvider, idp: IdentityProvider) -> User:
    """Complete a signed code exchange."""
    return await provider.complete_authorization(idp.pending, {"code": "valid-code"})


async def test_linking_is_off_by_default(
    provider: OIDCLoginProvider, idp: IdentityProvider
) -> None:
    """Without the option a matching username still gets a separate account (upstream)."""
    existing = await local_user(provider, "listener", UserRole.USER)
    user = await sign_in(provider, idp)
    assert user.user_id != existing.user_id
    assert await oidc_links(provider, existing) == []


async def test_first_sign_in_links_the_existing_account_and_keeps_it(
    provider: OIDCLoginProvider, idp: IdentityProvider
) -> None:
    """The account keeps its id, role and player filter; password login keeps working."""
    existing = await local_user(provider, "listener", UserRole.ADMIN)
    provider = configure(provider, link=True)
    provider._oidc_config["allow_self_registration"] = False
    user = await sign_in(provider, idp)
    assert user.user_id == existing.user_id
    assert user.role == UserRole.ADMIN
    assert user.player_filter == ["kitchen"]
    identity = oidc_identity_id(idp.issuer, "subject-1")
    assert await oidc_links(provider, existing) == [identity]
    # the next sign-in uses the link, even after a username change at the IdP
    idp.overrides["preferred_username"] = "renamed"
    assert (await sign_in(provider, idp)).user_id == existing.user_id
    local = await provider.auth_manager.authenticate_with_credentials(
        "builtin", {"username": "listener", "password": "listener-password"}
    )
    assert local.success
    assert local.user
    assert local.user.user_id == existing.user_id


async def test_an_account_linked_to_another_identity_is_not_claimed(
    provider: OIDCLoginProvider, idp: IdentityProvider
) -> None:
    """A second identity with the same username gets a separate account."""
    existing = await local_user(provider, "listener", UserRole.USER)
    provider = configure(provider, link=True)
    assert (await sign_in(provider, idp)).user_id == existing.user_id
    idp.overrides["sub"] = "subject-2"
    other = await sign_in(provider, idp)
    assert other.user_id != existing.user_id
    assert other.username.startswith("listener-")
    assert len(await oidc_links(provider, existing)) == 1


async def test_mapped_role_applies_to_the_linked_account(
    provider: OIDCLoginProvider, idp: IdentityProvider
) -> None:
    """Group mapping updates the linked account's role like any linked OIDC account."""
    existing = await local_user(provider, "listener", UserRole.GUEST)
    provider = configure(provider, link=True, mapping='{"family":"user"}')
    idp.overrides["groups"] = ["family"]
    user = await sign_in(provider, idp)
    assert user.user_id == existing.user_id
    assert user.role == UserRole.USER


async def test_denied_mapping_leaves_no_link(
    provider: OIDCLoginProvider, idp: IdentityProvider
) -> None:
    """A sign-in the mapping refuses neither links nor changes the account."""
    existing = await local_user(provider, "listener", UserRole.USER)
    provider = configure(provider, link=True, mapping='{"family":"user"}')
    idp.overrides["groups"] = ["strangers"]
    with pytest.raises(AuthenticationFailed, match="role mapping denied"):
        await sign_in(provider, idp)
    assert await oidc_links(provider, existing) == []
    unchanged = await provider.auth_manager.get_user(existing.user_id)
    assert unchanged
    assert unchanged.role == UserRole.USER


async def test_disabled_account_is_refused_and_not_linked(
    provider: OIDCLoginProvider, idp: IdentityProvider
) -> None:
    """A disabled matching account blocks the sign-in instead of spawning a new one."""
    existing = await local_user(provider, "listener", UserRole.USER)
    await provider.auth_manager.database.update(
        "users", {"user_id": existing.user_id}, {"enabled": 0}
    )
    provider = configure(provider, link=True)
    with pytest.raises(AuthenticationFailed, match="disabled"):
        await sign_in(provider, idp)
    assert await oidc_links(provider, existing) == []


async def test_mounted_configuration_accepts_the_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """link_existing_users is a boolean setting of the mounted configuration."""
    path = tmp_path / "oidc.json"
    settings = {
        "base_url": MA_URL,
        "issuer": "https://auth.example/application/o/ma/",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "link_existing_users": True,
    }
    path.write_text(json.dumps(settings))
    monkeypatch.setenv(OIDC_CONFIG_FILE_ENV, str(path))
    deployment = await load_oidc_deployment()
    assert deployment
    assert deployment.provider.get("link_existing_users") is True
    path.write_text(json.dumps({**settings, "link_existing_users": "yes"}))
    with pytest.raises(ValueError, match="must be a boolean"):
        await load_oidc_deployment()
