"""Tests for the digarr provider's configuration surface."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import ConfigEntryType, ProviderFeature, ProviderType

from music_assistant.providers.digarr import SUPPORTED_FEATURES, DigarrProvider
from music_assistant.providers.digarr.client import DigarrAuthError
from music_assistant.providers.digarr.constants import (
    CONF_ACTION_TEST,
    CONF_API_KEY,
    CONF_MA_USER,
    CONF_URL,
)


@pytest.fixture
def provider() -> DigarrProvider:
    """Construct the provider with a stubbed mass/manifest/config."""
    mass = MagicMock()
    mass.http_session = MagicMock()
    # Provider.get_setup_value() reads setup_data (mass.config.get(...)) first and
    # falls through to config.values / config.get_value when nothing was collected
    # by the setup flow. Give the fallthrough real (empty) shapes -- a bare
    # MagicMock() for either is truthy/non-None and short-circuits the real logic
    # into a nonsense chain rather than reaching the mocked config.get_value below.
    mass.config.get = MagicMock(return_value={})
    mass.config.decrypt_string = MagicMock(side_effect=lambda value: value)
    manifest = MagicMock()
    manifest.type = ProviderType.PLUGIN
    manifest.domain = "digarr"
    config = MagicMock()
    config.name = "digarr - Tom"
    config.instance_id = "digarr--abcd1234"
    config.values = {}
    values = {CONF_URL: "http://digarr:3000", CONF_API_KEY: "dgr_x_y", CONF_MA_USER: "tom"}
    config.get_value = MagicMock(side_effect=lambda key, default=None: values.get(key, default))
    return DigarrProvider(mass, manifest, config, SUPPORTED_FEATURES)


async def test_declares_only_recommendations(provider: DigarrProvider) -> None:  # noqa: ARG001
    """The provider owns no music of its own; it contributes rows."""
    assert {ProviderFeature.RECOMMENDATIONS} == SUPPORTED_FEATURES


async def test_config_entries_declare_url_and_api_key(provider: DigarrProvider) -> None:
    """
    Both setup-flow keys reappear as options entries.

    The framework re-parses stored config against exactly this tuple after
    construction, so a key missing here reads back as None forever.
    """
    entries = {entry.key: entry for entry in await provider.get_config_entries()}
    assert CONF_URL in entries
    assert CONF_API_KEY in entries
    assert entries[CONF_API_KEY].type is ConfigEntryType.SECURE_STRING
    assert entries[CONF_ACTION_TEST].type is ConfigEntryType.ACTION


async def test_test_connection_reports_the_resolved_identity(provider: DigarrProvider) -> None:
    """A successful probe names the digarr user the key resolves to."""
    provider._client.whoami = AsyncMock(return_value=("tom", False))
    entries = {entry.key: entry for entry in await provider.handle_config_action(CONF_ACTION_TEST)}
    assert entries["test_ok_label"].hidden is False
    assert entries["test_error_label"].hidden is True


async def test_test_connection_surfaces_an_auth_failure(provider: DigarrProvider) -> None:
    """A revoked key produces a visible alert rather than a silent no-op."""
    provider._client.whoami = AsyncMock(side_effect=DigarrAuthError("key revoked"))
    entries = {entry.key: entry for entry in await provider.handle_config_action(CONF_ACTION_TEST)}
    assert entries["test_error_label"].hidden is False
    assert "revoked" in (entries["test_error_label"].description or "")


async def test_client_uses_setup_flow_collected_values() -> None:
    """
    The client is built from setup_data, not from config.get_value.

    A freshly-created instance's config.values is {} (session.finish() persists
    collected fields into setup_data, not values), so reading CONF_URL/CONF_API_KEY
    through config.get_value resolves to DEFAULT_URL / "" no matter what the user
    entered during setup. This constructs a provider whose config.get_value is
    rigged to return deliberately-wrong values, so the test only passes if the
    client actually read through get_setup_value's setup_data path instead.
    """
    mass = MagicMock()
    mass.http_session = MagicMock()
    setup_data = {CONF_URL: "http://real-digarr:9999", CONF_API_KEY: "real-secret"}
    mass.config.get = MagicMock(return_value=setup_data)
    mass.config.decrypt_string = MagicMock(side_effect=lambda value: value)
    manifest = MagicMock()
    manifest.type = ProviderType.PLUGIN
    manifest.domain = "digarr"
    config = MagicMock()
    config.name = "digarr - Tom"
    config.instance_id = "digarr--abcd1234"
    config.values = {}
    config.get_value = MagicMock(
        side_effect=lambda key, default=None: {
            CONF_URL: "http://wrong-should-not-be-read:1",
            CONF_API_KEY: "wrong-should-not-be-read",
        }.get(key, default)
    )

    provider = DigarrProvider(mass, manifest, config, SUPPORTED_FEATURES)

    assert provider._client._base == "http://real-digarr:9999"
    assert provider._client._api_key == "real-secret"


async def test_ma_user_entry_is_not_required(provider: DigarrProvider) -> None:
    """
    CONF_MA_USER must not be required.

    On first load ma_user is in neither values nor setup_data, so a required entry
    here has an unresolvable (None) default_value. Config.validate() then raises,
    the load is treated as failed, and the just-created instance is rolled back --
    the provider could never be added at all.
    """
    entries = {entry.key: entry for entry in await provider.get_config_entries()}
    assert entries[CONF_MA_USER].required is False
