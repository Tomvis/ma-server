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
    manifest = MagicMock()
    manifest.type = ProviderType.PLUGIN
    manifest.domain = "digarr"
    config = MagicMock()
    config.name = "digarr - Tom"
    config.instance_id = "digarr--abcd1234"
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
