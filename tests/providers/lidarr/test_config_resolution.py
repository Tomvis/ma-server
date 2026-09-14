"""
Regression tests for lidarr's options config resolution through the real config machinery.

Mirrors tests/providers/digarr/test_config_resolution.py: mocking `config.get_value`
restates whatever the implementation under test itself does with it, rather than
exercising the real persist-only-when-changed semantics that made this fragile. These
drive the actual load path (`seed_stored_config_values` -> construct ->
`rehydrate_provider_config`) and, for a save, the actual `Config.update`/`to_raw` --
the same objects `_update_provider_config` operates on -- against `mass_minimal`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

from music_assistant_models.config_entries import ProviderConfig
from music_assistant_models.enums import ProviderType
from music_assistant_models.provider import ProviderManifest

from music_assistant.constants import CONF_PROVIDERS, DEFAULT_PROVIDER_CONFIG_ENTRIES
from music_assistant.providers.lidarr.constants import CONF_URL, CONF_VERIFY_SSL
from music_assistant.providers.lidarr.provider import LidarrProvider

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant

# lidarr is single-instance, so instance_id == domain (matches the deployed instance).
INSTANCE_ID = "lidarr"

MANIFEST = ProviderManifest(
    type=ProviderType.PLUGIN,
    domain="lidarr",
    name="Lidarr",
    description="",
    codeowners=[],
)


async def _load_provider(
    mass: MusicAssistant,
    values: dict[str, object] | None = None,
    setup_data: dict[str, str] | None = None,
) -> LidarrProvider:
    """
    Load a lidarr provider the way the server does, and return the instance.

    :param mass: The MusicAssistant instance to load into.
    :param values: The provider's stored raw config `values`.
    :param setup_data: The provider's setup_data, plaintext (encrypted here, as it
        is at rest).
    """
    mass._http_session = MagicMock()  # avoid a real ClientSession
    encrypted_setup_data = {
        key: mass.config.encrypt_string(value) if isinstance(value, str) else value
        for key, value in (setup_data or {}).items()
    }
    raw = {
        "type": ProviderType.PLUGIN.value,
        "domain": "lidarr",
        "instance_id": INSTANCE_ID,
        "default_name": "Lidarr",
        "enabled": True,
        "values": values or {},
        "setup_data": encrypted_setup_data,
    }
    mass.config.set(f"{CONF_PROVIDERS}/{INSTANCE_ID}", raw)
    return await _reload_provider(mass)


async def _reload_provider(mass: MusicAssistant) -> LidarrProvider:
    """
    Reconstruct a fresh provider instance from whatever is currently stored for it.

    Mirrors what a real reload does: `rehydrate_provider_config` runs against the
    already-persisted raw config on disk, exactly as it would after a restart.

    :param mass: The MusicAssistant instance to load into.
    """
    raw = mass.config.get(f"{CONF_PROVIDERS}/{INSTANCE_ID}")
    # the load-time ordering matters: the config carries the server defaults only
    # until rehydrate_provider_config re-parses it against the provider's own entries
    config = cast("ProviderConfig", ProviderConfig.parse(DEFAULT_PROVIDER_CONFIG_ENTRIES, raw))
    mass.config.seed_stored_config_values(config)
    provider = LidarrProvider(mass, MANIFEST, config)
    await mass.config.rehydrate_provider_config(provider)
    return provider


async def _resave(
    mass: MusicAssistant, provider: LidarrProvider, posted: dict[str, object]
) -> set[str]:
    """
    Apply an options-page save exactly as `_update_provider_config` does, then call the hook.

    :param mass: The MusicAssistant instance the provider is loaded into.
    :param provider: The loaded provider instance being saved.
    :param posted: The submitted values, keyed like the options entries.
    """
    entries = await provider.get_config_entries()
    raw = mass.config.get(f"{CONF_PROVIDERS}/{INSTANCE_ID}")
    new_config = cast("ProviderConfig", ProviderConfig.parse(entries, raw))
    changed_keys = new_config.update(posted)
    mass.config.set(f"{CONF_PROVIDERS}/{INSTANCE_ID}", new_config.to_raw())
    await provider.update_config(new_config, changed_keys)
    return changed_keys  # type: ignore[no-any-return]


async def test_url_survives_two_consecutive_unrelated_saves(mass_minimal: MusicAssistant) -> None:
    """
    An explicit url must not be erased by later, unrelated saves.

    Regression test for mirroring get_config_entries()'s default_value to the field's
    own current value: Config.to_raw() persists an entry only when value !=
    default_value, so a default that tracks the live value converges to match it
    after one reload -- and the *next* save of any field then silently drops it from
    storage. This reproduces the exact deployed state (url in `values`, nothing in
    `setup_data`) and resaves an unrelated field (verify_ssl) twice in a row, since
    the erasure only manifests from the *second* save onward once the default has
    had one reload to converge.
    """
    provider = await _load_provider(mass_minimal, values={CONF_URL: "http://music-rater"})
    assert provider._client._base == "http://music-rater"

    changed_keys = await _resave(mass_minimal, provider, {CONF_VERIFY_SSL: False})
    assert f"values/{CONF_VERIFY_SSL}" in changed_keys
    stored = mass_minimal.config.get(f"{CONF_PROVIDERS}/{INSTANCE_ID}")
    assert stored["values"].get(CONF_URL) == "http://music-rater"

    reloaded = await _reload_provider(mass_minimal)
    assert reloaded._client._base == "http://music-rater"

    # a second consecutive save of a different value must not regress it either
    changed_keys = await _resave(mass_minimal, provider, {CONF_VERIFY_SSL: True})
    assert f"values/{CONF_VERIFY_SSL}" in changed_keys
    stored = mass_minimal.config.get(f"{CONF_PROVIDERS}/{INSTANCE_ID}")
    assert stored["values"].get(CONF_URL) == "http://music-rater"

    reloaded_again = await _reload_provider(mass_minimal)
    assert reloaded_again._client._base == "http://music-rater"


async def test_config_entry_default_never_shadows_the_stored_value_display(
    mass_minimal: MusicAssistant,
) -> None:
    """
    The options page's displayed value is the stored one; the default never mirrors it.

    `provider.config` (not a fresh `get_config_entries()` call, which never carries a
    `.value`) is what the framework actually re-parses stored config against and what
    the options page renders `.value` from. `.default_value` must stay None so it can
    never converge to equal the live value (see test above for why that matters).
    """
    provider = await _load_provider(mass_minimal, values={CONF_URL: "http://music-rater"})

    entry = provider.config.values[CONF_URL]

    assert entry.default_value is None
    assert entry.value == "http://music-rater"
