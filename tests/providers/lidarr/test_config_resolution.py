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

import json
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.config_entries import ProviderConfig
from music_assistant_models.constants import SECURE_STRING_SUBSTITUTE
from music_assistant_models.enums import ProviderType
from music_assistant_models.provider import ProviderManifest

from music_assistant.constants import CONF_PROVIDERS, DEFAULT_PROVIDER_CONFIG_ENTRIES
from music_assistant.providers.lidarr.client import LidarrError
from music_assistant.providers.lidarr.constants import (
    CONF_API_KEY,
    CONF_ROOT_FOLDER_PREFIX,
    CONF_URL,
    CONF_VERIFY_SSL,
)
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
    default_value. Against the pre-fix code this erases url on the very *first*
    unrelated save, not a later one -- get_config_entries() computes that default
    from the provider's already-loaded, already-rehydrated self.config (which
    already holds the live url the moment the instance exists), not from some
    pre-rehydrate snapshot that would need a reload to catch up. Resaves an
    unrelated field (verify_ssl) twice in a row here only to also confirm the
    already-broken state doesn't regress any further on a second save.
    """
    provider = await _load_provider(mass_minimal, values={CONF_URL: "http://lidarr"})
    assert provider._client._base == "http://lidarr"

    changed_keys = await _resave(mass_minimal, provider, {CONF_VERIFY_SSL: False})
    assert f"values/{CONF_VERIFY_SSL}" in changed_keys
    stored = mass_minimal.config.get(f"{CONF_PROVIDERS}/{INSTANCE_ID}")
    assert stored["values"].get(CONF_URL) == "http://lidarr"

    reloaded = await _reload_provider(mass_minimal)
    assert reloaded._client._base == "http://lidarr"

    # a second consecutive save of a different value must not regress it either
    changed_keys = await _resave(mass_minimal, provider, {CONF_VERIFY_SSL: True})
    assert f"values/{CONF_VERIFY_SSL}" in changed_keys
    stored = mass_minimal.config.get(f"{CONF_PROVIDERS}/{INSTANCE_ID}")
    assert stored["values"].get(CONF_URL) == "http://lidarr"

    reloaded_again = await _reload_provider(mass_minimal)
    assert reloaded_again._client._base == "http://lidarr"


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
    provider = await _load_provider(mass_minimal, values={CONF_URL: "http://lidarr"})

    entry = provider.config.values[CONF_URL]

    assert entry.default_value is None
    assert entry.value == "http://lidarr"


async def test_fresh_instance_reads_url_from_setup_data(mass_minimal: MusicAssistant) -> None:
    """
    A newly-added instance's url (collected by the setup flow into setup_data) must work.

    setup_flow.py collects CONF_URL into setup_data, not `values` -- a freshly created
    instance's `values` is {} (controllers/config/flows.py's _finish_provider_setup).
    Reading url only through plain config.get_value (as an earlier version of
    __init__ did) misses setup_data entirely, so any instance added since the setup
    flow started collecting it there would build MusicRaterClient(url=None, ...) and
    crash immediately on `url.rstrip("/")` -- worse than the erasure bug this whole
    file otherwise guards against, since it means the provider can never be added at
    all rather than merely losing its url later.
    """
    provider = await _load_provider(mass_minimal, values={}, setup_data={CONF_URL: "http://lidarr"})

    assert provider._client._base == "http://lidarr"


async def test_explicit_options_edit_overrides_a_setup_value(mass_minimal: MusicAssistant) -> None:
    """An options-page url edit must win over whatever the setup flow originally collected."""
    provider = await _load_provider(
        mass_minimal,
        values={CONF_URL: "http://fixed-with-port:4533"},
        setup_data={CONF_URL: "http://stale-no-port"},
    )

    assert provider._client._base == "http://fixed-with-port:4533"


async def test_url_in_neither_store_fails_to_construct(mass_minimal: MusicAssistant) -> None:
    """
    The fourth url-location cell (present in neither `values` nor `setup_data`) must fail.

    Pins current (unchanged by any of today's fixes) behaviour rather than the
    cleaner outcome it might look like at a glance: `_config_or_setup_value` returns
    None here (nothing to prefer, nothing to fall back to), and MusicRaterClient's
    constructor dereferences that url immediately (`url.rstrip("/")`) before
    Config.validate() ever gets a chance to run and report "url is required"
    instead. The real load path (mass.py's `_provider_load_step`) re-raises this
    unwrapped rather than turning it into a SetupFailedError, since a non-empty
    `str(err)` skips that wrapping -- so this is a raw AttributeError, not a clean
    validation failure. Only pinned here, not fixed: no url-location cell this file
    covers is worse off than before today's fixes, and hardening this one further
    (e.g. giving MusicRaterClient a tolerant default) is a separate change.
    """
    with pytest.raises(AttributeError):
        await _load_provider(mass_minimal, values={}, setup_data={})


async def test_loaded_in_mass_warns_when_reconfigure_was_silently_overridden(
    mass_minimal: MusicAssistant,
) -> None:
    """
    A Reconfigure whose new url lost to a stale `values` entry must be flagged.

    `_finish_provider_reconfigure` (controllers/config/flows.py) only ever writes
    setup_data, never `values` -- so on the deployed shape (url already in `values`),
    submitting a new url via Reconfigure is silently overridden by
    `_config_or_setup_value`'s values-wins precedence: the reload reports success and
    nothing tells the admin their new url was ignored. This warns instead.
    """
    provider = await _load_provider(
        mass_minimal,
        values={CONF_URL: "http://lidarr"},
        setup_data={CONF_URL: "http://just-reconfigured"},
    )
    provider._client.system_status = AsyncMock()  # type: ignore[method-assign]
    provider.logger = MagicMock()

    await provider.loaded_in_mass()

    assert provider.logger.warning.call_count == 1
    message = provider.logger.warning.call_args.args[0] % provider.logger.warning.call_args.args[1:]
    assert "reconfigure" in message.lower()


async def test_loaded_in_mass_does_not_warn_without_a_conflicting_reconfigure(
    mass_minimal: MusicAssistant,
) -> None:
    """No spurious warning for the ordinary deployed shape (nothing in setup_data at all)."""
    provider = await _load_provider(mass_minimal, values={CONF_URL: "http://lidarr"})
    provider._client.system_status = AsyncMock()  # type: ignore[method-assign]
    provider.logger = MagicMock()

    await provider.loaded_in_mass()

    provider.logger.warning.assert_not_called()


async def test_api_key_from_setup_data_reaches_the_client(mass_minimal: MusicAssistant) -> None:
    """A setup-collected api_key is sent as Lidarr's X-Api-Key header."""
    provider = await _load_provider(
        mass_minimal, setup_data={CONF_URL: "http://lidarr", CONF_API_KEY: "k3y"}
    )

    assert provider._client._headers["X-Api-Key"] == "k3y"


async def test_client_never_reads_the_key_through_options_values(
    mass_minimal: MusicAssistant,
) -> None:
    """
    Ciphertext in `values` must never become the key.

    A key saved to `values` without reaching setup_data is encrypted there, and the
    pre-rehydrate passthrough entry is a plain STRING, so reading it would send ciphertext.
    """
    stray_ciphertext = mass_minimal.config.encrypt_string("should-not-be-used")
    provider = await _load_provider(
        mass_minimal, values={CONF_URL: "http://lidarr", CONF_API_KEY: stray_ciphertext}
    )

    assert provider._client._headers["X-Api-Key"] == ""


async def test_api_key_never_appears_in_a_serialized_config_entry(
    mass_minimal: MusicAssistant,
) -> None:
    """__post_serialize__ masks a SECURE_STRING value but not its default: never set one."""
    provider = await _load_provider(
        mass_minimal, setup_data={CONF_URL: "http://lidarr", CONF_API_KEY: "secret-key"}
    )

    provider.config.validate()  # must not raise
    entries = {entry.key: entry for entry in await provider.get_config_entries()}
    assert entries[CONF_API_KEY].default_value is None
    assert entries[CONF_API_KEY].required is False
    assert "secret-key" not in json.dumps(provider.config.to_dict())


async def test_api_key_options_edit_reaches_the_client_after_a_reload(
    mass_minimal: MusicAssistant,
) -> None:
    """The deployed instance (url in values, no key) gets its key via the options page."""
    provider = await _load_provider(mass_minimal, values={CONF_URL: "http://lidarr"})
    changed_keys = await _resave(mass_minimal, provider, {CONF_API_KEY: "new-key"})
    assert f"values/{CONF_API_KEY}" in changed_keys

    reloaded = await _reload_provider(mass_minimal)

    assert reloaded._client._headers["X-Api-Key"] == "new-key"
    assert reloaded._client._base == "http://lidarr"


async def test_resubmitted_placeholder_is_not_treated_as_a_rotation(
    mass_minimal: MusicAssistant,
) -> None:
    """A non-frontend caller echoing the masked placeholder must not clobber the key."""
    provider = await _load_provider(
        mass_minimal, setup_data={CONF_URL: "http://lidarr", CONF_API_KEY: "old-key"}
    )
    posted = {CONF_API_KEY: SECURE_STRING_SUBSTITUTE, CONF_VERIFY_SSL: False}
    changed_keys = await _resave(mass_minimal, provider, posted)
    assert f"values/{CONF_API_KEY}" in changed_keys

    reloaded = await _reload_provider(mass_minimal)

    assert reloaded._client._headers["X-Api-Key"] == "old-key"


def _users(*names: str) -> AsyncMock:
    return AsyncMock(return_value=[MagicMock(username=name) for name in names])


async def test_one_root_folder_entry_per_user_offering_lidarrs_roots(
    mass_minimal: MusicAssistant,
) -> None:
    """Each MA user gets a root-folder picker filled from Lidarr's live root folders."""
    provider = await _load_provider(mass_minimal, values={CONF_URL: "http://lidarr"})
    provider._usernames = AsyncMock(return_value=["lera", "tom"])  # type: ignore[method-assign]
    provider._client.list_root_folders = AsyncMock(  # type: ignore[method-assign]
        return_value=[{"path": "/music/tom"}, {"path": "/music/lera"}]
    )

    entries = {entry.key: entry for entry in await provider.get_config_entries()}

    for user in ("lera", "tom"):
        entry = entries[f"{CONF_ROOT_FOLDER_PREFIX}{user}"]
        assert entry.required is False
        assert entry.translation_params == [user]
        assert [o.value for o in entry.options] == ["/music/tom", "/music/lera"]


async def test_root_folder_mapping_survives_an_unrelated_save(mass_minimal: MusicAssistant) -> None:
    """A per-user mapping persists across a save of another field and a reload."""
    key = f"{CONF_ROOT_FOLDER_PREFIX}lera"
    with patch.object(LidarrProvider, "_usernames", AsyncMock(return_value=["lera", "tom"])):
        provider = await _load_provider(mass_minimal, values={CONF_URL: "http://lidarr"})
        await _resave(mass_minimal, provider, {key: "/music/lera"})
        await _resave(mass_minimal, provider, {CONF_VERIFY_SSL: False})

        reloaded = await _reload_provider(mass_minimal)

    assert reloaded.config.get_value(key) == "/music/lera"


async def test_add_album_uses_the_acting_users_root_folder(mass_minimal: MusicAssistant) -> None:
    """The root folder comes from the signed-in user's mapping."""
    provider = await _load_provider(
        mass_minimal,
        values={CONF_URL: "http://lidarr", f"{CONF_ROOT_FOLDER_PREFIX}lera": "/music/lera"},
    )
    with patch(
        "music_assistant.providers.lidarr.provider.get_current_user",
        return_value=MagicMock(username="lera"),
    ):
        assert provider._root_folder_for_current_user() == "/music/lera"


async def test_add_album_refuses_a_user_without_a_mapping(mass_minimal: MusicAssistant) -> None:
    """No guessed folder: an unmapped user gets an error naming the fix."""
    provider = await _load_provider(mass_minimal, values={CONF_URL: "http://lidarr"})
    with (
        patch(
            "music_assistant.providers.lidarr.provider.get_current_user",
            return_value=MagicMock(username="guest"),
        ),
        pytest.raises(LidarrError, match="options page"),
    ):
        await provider.add_album("tidal://album/1")
