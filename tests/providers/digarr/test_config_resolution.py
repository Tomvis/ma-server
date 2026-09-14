"""
Regression tests for digarr's setup/options config resolution through the real config machinery.

Mirrors tests/providers/filesystem/test_content_type_resolution.py: mocking
`config.get_value` (as test_config.py's fixture does for unrelated assertions) restates
whatever the implementation under test itself does with it, rather than exercising the
real decrypt-on-read and persist-only-when-changed semantics that made this fragile.
These drive the actual load path (`seed_stored_config_values` -> construct ->
`rehydrate_provider_config`) and, for a save, the actual `Config.update`/`to_raw` -- the
same objects `_update_provider_config` operates on -- against `mass_minimal`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

from music_assistant_models.config_entries import ProviderConfig
from music_assistant_models.constants import SECURE_STRING_SUBSTITUTE
from music_assistant_models.enums import ProviderType
from music_assistant_models.provider import ProviderManifest

import music_assistant.providers.digarr as digarr_module
from music_assistant.constants import CONF_PROVIDERS, DEFAULT_PROVIDER_CONFIG_ENTRIES
from music_assistant.providers.digarr import SUPPORTED_FEATURES, DigarrProvider
from music_assistant.providers.digarr.constants import (
    CONF_API_KEY,
    CONF_MA_USER,
    CONF_ROW_SIZE,
    CONF_URL,
)

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant

INSTANCE_ID = "digarr--test"

MANIFEST = ProviderManifest(
    type=ProviderType.PLUGIN,
    domain="digarr",
    name="digarr",
    description="",
    codeowners=[],
)


async def _load_provider(
    mass: MusicAssistant,
    setup_data: dict[str, str],
    values: dict[str, str] | None = None,
) -> DigarrProvider:
    """
    Load a digarr provider the way the server does, and return the instance.

    :param mass: The MusicAssistant instance to load into.
    :param setup_data: The provider's setup_data, plaintext (encrypted here, as it is
        at rest -- setup_data encrypts every string regardless of field).
    :param values: The provider's stored raw config `values`, already in their at-rest
        form (i.e. pass real ciphertext for a SECURE_STRING key).
    """
    mass._http_session = MagicMock()  # avoid a real ClientSession
    encrypted_setup_data = {
        key: mass.config.encrypt_string(value) if isinstance(value, str) else value
        for key, value in setup_data.items()
    }
    raw = {
        "type": ProviderType.PLUGIN.value,
        "domain": "digarr",
        "instance_id": INSTANCE_ID,
        "default_name": "digarr",
        "enabled": True,
        "values": values or {},
        "setup_data": encrypted_setup_data,
    }
    mass.config.set(f"{CONF_PROVIDERS}/{INSTANCE_ID}", raw)
    return await _reload_provider(mass)


async def _reload_provider(mass: MusicAssistant) -> DigarrProvider:
    """
    Reconstruct a fresh provider instance from whatever is currently stored for it.

    Mirrors what a real reload does: `rehydrate_provider_config` runs against the
    already-encrypted raw config on disk, exactly as it would after a restart.

    :param mass: The MusicAssistant instance to load into.
    """
    raw = mass.config.get(f"{CONF_PROVIDERS}/{INSTANCE_ID}")
    # the load-time ordering matters: the config carries the server defaults only
    # until rehydrate_provider_config re-parses it against the provider's own entries
    config = cast("ProviderConfig", ProviderConfig.parse(DEFAULT_PROVIDER_CONFIG_ENTRIES, raw))
    mass.config.seed_stored_config_values(config)
    provider = DigarrProvider(mass, MANIFEST, config, SUPPORTED_FEATURES)
    await mass.config.rehydrate_provider_config(provider)
    return provider


async def _resave(
    mass: MusicAssistant, provider: DigarrProvider, posted: dict[str, object]
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


async def test_client_never_reads_a_secure_string_through_options_values(
    mass_minimal: MusicAssistant,
) -> None:
    """
    The client must never resolve CONF_API_KEY through the options `values` blob.

    CONF_API_KEY is a SECURE_STRING, but the seed_stored_config_values passthrough
    entry construction-time reads go through is a plain STRING -- Config.get_value
    only decrypts when the entry it reads *through* is typed SECURE_STRING, so reading
    it there would hand back raw ciphertext. Real ciphertext for a decoy key is seeded
    into `values` here; the client must still come from setup_data alone.
    """
    stray_ciphertext = mass_minimal.config.encrypt_string("SHOULD-NOT-BE-USED")
    provider = await _load_provider(
        mass_minimal,
        setup_data={CONF_URL: "http://digarr:3000", CONF_API_KEY: "real-key"},
        values={CONF_API_KEY: stray_ciphertext},
    )

    assert provider._client._api_key == "real-key"


async def test_api_key_never_appears_in_a_serialized_config_entry(
    mass_minimal: MusicAssistant,
) -> None:
    """
    CONF_API_KEY's serialized entry must never carry the decrypted key.

    Config.__post_serialize__ masks a SECURE_STRING's `value` to
    SECURE_STRING_SUBSTITUTE before an entry ever reaches an API response, but does
    NOT mask `default_value` -- so giving this entry any default_value derived from
    the decrypted setup value (as an earlier version of get_config_entries did) would
    serve the plaintext key to any caller with CONFIG_PROVIDERS_READ, i.e. any
    household member, not just the instance's owner. The entry must also stay
    loadable without one: a required entry with neither a value nor a default fails
    Config.validate() and would stop the instance loading.
    """
    provider = await _load_provider(
        mass_minimal,
        setup_data={CONF_URL: "http://digarr:3000", CONF_API_KEY: "super-secret-key"},
    )

    provider.config.validate()  # must not raise

    entries = {entry.key: entry for entry in await provider.get_config_entries()}
    assert entries[CONF_API_KEY].default_value is None
    assert entries[CONF_API_KEY].required is False

    payload = json.dumps(provider.config.to_dict())
    assert "super-secret-key" not in payload


async def test_api_key_never_appears_after_an_explicit_options_edit(
    mass_minimal: MusicAssistant,
) -> None:
    """The same guarantee holds once api_key has also been explicitly saved to `values`."""
    provider = await _load_provider(
        mass_minimal,
        setup_data={CONF_URL: "http://digarr:3000", CONF_API_KEY: "old-secret-key"},
        values={CONF_API_KEY: mass_minimal.config.encrypt_string("new-secret-key")},
    )

    provider.config.validate()  # must not raise

    payload = json.dumps(provider.config.to_dict())
    assert "old-secret-key" not in payload
    assert "new-secret-key" not in payload


async def test_api_key_rotation_reaches_the_client_after_a_reload(
    mass_minimal: MusicAssistant,
) -> None:
    """
    An options-page api_key rotation must reach DigarrClient in plaintext, after a reload.

    update_config() is what mirrors a real rotation into setup_data -- exercised here
    exactly as the config controller calls it (parse -> update -> persist -> hook).
    """
    provider = await _load_provider(
        mass_minimal, setup_data={CONF_URL: "http://digarr:3000", CONF_API_KEY: "OLDKEY"}
    )
    changed_keys = await _resave(mass_minimal, provider, {CONF_API_KEY: "NEWKEY"})
    assert f"values/{CONF_API_KEY}" in changed_keys

    reloaded = await _reload_provider(mass_minimal)

    assert reloaded._client._api_key == "NEWKEY"


async def test_unrelated_save_does_not_corrupt_the_key_with_the_substitute_placeholder(
    mass_minimal: MusicAssistant,
) -> None:
    """
    A caller resubmitting the secret placeholder must not have it treated as a rotation.

    The shipped frontend skips resubmitting an untouched SECURE_STRING entirely
    (EditConfig.vue/SetupFlowDialog.vue), so a normal options-page save never
    exercises this path. A non-frontend caller is not guaranteed to: the MCP config
    tool, for one, masks the current value to SECURE_STRING_SUBSTITUTE before
    deciding what to forward, and could echo it straight back. That placeholder
    never equals the stored ciphertext, so Config.update() would flag it "changed"
    regardless -- this guard is defence against such a caller clobbering the real
    key with the literal placeholder text, not against the frontend.
    """
    provider = await _load_provider(
        mass_minimal, setup_data={CONF_URL: "http://digarr:3000", CONF_API_KEY: "OLDKEY"}
    )
    # a non-frontend caller resubmits the masked placeholder alongside the field it
    # actually means to change
    posted = {CONF_API_KEY: SECURE_STRING_SUBSTITUTE, CONF_ROW_SIZE: 5}
    changed_keys = await _resave(mass_minimal, provider, posted)
    # confirms the spurious-change premise the guard exists for
    assert f"values/{CONF_API_KEY}" in changed_keys

    reloaded = await _reload_provider(mass_minimal)

    assert reloaded._client._api_key == "OLDKEY"


async def test_url_edit_survives_an_unrelated_save(mass_minimal: MusicAssistant) -> None:
    """
    An explicit url edit must not be erased by a later save of an unrelated field.

    Regression test for mirroring get_config_entries()'s default_value to the field's
    own current value: Config.to_raw() persists an entry only when value !=
    default_value, so a default that tracks the live value converges to match it after
    one reload -- and the *next* save of any field then silently drops it from storage.
    """
    provider = await _load_provider(
        mass_minimal,
        setup_data={CONF_URL: "http://stale-no-port", CONF_API_KEY: "k"},
        values={CONF_URL: "http://fixed-with-port:4533"},
    )
    assert provider._client._base == "http://fixed-with-port:4533"

    # the field actually being changed; url is not resubmitted at all, matching the
    # production symptom exactly ("saving only row_size erased url and ma_user")
    changed_keys = await _resave(mass_minimal, provider, {CONF_ROW_SIZE: 5})
    assert f"values/{CONF_ROW_SIZE}" in changed_keys
    stored = mass_minimal.config.get(f"{CONF_PROVIDERS}/{INSTANCE_ID}")
    assert stored["values"].get(CONF_URL) == "http://fixed-with-port:4533"

    reloaded = await _reload_provider(mass_minimal)

    assert reloaded._client._base == "http://fixed-with-port:4533"


async def test_ma_user_edit_survives_an_unrelated_save(mass_minimal: MusicAssistant) -> None:
    """
    An explicit ma_user edit must not be erased by a later save of an unrelated field.

    Same hazard as the url case, and the one actually live in production: both
    deployed instances already have ma_user set in `values` with nothing in
    setup_data, which -- with a default_value that mirrored the live value -- meant
    saving *anything* on the options page erased it and the Discover row went
    invisible with no error explaining why.
    """
    provider = await _load_provider(
        mass_minimal,
        setup_data={CONF_URL: "http://digarr:3000", CONF_API_KEY: "k"},
        values={CONF_MA_USER: "tom"},
    )
    assert provider._ma_user == "tom"

    await _resave(mass_minimal, provider, {CONF_ROW_SIZE: 5})
    stored = mass_minimal.config.get(f"{CONF_PROVIDERS}/{INSTANCE_ID}")
    assert stored["values"].get(CONF_MA_USER) == "tom"

    reloaded = await _reload_provider(mass_minimal)

    assert reloaded._ma_user == "tom"


async def test_ma_user_is_read_from_setup_data_when_options_are_unset(
    mass_minimal: MusicAssistant,
) -> None:
    """
    A freshly set-up instance is bound to a user without ever visiting the options page.

    Before this, CONF_MA_USER was options-only, so a brand-new instance always loaded
    with ma_user=None and its Discover row was invisible to every viewer with no error
    explaining why. The setup flow now collects it into setup_data exactly like url and
    api_key, so it must be read back the same way.
    """
    provider = await _load_provider(
        mass_minimal,
        setup_data={CONF_URL: "http://digarr:3000", CONF_API_KEY: "k", CONF_MA_USER: "tom"},
    )

    assert provider._ma_user == "tom"


async def test_config_entry_defaults_never_shadow_the_setup_value_display(
    mass_minimal: MusicAssistant,
) -> None:
    """
    The options page's displayed default is the setup value, unaffected by any edit.

    `provider.config` (not a fresh `get_config_entries()` call, which never carries a
    `.value`) is what the framework actually re-parses stored config against and what
    the options page renders `.value` from -- `.default_value` is only ever the
    fallback shown when `.value` is unset.
    """
    provider = await _load_provider(
        mass_minimal,
        setup_data={CONF_URL: "http://stale-no-port", CONF_API_KEY: "k"},
        values={CONF_URL: "http://fixed-with-port:4533"},
    )

    entry = provider.config.values[CONF_URL]

    assert entry.default_value == "http://stale-no-port"
    assert entry.value == "http://fixed-with-port:4533"


async def test_loaded_in_mass_does_not_refetch_ma_usernames(mass_minimal: MusicAssistant) -> None:
    """
    loaded_in_mass()'s ma_user check must reuse get_config_entries()'s user lookup.

    Both used to call ma_usernames() (and thus list_users()) independently for the
    same load -- get_config_entries() (already run once by rehydrate during
    `_load_provider` above) must have cached its result for loaded_in_mass() to reuse
    instead of fetching its own copy.
    """
    provider = await _load_provider(
        mass_minimal,
        setup_data={CONF_URL: "http://digarr:3000", CONF_API_KEY: "k", CONF_MA_USER: "tom"},
    )
    with patch.object(
        digarr_module,
        "ma_usernames",
        new=AsyncMock(side_effect=AssertionError("ma_usernames was re-fetched")),
    ):
        await provider.loaded_in_mass()


async def test_warns_exactly_once_when_a_pre_fix_values_api_key_is_being_ignored(
    mass_minimal: MusicAssistant,
) -> None:
    """
    A pre-fix rotation left in `values` must be flagged, not silently dropped.

    Before today's fix, an api_key rotation on the options page landed in `values`
    only; the client is now built exclusively from setup_data, so an instance left in
    that state silently keeps using the old key with nothing telling the user their
    rotation never took effect -- the options page only ever shows the masked
    substitute, so there is no way to spot this from the UI either.

    Asserts the exact call count (not just "some warning matched"): there is no
    de-dup mechanism behind this warning -- it is naturally single-fire because
    loaded_in_mass() itself only runs once per load and the check has no loop -- and
    an assertion that only checks a match among the calls would not catch a second,
    accidental warning appearing alongside it.
    """
    provider = await _load_provider(
        mass_minimal,
        setup_data={CONF_URL: "http://digarr:3000", CONF_API_KEY: "OLDKEY"},
        values={CONF_API_KEY: mass_minimal.config.encrypt_string("NEVER-APPLIED")},
    )
    assert provider._client._api_key == "OLDKEY"
    provider._client.whoami = AsyncMock(  # type: ignore[method-assign]
        return_value=("test-user", False)
    )
    provider.logger = MagicMock()

    await provider.loaded_in_mass()

    assert provider.logger.warning.call_count == 1
    message = provider.logger.warning.call_args.args[0] % provider.logger.warning.call_args.args[1:]
    assert "re-enter the api_key" in message.lower()


async def test_does_not_warn_when_values_api_key_matches_the_active_one(
    mass_minimal: MusicAssistant,
) -> None:
    """No spurious warning once `values` and setup_data legitimately hold the same key."""
    provider = await _load_provider(
        mass_minimal,
        setup_data={CONF_URL: "http://digarr:3000", CONF_API_KEY: "SAMEKEY"},
        values={CONF_API_KEY: mass_minimal.config.encrypt_string("SAMEKEY")},
    )
    assert provider._client._api_key == "SAMEKEY"
    provider._client.whoami = AsyncMock(  # type: ignore[method-assign]
        return_value=("test-user", False)
    )
    provider.logger = MagicMock()

    await provider.loaded_in_mass()

    provider.logger.warning.assert_not_called()


async def test_does_not_warn_when_values_has_no_api_key_at_all(
    mass_minimal: MusicAssistant,
) -> None:
    """The ordinary case (api_key only ever in setup_data) must never warn."""
    provider = await _load_provider(
        mass_minimal, setup_data={CONF_URL: "http://digarr:3000", CONF_API_KEY: "k"}
    )
    provider._client.whoami = AsyncMock(  # type: ignore[method-assign]
        return_value=("test-user", False)
    )
    provider.logger = MagicMock()

    await provider.loaded_in_mass()

    provider.logger.warning.assert_not_called()
