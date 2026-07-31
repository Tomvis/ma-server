"""Tests that the Lidarr provider registers its API command correctly."""

from __future__ import annotations

import inspect

from music_assistant_models.auth import Scope

from music_assistant.mass import MusicAssistant
from music_assistant.providers.lidarr.provider import LidarrProvider


def test_add_album_registration_matches_register_api_command() -> None:
    """
    The dynamic registration must use the kwargs register_api_command actually accepts.

    loaded_in_mass registers lidarr/add_album at runtime rather than via the
    @api_command decorator, so a signature change upstream (2.10 replaced
    required_role with required_scope) is invisible to import checks and the rest
    of the suite - it only surfaces as the provider failing to load on a real boot.
    """
    src = inspect.getsource(LidarrProvider.loaded_in_mass)
    assert "register_api_command" in src
    accepted = set(inspect.signature(MusicAssistant.register_api_command).parameters)
    # every keyword the call site passes must exist on the target signature
    for kwarg in ("required_scope",):
        assert kwarg in accepted, f"register_api_command no longer accepts {kwarg}"
        assert f"{kwarg}=" in src, f"lidarr registration should pass {kwarg}"
    assert "required_role" not in src, "required_role was removed in 2.10"


def test_add_album_scope_is_a_real_scope() -> None:
    """The scope the registration passes must be a member of the Scope enum."""
    src = inspect.getsource(LidarrProvider.loaded_in_mass)
    used = [name for name in dir(Scope) if not name.startswith("_") and f"Scope.{name}" in src]
    assert used, "no Scope member referenced in the registration"
    for name in used:
        assert isinstance(getattr(Scope, name), Scope)
