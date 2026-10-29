"""Shared fixtures for Open Subsonic provider tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from music_assistant.providers.opensubsonic import SUPPORTED_FEATURES
from music_assistant.providers.opensubsonic.sonic_provider import OpenSonicProvider


@pytest.fixture
def provider() -> OpenSonicProvider:
    """Create a real OpenSonicProvider with mocked dependencies."""
    mass = Mock()
    manifest = Mock()
    manifest.domain = "opensubsonic"
    config = Mock()
    config.instance_id = "opensubsonic--test123"
    config.name = "Open Subsonic Test"
    config.enabled = True
    config.get_value.side_effect = lambda key, default=None: {
        "log_level": "GLOBAL",
    }.get(key, default)
    mass.cache.get = AsyncMock(return_value=None)
    mass.cache.set = AsyncMock()
    mass.cache.delete = AsyncMock()
    mass.streams.audio_analysis.set_track_tags = AsyncMock()
    return OpenSonicProvider(mass, manifest, config, SUPPORTED_FEATURES)
