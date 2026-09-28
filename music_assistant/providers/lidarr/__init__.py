"""
Lidarr Plugin Provider for Music Assistant.

Adds an "Add to Lidarr" action exposed through the WebSocket API: the album's artist
is added to Lidarr if missing (into the acting user's root folder), then only that
album is monitored and searched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from music_assistant.providers.lidarr.provider import LidarrProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return LidarrProvider(mass, manifest, config)
