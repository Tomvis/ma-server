"""
Lidarr Plugin Provider for Music Assistant (music-rater bridge).

Adds an "Add to Lidarr" action exposed through the WebSocket API. The action
hands the album off to music-rater (operator-run companion service), which
owns the Lidarr sync logic. MA just provides the album's MA URI; music-rater
resolves its own album record, sets lidarr_manual_add=True, and runs an
inline single-album sync against its configured Lidarr.
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
