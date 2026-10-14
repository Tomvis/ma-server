"""Update rules for album critical_reception (review) metadata."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from music_assistant_models.media_items.metadata import CriticalReception


def critical_reception_for_update(
    new: CriticalReception | None, stored: CriticalReception | None
) -> CriticalReception | None:
    """
    Return the critical_reception to store when a provider reports `new`.

    A reported CR replaces the stored one outright, dropped accolades, sources and
    review text included: it is read from the files' review tags, which music-rater
    rewrites when it corrects a match (MUSIC-80). No CR, or one without sources or
    amg_dr, is not a report and keeps `stored`. The stored amg_dr survives a CR that
    does not report one, since that tag has left the files (MUSIC-78).

    :param new: The critical_reception the provider reported, if any.
    :param stored: The critical_reception currently stored for the album.
    """
    if new is None or (not new.sources and new.amg_dr is None):
        return stored
    if new.amg_dr is None and stored is not None and stored.amg_dr is not None:
        return replace(new, amg_dr=stored.amg_dr)
    return new
