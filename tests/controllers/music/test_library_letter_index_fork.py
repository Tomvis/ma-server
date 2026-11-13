"""Fork: the album letter index honours the fork's listen-later and DR/AMG/TPS filters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from music_assistant_models.enums import SortDirection, SortField

from tests.controllers.music.test_library_letter_index import (
    GET_CURRENT_USER,
    LISTING_USER,
    _assert_matches_listing,
    seeded_mass,  # noqa: F401
)

if TYPE_CHECKING:
    from music_assistant.mass import MusicAssistant

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dr_buckets": ["untagged"]},
        {"amg_untagged": True, "tps_untagged": True},
        {"listen_later": False},
        {"dr_buckets": ["excellent"], "amg_untagged": True, "critical_reception_match": "any"},
    ],
)
async def test_album_letter_index_applies_fork_filters(
    seeded_mass: MusicAssistant,  # noqa: F811
    kwargs: dict[str, Any],
) -> None:
    """The offsets follow the album listing narrowed by the fork's filters."""
    index = await _assert_matches_listing(
        seeded_mass.music.albums,
        sort_field=SortField.SORT_NAME,
        sort_direction=SortDirection.ASC,
        **kwargs,
    )
    assert index


async def test_album_letter_index_fork_filter_excludes(
    seeded_mass: MusicAssistant,  # noqa: F811
) -> None:
    """A fork filter that matches nothing gives no index."""
    with patch(GET_CURRENT_USER, return_value=LISTING_USER):
        assert (
            await seeded_mass.music.albums.letter_index(
                sort_field=SortField.SORT_NAME, dr_buckets=["excellent"]
            )
            == []
        )
        assert (
            await seeded_mass.music.albums.letter_index(
                sort_field=SortField.SORT_NAME, listen_later=True
            )
            == []
        )
