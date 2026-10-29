"""Tests for the Camelot wheel helpers."""

from __future__ import annotations

import pytest

from music_assistant.helpers.camelot import camelot_key, parse_key_tag


@pytest.mark.parametrize(
    ("key", "mode", "expected"),
    [
        ("A", "minor", "8A"),
        ("C", "major", "8B"),
        ("Eb", "major", "5B"),
        ("F#", "minor", "11A"),
        ("C", "dorian", None),
        ("C", None, None),
        (None, "major", None),
    ],
)
def test_camelot_key(key: str | None, mode: str | None, expected: str | None) -> None:
    """Keys map to their Camelot code, unknown input yields None."""
    assert camelot_key(key, mode) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Em", ("E", "minor")),
        ("D#", ("D#", "major")),
        ("F#m", ("F#", "minor")),
        ("Bbm", ("Bb", "minor")),
        ("A minor", ("A", "minor")),
        ("C major", ("C", "major")),
        ("c#min", ("C#", "minor")),
        ("E♭", ("Eb", "major")),
        (" Am ", ("A", "minor")),
        ("A major; G major", None),
        ("Zer", None),
        ("H", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_key_tag(value: str | None, expected: tuple[str, str] | None) -> None:
    """Key tags in the usual spellings parse to (pitch class, mode); anything else to None."""
    assert parse_key_tag(value) == expected
