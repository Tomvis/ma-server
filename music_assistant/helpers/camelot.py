"""Helpers for the Camelot wheel notation of musical keys."""

from __future__ import annotations

import re


def camelot_code(key: str | None, mode: str | None) -> tuple[int, bool] | None:
    """Return (wheel number, is_major) or None for unknown input."""
    if not key or mode not in ("major", "minor"):
        return None
    key = _FLAT_TO_SHARP.get(key, key)
    table = _CAMELOT_MAJOR if mode == "major" else _CAMELOT_MINOR
    num = table.get(key)
    return (num, mode == "major") if num else None


def camelot_key(key: str | None, mode: str | None) -> str | None:
    """Return the Camelot code of a key (e.g. "8A" for A minor, "8B" for C major) or None."""
    code = camelot_code(key, mode)
    if code is None:
        return None
    num, is_major = code
    return f"{num}{'B' if is_major else 'A'}"


def parse_key_tag(value: str | None) -> tuple[str, str] | None:
    """
    Return (pitch class, mode) of a file's key tag (e.g. "F#m", "Eb", "A minor"), or None.

    :param value: The tag value; a bare pitch class means major.
    """
    if not value or not (match := _KEY_TAG.fullmatch(value.strip())):
        return None
    note, accidental, mode = match.groups()
    accidental = {"♯": "#", "♭": "b"}.get(accidental, accidental.lower())
    is_minor = mode is not None and mode.lower() in ("m", "min", "minor")
    return f"{note.upper()}{accidental}", "minor" if is_minor else "major"


_KEY_TAG = re.compile(r"([A-Ga-g])([#b♯♭]?)\s*(m|min|minor|maj|major)?", re.IGNORECASE)
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
# Camelot wheel numbers; relative major/minor share a number (Am=8A, C=8B)
_CAMELOT_MAJOR = {
    "C": 8,
    "G": 9,
    "D": 10,
    "A": 11,
    "E": 12,
    "B": 1,
    "F#": 2,
    "C#": 3,
    "G#": 4,
    "D#": 5,
    "A#": 6,
    "F": 7,
}
_CAMELOT_MINOR = {
    "A": 8,
    "E": 9,
    "B": 10,
    "F#": 11,
    "C#": 12,
    "G#": 1,
    "D#": 2,
    "A#": 3,
    "F": 4,
    "C": 5,
    "G": 6,
    "D": 7,
}
