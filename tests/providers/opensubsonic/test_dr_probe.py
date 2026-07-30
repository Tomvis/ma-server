"""Tests for the FLAC tag-prefix trimming used by DR/critical-reception extraction."""

from music_assistant.providers.opensubsonic.sonic_provider import _flac_tag_prefix

# _flac_tag_prefix never inspects STREAMINFO contents, only block headers, so a
# zero-filled 34-byte payload is a fine stand-in for a real STREAMINFO block.
_STREAMINFO = b"\x00" * 34
# Byte offset of the block that follows STREAMINFO: "fLaC" marker (4) + STREAMINFO
# header (4) + STREAMINFO payload (34).
_SECOND_BLOCK = 4 + 4 + len(_STREAMINFO)


def _block(block_type: int, payload: bytes, *, last: bool) -> bytes:
    header = (0x80 if last else 0x00) | (block_type & 0x7F)
    return bytes([header]) + len(payload).to_bytes(3, "big") + payload


def _vorbis_comment(**tags: str) -> bytes:
    vendor = b"test"
    out = len(vendor).to_bytes(4, "little") + vendor
    out += len(tags).to_bytes(4, "little")
    for key, value in tags.items():
        entry = f"{key}={value}".encode()
        out += len(entry).to_bytes(4, "little") + entry
    return out


def _flac(*blocks: bytes) -> bytes:
    return b"fLaC" + b"".join(blocks)


def test_trims_before_large_picture() -> None:
    """The DR tag survives even when the file is cut inside a huge PICTURE block."""
    streaminfo = _block(0, _STREAMINFO, last=False)
    comment = _block(4, _vorbis_comment(ALBUM_DYNAMIC_RANGE="10"), last=False)
    picture = _block(6, b"\xab" * 3_000_000, last=True)
    truncated = _flac(streaminfo, comment, picture)[: 512 * 1024]  # cut mid-picture

    out = _flac_tag_prefix(truncated)

    assert out is not None
    assert out.startswith(b"fLaC")
    assert b"ALBUM_DYNAMIC_RANGE=10" in out
    assert b"\xab" * 100 not in out  # the picture payload was dropped
    assert out[_SECOND_BLOCK] & 0x80  # kept comment flagged as the last block


def test_rejects_non_flac() -> None:
    """Non-FLAC and too-short inputs are left for the raw-prefix fallback."""
    assert _flac_tag_prefix(b"ID3\x04" + b"\x00" * 64) is None
    assert _flac_tag_prefix(b"") is None
    assert _flac_tag_prefix(b"fLa") is None


def test_incomplete_comment_block_returns_none() -> None:
    """A prefix that stops before the comment block completes can't be used yet."""
    streaminfo = _block(0, _STREAMINFO, last=False)
    comment = _block(4, _vorbis_comment(ALBUM_DYNAMIC_RANGE="10"), last=True)
    full = _flac(streaminfo, comment)

    assert _flac_tag_prefix(full[:-5]) is None  # comment payload truncated
    assert _flac_tag_prefix(full) == full  # complete (already the last block)


def test_picture_before_comment_when_truncated_returns_none() -> None:
    """Pathological block order: can't reach the comment, so fall back (None)."""
    streaminfo = _block(0, _STREAMINFO, last=False)
    picture = _block(6, b"\xab" * 3_000_000, last=False)
    comment = _block(4, _vorbis_comment(ALBUM_DYNAMIC_RANGE="10"), last=True)
    truncated = _flac(streaminfo, picture, comment)[: 512 * 1024]

    assert _flac_tag_prefix(truncated) is None


def test_no_comment_block_returns_none() -> None:
    """Metadata with no VORBIS_COMMENT block yields nothing to trim to."""
    streaminfo = _block(0, _STREAMINFO, last=True)
    assert _flac_tag_prefix(_flac(streaminfo)) is None
