"""Tests for the tag-prefix reader used by DR/critical-reception extraction."""

from music_assistant.providers.opensubsonic.sonic_provider import (
    CRITICAL_RECEPTION_PROBE_BYTES,
    _TagPrefixReader,
)

# The reader never inspects STREAMINFO contents, only block headers, so a
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


def _id3(payload_size: int) -> bytes:
    size = payload_size
    synchsafe = bytes((size >> shift) & 0x7F for shift in (21, 14, 7, 0))
    return b"ID3\x04\x00\x00" + synchsafe + b"\x11" * payload_size


def _read(data: bytes, chunk: int = 64 * 1024) -> tuple[bytes, int]:
    """Feed data in stream-sized chunks; return the prefix and the bytes consumed."""
    reader = _TagPrefixReader()
    consumed = 0
    for pos in range(0, len(data), chunk):
        piece = data[pos : pos + chunk]
        consumed += len(piece)
        if reader.feed(piece):
            break
    return reader.result(), consumed


def test_flac_trims_before_large_picture() -> None:
    """The DR tag survives a huge PICTURE block, and reading stops at the comment."""
    streaminfo = _block(0, _STREAMINFO, last=False)
    comment = _block(4, _vorbis_comment(ALBUM_DYNAMIC_RANGE="10"), last=False)
    picture = _block(6, b"\xab" * 3_000_000, last=True)

    out, consumed = _read(_flac(streaminfo, comment, picture))

    assert out.startswith(b"fLaC")
    assert b"ALBUM_DYNAMIC_RANGE=10" in out
    assert b"\xab" * 100 not in out
    assert out[_SECOND_BLOCK] & 0x80  # kept comment flagged as the last block
    assert consumed < 128 * 1024


def test_flac_reads_past_picture_before_comment() -> None:
    """Cover art ahead of the comment is read past, not kept (MUSIC-20)."""
    streaminfo = _block(0, _STREAMINFO, last=False)
    picture = _block(6, b"\xab" * 3_000_000, last=False)
    comment = _block(4, _vorbis_comment(ALBUM_DYNAMIC_RANGE="10"), last=False)
    padding = _block(1, b"\x00" * 300, last=True)

    out, _ = _read(_flac(streaminfo, picture, comment, padding) + b"\xff" * 100_000)

    assert out == _flac(streaminfo, _block(4, _vorbis_comment(ALBUM_DYNAMIC_RANGE="10"), last=True))


def test_flac_behind_id3_tag() -> None:
    """A FLAC carrying a leading ID3v2 tag still yields its comment block."""
    streaminfo = _block(0, _STREAMINFO, last=False)
    comment = _block(4, _vorbis_comment(ALBUM_DYNAMIC_RANGE="7"), last=False)
    picture = _block(6, b"\xab" * 1_000_000, last=True)

    out, _ = _read(_id3(4_000) + _flac(streaminfo, comment, picture))

    assert out.startswith(b"fLaC")
    assert b"ALBUM_DYNAMIC_RANGE=7" in out
    assert b"\xab" * 100 not in out


def test_flac_without_comment_ends_at_last_block() -> None:
    """Metadata with no VORBIS_COMMENT block ends the read at the last block."""
    streaminfo = _block(0, _STREAMINFO, last=False)
    padding = _block(1, b"\x00" * 50, last=True)

    out, _ = _read(_flac(streaminfo, padding) + b"\xff" * 10_000)

    assert out == _flac(_block(0, _STREAMINFO, last=True))


def test_mp3_keeps_whole_id3_tag_plus_audio() -> None:
    """An ID3v2 tag larger than the raw cap is kept whole, followed by some audio."""
    tag = _id3(737_000)
    audio = b"\xff\xfb" * 400_000

    out, _ = _read(tag + audio)

    assert out.startswith(tag)
    assert len(tag) < len(out) < len(tag) + 256 * 1024


def test_other_formats_fall_back_to_raw_prefix() -> None:
    """Without ID3 or FLAC markers the reader keeps a fixed-size raw prefix."""
    data = b"\x00\x00\x00\x20ftypM4A " + b"\x01" * 2_000_000

    out, _ = _read(data)

    assert out == data[:CRITICAL_RECEPTION_PROBE_BYTES]


def test_short_stream_returns_what_arrived() -> None:
    """A stream ending early still hands back everything it delivered."""
    reader = _TagPrefixReader()
    assert not reader.feed(b"\x00\x01")
    assert reader.result() == b"\x00\x01"
