"""
Braille Codec — encode/decode arbitrary bytes as Braille Unicode.

The 256 Unicode Braille characters (U+2800 to U+28FF) map 1:1 to byte values
0x00–0xFF. Each character's Unicode codepoint offset from U+2800 IS the byte
value. This gives us a natural, lossless, bijective encoding.

Properties:
  - Bijective:     every byte sequence has exactly one Braille representation
  - Lossless:      decode(encode(x)) == x for all x
  - Fixed density:  1 byte = 1 Braille character = 3 UTF-8 bytes

Token efficiency:
  Most LLM tokenizers treat Braille characters as single tokens or small
  multi-byte sequences, making this encoding competitive with or better than
  hex/base64 for transmitting binary data through language models.

Visual examples:
  0x00 → ⠀ (blank)    0x01 → ⠁    0x41 ('A') → ⡁    0xFF → ⣿ (all dots)
"""

# The base codepoint for Braille patterns
_BRAILLE_BASE = 0x2800

# Pre-compute lookup tables for speed
_BYTE_TO_BRAILLE: list[str] = [chr(_BRAILLE_BASE + i) for i in range(256)]
_BRAILLE_TO_BYTE: dict[str, int] = {chr(_BRAILLE_BASE + i): i for i in range(256)}


def encode(data: bytes) -> str:
    """Encode arbitrary bytes as a Braille Unicode string.

    Args:
        data: Arbitrary byte sequence.

    Returns:
        String of Braille characters, one per input byte.

    Examples:
        >>> encode(b'\\x00')
        '⠀'
        >>> encode(b'\\xff')
        '⣿'
        >>> encode(b'hello')
        '⡨⡥⡬⡬⡯'
    """
    return "".join(_BYTE_TO_BRAILLE[b] for b in data)


def decode(braille: str) -> bytes:
    """Decode a Braille Unicode string back to bytes.

    Args:
        braille: String of Braille characters.

    Returns:
        Original byte sequence.

    Raises:
        ValueError: If string contains non-Braille characters.

    Examples:
        >>> decode('⡨⡥⡬⡬⡯')
        b'hello'
    """
    result = bytearray(len(braille))
    for i, char in enumerate(braille):
        cp = ord(char) - _BRAILLE_BASE
        if cp < 0 or cp > 255:
            raise ValueError(
                f"Character at position {i} is not a Braille pattern: "
                f"U+{ord(char):04X} ({char!r})"
            )
        result[i] = cp
    return bytes(result)


def encode_hex(hex_str: str) -> str:
    """Encode a hex string as Braille.

    Args:
        hex_str: Hex string (with or without '0x' prefix, spaces allowed).

    Returns:
        Braille-encoded string.

    Examples:
        >>> encode_hex('deadbeef')
        '⣞⣭⣾⣯'
    """
    hex_str = hex_str.replace(" ", "").replace("0x", "").replace("0X", "")
    return encode(bytes.fromhex(hex_str))


def encode_int(n: int, width: int = 4) -> str:
    """Encode an integer as fixed-width Braille (big-endian).

    Args:
        n: Non-negative integer.
        width: Number of bytes (= number of Braille characters).

    Returns:
        Fixed-width Braille string.

    Raises:
        ValueError: If n is negative or doesn't fit in width bytes.

    Examples:
        >>> encode_int(0, width=2)
        '⠀⠀'
        >>> encode_int(255, width=1)
        '⣿'
        >>> encode_int(65535, width=2)
        '⣿⣿'
    """
    if n < 0:
        raise ValueError(f"Cannot encode negative integer: {n}")
    max_val = (1 << (width * 8)) - 1
    if n > max_val:
        raise ValueError(
            f"Integer {n} does not fit in {width} bytes (max {max_val})"
        )
    return encode(n.to_bytes(width, byteorder="big"))


def decode_int(braille: str) -> int:
    """Decode a Braille string back to an integer (big-endian).

    Args:
        braille: Braille-encoded integer.

    Returns:
        The decoded integer.
    """
    return int.from_bytes(decode(braille), byteorder="big")


def is_braille(char: str) -> bool:
    """Check if a character is a Braille pattern (U+2800–U+28FF)."""
    cp = ord(char)
    return _BRAILLE_BASE <= cp <= _BRAILLE_BASE + 255


def braille_len(braille: str) -> int:
    """Return the number of bytes encoded in a Braille string.

    Same as len(braille) since the encoding is 1:1, but explicit
    about the semantic meaning.
    """
    return len(braille)


def to_hex(braille: str) -> str:
    """Convert Braille string to hex representation.

    Useful for debugging and display.

    Examples:
        >>> to_hex('⣞⣭⣾⣯')
        'deadbeef'
    """
    return decode(braille).hex()
