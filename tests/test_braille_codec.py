"""Tests for Braille codec — encode/decode bytes ↔ Braille Unicode."""

import random
import pytest
from src.braille.codec import (
    encode, decode, encode_hex, encode_int, decode_int,
    is_braille, braille_len, to_hex,
)


class TestEncodeDecode:
    """Core bijective encoding tests."""

    def test_empty(self):
        assert encode(b"") == ""
        assert decode("") == b""

    def test_single_byte_zero(self):
        result = encode(b"\x00")
        assert result == "⠀"  # U+2800
        assert decode(result) == b"\x00"

    def test_single_byte_max(self):
        result = encode(b"\xff")
        assert result == "⣿"  # U+28FF
        assert decode(result) == b"\xff"

    def test_all_256_bytes(self):
        """Every byte value round-trips correctly."""
        for i in range(256):
            b = bytes([i])
            encoded = encode(b)
            assert len(encoded) == 1
            assert decode(encoded) == b
            # Verify codepoint math
            assert ord(encoded) == 0x2800 + i

    def test_hello(self):
        encoded = encode(b"hello")
        assert len(encoded) == 5
        assert decode(encoded) == b"hello"

    def test_random_round_trips(self):
        """100 random byte strings of various lengths all round-trip."""
        rng = random.Random(42)  # deterministic seed
        for _ in range(100):
            length = rng.randint(0, 128)
            data = bytes(rng.randint(0, 255) for _ in range(length))
            assert decode(encode(data)) == data

    def test_fixed_density(self):
        """Each byte produces exactly 1 Braille char (3 UTF-8 bytes)."""
        data = b"test data with various bytes"
        encoded = encode(data)
        assert len(encoded) == len(data)
        assert len(encoded.encode("utf-8")) == len(data) * 3


class TestDecodeErrors:
    """Error handling for invalid input."""

    def test_non_braille_char(self):
        with pytest.raises(ValueError, match="not a Braille pattern"):
            decode("hello")

    def test_mixed_braille_and_ascii(self):
        with pytest.raises(ValueError):
            decode("⠁X⠃")


class TestHexEncoding:
    """Hex string encoding."""

    def test_deadbeef(self):
        result = encode_hex("deadbeef")
        assert to_hex(result) == "deadbeef"

    def test_with_prefix(self):
        result = encode_hex("0xdeadbeef")
        assert to_hex(result) == "deadbeef"

    def test_with_spaces(self):
        result = encode_hex("de ad be ef")
        assert to_hex(result) == "deadbeef"

    def test_empty_hex(self):
        assert encode_hex("") == ""


class TestIntEncoding:
    """Integer encoding."""

    def test_zero(self):
        result = encode_int(0, width=2)
        assert decode_int(result) == 0

    def test_max_single_byte(self):
        result = encode_int(255, width=1)
        assert decode_int(result) == 255

    def test_max_two_bytes(self):
        result = encode_int(65535, width=2)
        assert decode_int(result) == 65535

    def test_arbitrary_value(self):
        result = encode_int(12345, width=4)
        assert decode_int(result) == 12345

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="negative"):
            encode_int(-1)

    def test_overflow_raises(self):
        with pytest.raises(ValueError, match="does not fit"):
            encode_int(256, width=1)

    def test_width_determines_length(self):
        for width in [1, 2, 4, 8]:
            result = encode_int(0, width=width)
            assert len(result) == width


class TestUtilities:
    """Utility function tests."""

    def test_is_braille(self):
        assert is_braille("⠀")  # U+2800
        assert is_braille("⣿")  # U+28FF
        assert not is_braille("A")
        assert not is_braille("0")

    def test_braille_len(self):
        assert braille_len(encode(b"hello")) == 5
        assert braille_len("") == 0

    def test_to_hex(self):
        assert to_hex(encode(b"\xde\xad")) == "dead"
