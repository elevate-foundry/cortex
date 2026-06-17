"""Tests for Braille fingerprints — SCLRecord → fixed-width Braille hash."""

import pytest
from src.scl.types import Anchor, Relation, Scope, SCLRecord, SCLDocument
from src.braille.fingerprint import (
    fingerprint, fingerprint_document, fingerprint_batch,
    fingerprint_match, similarity, hamming_distance,
)
from src.braille.codec import is_braille


class TestFingerprint:
    """Core fingerprinting tests."""

    def test_deterministic(self):
        """Same input always produces same fingerprint."""
        r = SCLRecord(Anchor("router"), Relation("select"), Scope({"tier": "L3"}))
        fp1 = fingerprint(r)
        fp2 = fingerprint(r)
        assert fp1 == fp2

    def test_default_width(self):
        r = SCLRecord(Anchor("test"), Relation("verify"), Scope({}))
        fp = fingerprint(r)
        assert len(fp) == 4  # default width

    def test_custom_width(self):
        r = SCLRecord(Anchor("test"), Relation("verify"), Scope({}))
        for width in [1, 2, 4, 8, 16]:
            fp = fingerprint(r, width=width)
            assert len(fp) == width

    def test_all_braille_chars(self):
        r = SCLRecord(Anchor("test"), Relation("verify"), Scope({"x": "y"}))
        fp = fingerprint(r, width=8)
        for char in fp:
            assert is_braille(char)

    def test_different_inputs_different_fingerprints(self):
        r1 = SCLRecord(Anchor("router"), Relation("select"), Scope({"tier": "L3"}))
        r2 = SCLRecord(Anchor("router"), Relation("select"), Scope({"tier": "L5"}))
        assert fingerprint(r1) != fingerprint(r2)

    def test_collision_resistance(self):
        """Generate 100 different records, check for collisions at width=4."""
        fps = set()
        for i in range(100):
            r = SCLRecord(Anchor(f"test_{i}"), Relation("check"), Scope({"n": str(i)}))
            fps.add(fingerprint(r))
        # At width=4 (32 bits), 100 records should have ~0 collisions
        assert len(fps) == 100


class TestDocumentFingerprint:
    """Document-level fingerprinting."""

    def test_deterministic(self):
        doc = SCLDocument(records=[
            SCLRecord(Anchor("a"), Relation("b"), Scope({"c": "d"})),
            SCLRecord(Anchor("e"), Relation("f"), Scope({"g": "h"})),
        ])
        assert fingerprint_document(doc) == fingerprint_document(doc)

    def test_default_width_8(self):
        doc = SCLDocument(records=[
            SCLRecord(Anchor("test"), Relation("x"), Scope({})),
        ])
        assert len(fingerprint_document(doc)) == 8

    def test_batch_equals_document(self):
        records = [
            SCLRecord(Anchor("a"), Relation("b"), Scope({})),
            SCLRecord(Anchor("c"), Relation("d"), Scope({})),
        ]
        doc = SCLDocument(records=records)
        assert fingerprint_batch(records) == fingerprint_document(doc)


class TestSimilarity:
    """Similarity and Hamming distance tests."""

    def test_identical_fingerprints(self):
        r = SCLRecord(Anchor("x"), Relation("y"), Scope({"z": "1"}))
        fp = fingerprint(r)
        assert similarity(fp, fp) == 1.0
        assert hamming_distance(fp, fp) == 0

    def test_different_fingerprints(self):
        r1 = SCLRecord(Anchor("a"), Relation("b"), Scope({"c": "1"}))
        r2 = SCLRecord(Anchor("x"), Relation("y"), Scope({"z": "2"}))
        fp1 = fingerprint(r1)
        fp2 = fingerprint(r2)
        sim = similarity(fp1, fp2)
        # Random-ish fingerprints should have ~50% similarity
        assert 0.0 <= sim <= 1.0
        assert sim < 1.0  # not identical

    def test_hamming_distance_range(self):
        r1 = SCLRecord(Anchor("a"), Relation("b"), Scope({}))
        r2 = SCLRecord(Anchor("c"), Relation("d"), Scope({}))
        fp1 = fingerprint(r1, width=4)
        fp2 = fingerprint(r2, width=4)
        hd = hamming_distance(fp1, fp2)
        assert 0 <= hd <= 32  # 4 bytes * 8 bits

    def test_similarity_plus_distance(self):
        """Similarity and Hamming distance should be complementary."""
        r1 = SCLRecord(Anchor("a"), Relation("b"), Scope({}))
        r2 = SCLRecord(Anchor("c"), Relation("d"), Scope({}))
        fp1 = fingerprint(r1, width=4)
        fp2 = fingerprint(r2, width=4)
        sim = similarity(fp1, fp2)
        hd = hamming_distance(fp1, fp2)
        total_bits = 32
        assert abs(sim - (total_bits - hd) / total_bits) < 0.001

    def test_different_length_raises(self):
        r = SCLRecord(Anchor("x"), Relation("y"), Scope({}))
        fp_short = fingerprint(r, width=2)
        fp_long = fingerprint(r, width=4)
        with pytest.raises(ValueError, match="same length"):
            similarity(fp_short, fp_long)

    def test_empty_fingerprints(self):
        assert similarity("", "") == 1.0

    def test_match(self):
        r = SCLRecord(Anchor("x"), Relation("y"), Scope({}))
        fp = fingerprint(r)
        assert fingerprint_match(fp, fp)
        fp2 = fingerprint(SCLRecord(Anchor("z"), Relation("w"), Scope({})))
        assert not fingerprint_match(fp, fp2)
