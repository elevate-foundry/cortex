"""Tests for SCL parser — text → SCL AST."""

import pytest
from src.scl.types import Anchor, Relation, Scope, SCLRecord, SCLDocument
from src.scl.parser import parse_record, parse_document, parse_record_strict, ParseError


class TestParseRecord:
    """Test single-record parsing."""

    def test_basic_record(self):
        r = parse_record_strict("@router → select [model: qwen3:4b]")
        assert r.anchor.name == "router"
        assert r.relation.verb == "select"
        assert r.scope.get("model") == "qwen3:4b"

    def test_multiple_entries(self):
        r = parse_record_strict("@router → select [model: qwen3:4b, confidence: 0.82]")
        assert r.scope.get("model") == "qwen3:4b"
        assert r.scope.get("confidence") == "0.82"

    def test_ascii_arrow(self):
        r = parse_record_strict("@agent_1 -> own [role: foundation]")
        assert r.anchor.name == "agent_1"
        assert r.relation.verb == "own"

    def test_bare_values(self):
        r = parse_record_strict("@cortex → missing [tests, scl_parser, braille_codec]")
        assert "tests" in r.scope.entries
        assert "scl_parser" in r.scope.entries

    def test_empty_scope(self):
        r = parse_record_strict("@system → reset []")
        assert len(r.scope) == 0

    def test_dotted_anchor(self):
        r = parse_record_strict("@agent.foundation → own [dir: tests/]")
        assert r.anchor.name == "agent.foundation"

    def test_colon_in_value(self):
        """Values can contain colons (e.g. model tags)."""
        r = parse_record_strict("@core → answer [model: granite3.3:8b, family: granite]")
        assert r.scope.get("model") == "granite3.3:8b"

    def test_round_trip_text(self):
        original = "@router → select [model: qwen3:4b, confidence: 0.82]"
        r = parse_record_strict(original)
        reparsed = parse_record_strict(r.to_text())
        assert r == reparsed

    def test_round_trip_bytes(self):
        r = parse_record_strict("@task → classify [category: code, complexity: 0.45]")
        r2 = SCLRecord.from_bytes(r.to_bytes())
        assert r == r2

    def test_round_trip_dict(self):
        r = parse_record_strict("@swarm → query [models: 4, families: 3]")
        r2 = SCLRecord.from_dict(r.to_dict())
        assert r == r2


class TestParseErrors:
    """Test error handling."""

    def test_missing_anchor(self):
        result = parse_record("not valid scl")
        assert not result.ok
        assert "Expected '@'" in result.error.message

    def test_missing_arrow(self):
        result = parse_record("@router select [tier: L3]")
        assert not result.ok
        assert "→" in result.error.message

    def test_missing_scope(self):
        result = parse_record("@router → select")
        assert not result.ok
        assert "[" in result.error.message

    def test_unmatched_bracket(self):
        result = parse_record("@router → select [model: qwen3:4b")
        assert not result.ok
        assert "Unmatched" in result.error.message

    def test_empty_input(self):
        result = parse_record("")
        assert not result.ok

    def test_comment_line(self):
        result = parse_record("# this is a comment")
        assert not result.ok

    def test_strict_raises(self):
        with pytest.raises(ValueError):
            parse_record_strict("not valid")


class TestParseDocument:
    """Test multi-line document parsing."""

    def test_basic_document(self):
        text = """@task → classify [category: code]
@router → select [tier: L3]"""
        doc = parse_document(text)
        assert len(doc) == 2

    def test_skips_comments_and_blanks(self):
        text = """# Header
@task → classify [category: code]

// Comment
@router → select [tier: L3]

"""
        doc = parse_document(text)
        assert len(doc) == 2

    def test_skips_non_scl_lines(self):
        text = """Some prose here
@task → classify [category: code]
More prose
@router → select [tier: L3]"""
        doc = parse_document(text)
        assert len(doc) == 2

    def test_strict_mode_raises(self):
        text = """@task → classify [category: code]
bad line here starting with @invalid"""
        with pytest.raises(ValueError):
            parse_document(text, strict=True)

    def test_empty_document(self):
        doc = parse_document("")
        assert len(doc) == 0

    def test_document_filter_by_anchor(self):
        text = """@cortex → status [modules: 20]
@cortex → missing [tests]
@agent_1 → own [dir: tests/]"""
        doc = parse_document(text)
        cortex_docs = doc.filter_by_anchor("cortex")
        assert len(cortex_docs) == 2

    def test_document_filter_by_verb(self):
        text = """@agent_1 → own [dir: tests/]
@agent_2 → own [dir: src/scl/]
@agent_3 → depend [on: agent_2]"""
        doc = parse_document(text)
        own_docs = doc.filter_by_verb("own")
        assert len(own_docs) == 2

    def test_document_content_hash_deterministic(self):
        text = "@router → select [tier: L3]"
        doc1 = parse_document(text)
        doc2 = parse_document(text)
        assert doc1.content_hash() == doc2.content_hash()

    def test_document_round_trip(self):
        """Parse → emit → parse should preserve records."""
        text = """@task → classify [category: code, complexity: 0.45]
@router → select [tier: L3, confidence: 0.82]
@agent_1 → own [role: foundation, dir: tests/]"""
        doc1 = parse_document(text)
        emitted = doc1.to_text()
        doc2 = parse_document(emitted)
        assert len(doc1) == len(doc2)
        for r1, r2 in zip(doc1, doc2):
            assert r1 == r2
