"""
Braille Fingerprints — SCLRecord → fixed-width Braille hash.

Maps SCL records and documents to compact, fixed-width Braille strings
using SHA-256 hashing. These fingerprints serve as:

  1. **Dedup keys** — identical records produce identical fingerprints
  2. **LSH approximation** — similar records often share prefix bits
  3. **Gossip payloads** — 4-char fingerprint = 32 bits, cheap to propagate
  4. **Convergence checks** — Hamming distance between fingerprints estimates
     semantic divergence between agents

At scale (N → ∞):
  - Agents broadcast fingerprints instead of full state
  - Cluster-heads aggregate fingerprints for sub-swarms
  - Hamming distance < ε ⟹ agents are in agreement
  - Hamming distance > threshold ⟹ trigger challenger verification
"""

import hashlib

from .codec import encode
from ..scl.types import SCLRecord, SCLDocument


def fingerprint(record: SCLRecord, width: int = 4) -> str:
    """SCL record → fixed-width Braille fingerprint.

    Process:
      1. Serialize record to canonical bytes (record.to_bytes())
      2. SHA-256 hash
      3. Take first `width` bytes of hash
      4. Encode as Braille

    Args:
        record: The SCL record to fingerprint.
        width: Number of Braille characters (= bytes of hash). Default 4 (32 bits).

    Returns:
        String of `width` Braille characters.

    Examples:
        >>> from src.scl.types import Anchor, Relation, Scope, SCLRecord
        >>> r = SCLRecord(Anchor('router'), Relation('select'), Scope({'model': 'qwen3:4b'}))
        >>> len(fingerprint(r))
        4
        >>> fingerprint(r) == fingerprint(r)  # deterministic
        True
    """
    h = hashlib.sha256(record.to_bytes()).digest()
    return encode(h[:width])


def fingerprint_document(doc: SCLDocument, width: int = 8) -> str:
    """Full SCL document → single Braille fingerprint.

    Hashes the concatenated binary of all records.

    Args:
        doc: The SCL document to fingerprint.
        width: Number of Braille characters. Default 8 (64 bits).

    Returns:
        String of `width` Braille characters.
    """
    h = hashlib.sha256(doc.to_bytes()).digest()
    return encode(h[:width])


def fingerprint_batch(records: list[SCLRecord], width: int = 8) -> str:
    """Multiple SCL records → single Braille fingerprint.

    Convenience wrapper: creates a temporary document and fingerprints it.
    """
    doc = SCLDocument(records=records)
    return fingerprint_document(doc, width=width)


def fingerprint_match(fp1: str, fp2: str) -> bool:
    """Compare two fingerprints for exact equality."""
    return fp1 == fp2


def similarity(fp1: str, fp2: str) -> float:
    """Hamming-distance-based similarity between two fingerprints.

    Compares bit-by-bit. Returns 0.0 (completely different) to 1.0 (identical).

    At scale, this is the convergence metric:
      - similarity > 0.9 → agents agree, no action needed
      - similarity 0.5–0.9 → partial agreement, may need verification
      - similarity < 0.5 → disagreement, trigger challenger/swarm

    Args:
        fp1: First Braille fingerprint.
        fp2: Second Braille fingerprint.

    Returns:
        Float in [0.0, 1.0].

    Raises:
        ValueError: If fingerprints have different lengths.
    """
    if len(fp1) != len(fp2):
        raise ValueError(
            f"Fingerprints must have same length: {len(fp1)} vs {len(fp2)}"
        )

    if not fp1:
        return 1.0

    total_bits = len(fp1) * 8
    matching_bits = 0

    for c1, c2 in zip(fp1, fp2):
        b1 = ord(c1) - 0x2800
        b2 = ord(c2) - 0x2800
        # XOR gives bits that differ; popcount gives number of differing bits
        xor = b1 ^ b2
        differing = bin(xor).count("1")
        matching_bits += 8 - differing

    return matching_bits / total_bits


def hamming_distance(fp1: str, fp2: str) -> int:
    """Raw Hamming distance in bits between two fingerprints.

    Args:
        fp1: First Braille fingerprint.
        fp2: Second Braille fingerprint.

    Returns:
        Number of differing bits.
    """
    if len(fp1) != len(fp2):
        raise ValueError(
            f"Fingerprints must have same length: {len(fp1)} vs {len(fp2)}"
        )

    distance = 0
    for c1, c2 in zip(fp1, fp2):
        b1 = ord(c1) - 0x2800
        b2 = ord(c2) - 0x2800
        distance += bin(b1 ^ b2).count("1")
    return distance
