"""
CKM Dataset Pipeline — raw → normalized → tokenized mmap.

The three-stage pipeline:
  1. Raw: JSONL training pairs (from data_generator.py / trace_generator.py)
  2. Normalized: Deduplicated, curriculum-ordered, quality-filtered
  3. Tokenized: Pre-packed sequences in mmap format (zero re-tokenization)

The key insight: training should start from pre-tokenized packed sequences.
Tokenization is expensive and deterministic — cache it once.

Format of tokenized cache:
  - Header: magic + version + vocab_size + seq_len + n_sequences + checksum
  - Data: packed int16 token IDs in mmap-able format
  - Index: offset table for random access

Curriculum ordering (convergence optimization):
  1. Grammar (valid SCL structure)
  2. Verb/object correctness (observe/configure/deny distinction)
  3. Hardware-to-config mapping
  4. Failure recovery traces
  5. Full boot traces
"""

import hashlib
import json
import mmap
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator

DATASET_MAGIC = b"CKD\x01"  # Cortex Kernel Dataset
DATASET_VERSION = 1


# ---------------------------------------------------------------------------
# Curriculum levels (training order)
# ---------------------------------------------------------------------------

CURRICULUM_ORDER = {
    "grammar": 0,          # Pure SCL syntax examples
    "verb_selection": 1,   # observe/configure/deny distinction
    "safety_denial": 2,    # refuse /dev/mem and friends
    "boot_config": 3,      # hardware → optimal config
    "routing": 4,          # request → tier selection
    "recovery": 5,         # failure → repair traces
    "full_trace": 6,       # multi-step operational traces
    "policy": 7,           # self-modification patterns
}


def curriculum_key(pair: dict) -> int:
    """Assign a curriculum level to a training pair for ordering."""
    source = pair.get("source", "")
    output = pair.get("output", "")

    # Grammar: anything that's purely about SCL structure
    if "grammar" in source:
        return CURRICULUM_ORDER["grammar"]

    # Safety: denial records
    if "deny" in output or "unsafe" in output.lower():
        return CURRICULUM_ORDER["safety_denial"]

    # Verb selection: observe/configure distinction
    if source in ("synthetic_boot", "boot"):
        if "observe" in output or "state" in output:
            return CURRICULUM_ORDER["verb_selection"]
        return CURRICULUM_ORDER["boot_config"]

    # Recovery/repair traces
    if "repair" in source or "fail" in output:
        return CURRICULUM_ORDER["recovery"]

    # Routing
    if "route" in source:
        return CURRICULUM_ORDER["routing"]

    # Full traces
    if "trace" in source:
        return CURRICULUM_ORDER["full_trace"]

    # Policy
    if "policy" in source:
        return CURRICULUM_ORDER["policy"]

    return CURRICULUM_ORDER["boot_config"]  # default


# ---------------------------------------------------------------------------
# Tokenizer (character-level BPE for SCL — minimal, no dependencies)
# ---------------------------------------------------------------------------

class SCLTokenizer:
    """
    Minimal BPE tokenizer specialized for SCL text.

    SCL has a small vocabulary:
      - Structural: @ → [ ] , :
      - Verbs: configure, mutate, observe, deny, boot, select, ...
      - Anchors: hardware, boot, policy, service, ...
      - Values: numbers, model names, paths

    We use character-level with common SCL bigrams merged.
    This keeps vocab small (~512 tokens) for a tiny model.
    """

    # Special tokens
    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._built = False

        # Reserve special tokens
        self.token_to_id["<pad>"] = self.PAD
        self.token_to_id["<bos>"] = self.BOS
        self.token_to_id["<eos>"] = self.EOS
        self.token_to_id["<unk>"] = self.UNK
        for k, v in list(self.token_to_id.items()):
            self.id_to_token[v] = k

    def build_vocab(self, texts: list[str]) -> None:
        """Build vocabulary from training texts using byte-pair encoding."""
        # Start with all characters
        char_counts: dict[str, int] = {}
        for text in texts:
            for ch in text:
                char_counts[ch] = char_counts.get(ch, 0) + 1

        # Add all unique characters
        next_id = 4  # after special tokens
        for ch in sorted(char_counts.keys(), key=lambda c: -char_counts[c]):
            if next_id >= self.vocab_size:
                break
            if ch not in self.token_to_id:
                self.token_to_id[ch] = next_id
                self.id_to_token[next_id] = ch
                next_id += 1

        # Now do BPE merges for common bigrams
        # Count bigrams
        bigram_counts: dict[str, int] = {}
        for text in texts:
            for i in range(len(text) - 1):
                bigram = text[i:i+2]
                bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

        # Merge most common bigrams until vocab is full
        for bigram, count in sorted(bigram_counts.items(), key=lambda x: -x[1]):
            if next_id >= self.vocab_size:
                break
            if bigram not in self.token_to_id and count > 5:
                self.token_to_id[bigram] = next_id
                self.id_to_token[next_id] = bigram
                next_id += 1

        # Add common SCL tokens as full strings
        scl_tokens = [
            "→", "@", "configure", "mutate", "observe", "deny", "boot",
            "select", "state", "failed", "restart", "hardware", "service",
            "policy", "network", "inference", "optimal_threads",
            "optimal_gpu_layers", "optimal_ctx_size", "optimal_backend",
            "llama_cpp", "ollama", "/dev/mem", "unsafe",
        ]
        for token in scl_tokens:
            if next_id >= self.vocab_size:
                break
            if token not in self.token_to_id:
                self.token_to_id[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1

        self._built = True

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        if not self._built:
            # Fallback: character-level
            return [self.token_to_id.get(ch, self.UNK) for ch in text]

        tokens = [self.BOS]
        i = 0
        while i < len(text):
            # Greedy longest match
            best_len = 0
            best_id = self.UNK
            for length in range(min(20, len(text) - i), 0, -1):
                substr = text[i:i+length]
                if substr in self.token_to_id:
                    best_len = length
                    best_id = self.token_to_id[substr]
                    break
            if best_len == 0:
                tokens.append(self.UNK)
                i += 1
            else:
                tokens.append(best_id)
                i += best_len
        tokens.append(self.EOS)
        return tokens

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        result = []
        for token_id in ids:
            if token_id in (self.PAD, self.BOS, self.EOS):
                continue
            result.append(self.id_to_token.get(token_id, "?"))
        return "".join(result)

    def save(self, path: str) -> None:
        """Save tokenizer vocabulary."""
        data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str) -> "SCLTokenizer":
        """Load tokenizer vocabulary."""
        data = json.loads(Path(path).read_text())
        tok = cls(vocab_size=data["vocab_size"])
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {int(v): k for k, v in data["token_to_id"].items()}
        tok._built = True
        return tok


# ---------------------------------------------------------------------------
# Tokenized mmap dataset
# ---------------------------------------------------------------------------

@dataclass
class DatasetHeader:
    """Header for the mmap tokenized dataset."""
    magic: bytes = DATASET_MAGIC
    version: int = DATASET_VERSION
    vocab_size: int = 512
    seq_len: int = 256
    n_sequences: int = 0
    checksum: str = ""


class TokenizedDataset:
    """
    Mmap-backed tokenized dataset for fast training iteration.

    Structure:
      [header.bin]   — JSON metadata
      [data.bin]     — packed int16 tokens, shape (n_sequences, seq_len)
      [vocab.json]   — tokenizer vocabulary

    Usage:
      dataset = TokenizedDataset.build(jsonl_path, output_dir)
      dataset = TokenizedDataset.load(output_dir)
      batch = dataset.get_batch(batch_size=8)
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.header: Optional[DatasetHeader] = None
        self.tokenizer: Optional[SCLTokenizer] = None
        self._mmap = None
        self._fd = None

    @classmethod
    def build(
        cls,
        jsonl_path: str,
        output_dir: str,
        seq_len: int = 256,
        vocab_size: int = 512,
        min_quality: float = 0.3,
    ) -> "TokenizedDataset":
        """
        Build tokenized mmap dataset from JSONL training data.

        Pipeline:
          1. Load and filter pairs by quality
          2. Curriculum-order pairs
          3. Build tokenizer vocabulary
          4. Tokenize all pairs into packed sequences
          5. Write mmap file with header
          6. Compute checksum
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Load pairs
        pairs = []
        with open(jsonl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                pair = json.loads(line)
                if pair.get("quality", 1.0) >= min_quality:
                    pairs.append(pair)

        if not pairs:
            raise ValueError(f"No valid training pairs in {jsonl_path}")

        # 2. Curriculum ordering
        pairs.sort(key=curriculum_key)

        # 3. Build tokenizer
        all_texts = [p["input"] + "\n" + p["output"] for p in pairs]
        tokenizer = SCLTokenizer(vocab_size=vocab_size)
        tokenizer.build_vocab(all_texts)
        tokenizer.save(str(output_path / "vocab.json"))

        # 4. Tokenize into packed sequences
        sequences = []
        for pair in pairs:
            # Format: <bos> input \n output <eos>
            full_text = pair["input"] + "\n" + pair["output"]
            token_ids = tokenizer.encode(full_text)

            # Pad or truncate to seq_len
            if len(token_ids) > seq_len:
                token_ids = token_ids[:seq_len - 1] + [tokenizer.EOS]
            else:
                token_ids = token_ids + [tokenizer.PAD] * (seq_len - len(token_ids))

            sequences.append(token_ids)

        n_sequences = len(sequences)

        # 5. Write mmap data file (int16 packed)
        data_path = output_path / "data.bin"
        with open(data_path, "wb") as f:
            for seq in sequences:
                for token_id in seq:
                    f.write(struct.pack("<H", token_id))  # uint16

        # 6. Compute checksum
        sha = hashlib.sha256()
        with open(data_path, "rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        checksum = sha.hexdigest()[:16]

        # Write header
        header = DatasetHeader(
            vocab_size=vocab_size,
            seq_len=seq_len,
            n_sequences=n_sequences,
            checksum=checksum,
        )
        header_data = {
            "magic": DATASET_MAGIC.hex(),
            "version": DATASET_VERSION,
            "vocab_size": vocab_size,
            "seq_len": seq_len,
            "n_sequences": n_sequences,
            "checksum": checksum,
            "created_at": int(time.time()),
            "source": str(jsonl_path),
            "curriculum_levels": len(set(curriculum_key(p) for p in pairs)),
        }
        (output_path / "header.json").write_text(json.dumps(header_data, indent=2))

        # Return loaded dataset
        dataset = cls(str(output_path))
        dataset.header = header
        dataset.tokenizer = tokenizer
        return dataset

    @classmethod
    def load(cls, data_dir: str) -> "TokenizedDataset":
        """Load a pre-built tokenized dataset via mmap."""
        dataset = cls(data_dir)
        path = Path(data_dir)

        # Load header
        header_data = json.loads((path / "header.json").read_text())
        dataset.header = DatasetHeader(
            vocab_size=header_data["vocab_size"],
            seq_len=header_data["seq_len"],
            n_sequences=header_data["n_sequences"],
            checksum=header_data["checksum"],
        )

        # Load tokenizer
        dataset.tokenizer = SCLTokenizer.load(str(path / "vocab.json"))

        # Mmap the data
        data_path = path / "data.bin"
        dataset._fd = os.open(str(data_path), os.O_RDONLY)
        size = os.fstat(dataset._fd).st_size
        dataset._mmap = mmap.mmap(dataset._fd, size, access=mmap.ACCESS_READ)

        return dataset

    def get_sequence(self, idx: int) -> list[int]:
        """Get a single tokenized sequence by index."""
        if self._mmap is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        offset = idx * self.header.seq_len * 2  # 2 bytes per uint16
        tokens = []
        for i in range(self.header.seq_len):
            pos = offset + i * 2
            token_id = struct.unpack("<H", self._mmap[pos:pos+2])[0]
            tokens.append(token_id)
        return tokens

    def get_batch(self, indices: list[int]) -> list[list[int]]:
        """Get a batch of sequences."""
        return [self.get_sequence(i) for i in indices]

    def iter_batches(self, batch_size: int, shuffle: bool = True) -> Iterator[list[list[int]]]:
        """Iterate over the dataset in batches."""
        import random as _random
        indices = list(range(self.header.n_sequences))
        if shuffle:
            _random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            yield self.get_batch(batch_indices)

    def close(self) -> None:
        """Release mmap resources."""
        if self._mmap:
            self._mmap.close()
        if self._fd is not None:
            os.close(self._fd)

    def stats(self) -> dict:
        """Dataset statistics."""
        return {
            "n_sequences": self.header.n_sequences,
            "seq_len": self.header.seq_len,
            "vocab_size": self.header.vocab_size,
            "checksum": self.header.checksum,
            "data_size_mb": round(
                self.header.n_sequences * self.header.seq_len * 2 / (1024 * 1024), 2
            ),
            "data_dir": str(self.data_dir),
        }

    def is_stale(self, jsonl_path: str) -> bool:
        """Check if the tokenized cache is stale vs the source JSONL."""
        header_path = self.data_dir / "header.json"
        if not header_path.exists():
            return True
        jsonl_mtime = Path(jsonl_path).stat().st_mtime
        cache_mtime = header_path.stat().st_mtime
        return jsonl_mtime > cache_mtime
