"""
Braille Tokenizer — 256-token fixed vocabulary for SCL.

Instead of BPE with arbitrary merges, this tokenizer uses the 256 Unicode
Braille characters (U+2800–U+28FF) as a bijective byte-level vocabulary.

Properties:
  - Vocab size: exactly 259 (256 bytes + BOS + EOS + PAD)
  - No UNK tokens: every byte is representable
  - Fixed forever: no retraining, no vocabulary drift
  - Accessible: output is literally Braille, readable by touch
  - Auditable: any model output can be decoded back to exact bytes
  - Token-efficient: 1 character = 1 token (LLMs treat Braille as single tokens)

Trust argument:
  A model whose vocabulary IS an accessibility standard has built-in
  transparency. You can't hide tokens when every token has a physical
  dot-pattern representation readable by humans with any sensory capability.

Architecture:
  encode("@hardware → state [cpu: M1]")
  → [BOS, ⡀, ⡨, ⡡, ⡲, ⡤, ⡷, ⡡, ⡲, ⡥, ⠠, ⢀, ⠠, ⡳, ⡴, ⡡, ..., EOS]

  decode([⡀, ⡨, ⡡, ⡲, ...])
  → "@hardware → state [cpu: M1]"
"""

from ..braille.codec import encode as braille_encode, decode as braille_decode

# Special token IDs (above the 256 byte range)
PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
VOCAB_SIZE = 259

# Special token strings
PAD_TOKEN = "⣿\u200b"  # braille full + zero-width space (distinguishable)
BOS_TOKEN = "⠿\u200b"  # braille 1-6 + zero-width space
EOS_TOKEN = "⢿\u200b"  # braille 2-6 + zero-width space

# The core mapping: byte value → braille character
_BRAILLE_BASE = 0x2800
_BYTE_TO_BRAILLE = [chr(_BRAILLE_BASE + i) for i in range(256)]
_BRAILLE_TO_BYTE = {chr(_BRAILLE_BASE + i): i for i in range(256)}


class BrailleTokenizer:
    """
    Fixed 256-token Braille vocabulary for SCL text.

    Every byte maps 1:1 to a Braille glyph. No merges, no UNK, no drift.

    Usage:
        tok = BrailleTokenizer()
        ids = tok.encode("@hardware → state [cpu: M1]")
        text = tok.decode(ids)
        assert text == "@hardware → state [cpu: M1]"
    """

    def __init__(self, max_length: int = 512, add_special_tokens: bool = True):
        self.max_length = max_length
        self.add_special = add_special_tokens
        self.vocab_size = VOCAB_SIZE
        self.pad_id = PAD_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID

    def encode(self, text: str, add_special_tokens: bool = None,
               max_length: int = None, padding: bool = False) -> list[int]:
        """
        Encode text to token IDs via Braille byte encoding.

        Args:
            text: SCL text string
            add_special_tokens: prepend BOS, append EOS
            max_length: truncate/pad to this length
            padding: pad to max_length with PAD_ID

        Returns:
            List of token IDs (0-258)
        """
        add_special = add_special_tokens if add_special_tokens is not None else self.add_special
        max_len = max_length or self.max_length

        # Convert text to bytes, then to token IDs
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)  # Each byte IS a token ID (0-255)

        # Add special tokens
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]

        # Truncate
        if len(ids) > max_len:
            ids = ids[:max_len]
            if add_special and ids[-1] != self.eos_id:
                ids[-1] = self.eos_id  # Ensure EOS is preserved

        # Pad
        if padding and len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: list of token IDs
            skip_special_tokens: remove BOS/EOS/PAD from output

        Returns:
            Decoded string
        """
        byte_ids = []
        for tok_id in ids:
            if skip_special_tokens and tok_id in (self.pad_id, self.bos_id, self.eos_id):
                continue
            if 0 <= tok_id <= 255:
                byte_ids.append(tok_id)

        return bytes(byte_ids).decode("utf-8", errors="replace")

    def to_braille(self, text: str) -> str:
        """
        Convert SCL text to its Braille representation.

        This is the human-readable (and tactile-readable) form of the tokens.
        Each character in the output is one token.
        """
        return braille_encode(text.encode("utf-8"))

    def from_braille(self, braille_text: str) -> str:
        """
        Convert Braille representation back to SCL text.
        """
        return braille_decode(braille_text).decode("utf-8", errors="replace")

    def ids_to_braille(self, ids: list[int], skip_special: bool = True) -> str:
        """
        Convert token IDs to Braille character string.

        This is the visual/tactile representation of what the model "sees".
        """
        chars = []
        for tok_id in ids:
            if skip_special and tok_id in (self.pad_id, self.bos_id, self.eos_id):
                if tok_id == self.bos_id:
                    chars.append(BOS_TOKEN)
                elif tok_id == self.eos_id:
                    chars.append(EOS_TOKEN)
                continue
            if 0 <= tok_id <= 255:
                chars.append(_BYTE_TO_BRAILLE[tok_id])
        return "".join(chars)

    def braille_to_ids(self, braille_text: str) -> list[int]:
        """
        Convert Braille character string to token IDs.
        """
        ids = []
        for char in braille_text:
            if char in _BRAILLE_TO_BYTE:
                ids.append(_BRAILLE_TO_BYTE[char])
            elif char == BOS_TOKEN[0]:
                ids.append(self.bos_id)
            elif char == EOS_TOKEN[0]:
                ids.append(self.eos_id)
        return ids

    def batch_encode(self, texts: list[str], max_length: int = None,
                     padding: bool = True) -> dict:
        """
        Encode a batch of texts with padding.

        Returns dict compatible with PyTorch DataLoader:
          {"input_ids": [[...], ...], "attention_mask": [[...], ...]}
        """
        max_len = max_length or self.max_length
        all_ids = []
        attention_masks = []

        for text in texts:
            ids = self.encode(text, max_length=max_len, padding=padding)
            mask = [1 if tok_id != self.pad_id else 0 for tok_id in ids]
            all_ids.append(ids)
            attention_masks.append(mask)

        return {
            "input_ids": all_ids,
            "attention_mask": attention_masks,
        }

    def __repr__(self):
        return f"BrailleTokenizer(vocab_size={self.vocab_size}, max_length={self.max_length})"

    # --- Compatibility methods for HuggingFace-style interfaces ---

    def __call__(self, text, **kwargs):
        """HuggingFace-compatible call interface."""
        if isinstance(text, str):
            return self.batch_encode([text], **kwargs)
        return self.batch_encode(text, **kwargs)

    @property
    def model_max_length(self):
        return self.max_length

    @property
    def pad_token_id(self):
        return self.pad_id

    @property
    def eos_token_id(self):
        return self.eos_id

    @property
    def bos_token_id(self):
        return self.bos_id


def demonstrate():
    """Show the Braille tokenizer in action."""
    tok = BrailleTokenizer(max_length=128)

    test_cases = [
        "@hardware → state [cpu: Apple M1, cores: 8, ram_mb: 16384]",
        "@router → select [tier: L3, model: qwen3:4b, confidence: 0.82]",
        "@init → boot [phase: cold_start, pid: 1]",
        "@safety → deny [target: /dev/mem, action: write, severity: critical]",
    ]

    print("Braille Tokenizer Demo")
    print("=" * 70)
    print(f"Vocab size: {tok.vocab_size} (256 bytes + BOS + EOS + PAD)")
    print(f"Properties: bijective, lossless, fixed, accessible")
    print()

    for text in test_cases:
        ids = tok.encode(text)
        braille = tok.to_braille(text)
        decoded = tok.decode(ids)

        print(f"SCL:     {text}")
        print(f"Braille: {braille}")
        print(f"Tokens:  {len(ids)} ids")
        print(f"Round:   {'✓' if decoded == text else '✗'} {decoded[:50]}...")
        print()

    # Show accessibility property
    print("Accessibility Audit:")
    print("-" * 70)
    print("Every token ID maps to exactly one Braille dot pattern:")
    print("  0x40 '@' → ⡀ (dot 7)")
    print("  0x68 'h' → ⡨ (dots 4,6,7)")
    print("  0x20 ' ' → ⠠ (dot 6)")
    print("  0x5B '[' → ⡛ (dots 1,2,4,5,7)")
    print()
    print("A blind operator can read the model's tokenized state by touch.")
    print("The model cannot produce tokens that are not Braille-representable.")
    print("Trust is architectural, not behavioural.")


if __name__ == "__main__":
    demonstrate()
