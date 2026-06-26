"""
CKM Braille-Native Training — 259-token transformer from scratch.

This is NOT a fine-tune. This is NOT wrapping Braille in someone else's tokenizer.

This is a transformer whose vocabulary IS the 256 Unicode Braille characters
plus BOS, EOS, and PAD. The embedding table is 259 × d_model. The output head
is d_model × 259. The model architecturally cannot produce a non-Braille token.

Architecture:
  - Vocab: 259 (256 Braille bytes + BOS + EOS + PAD)
  - Decoder-only transformer (RMSNorm, RoPE, SwiGLU, causal mask)
  - Weight-tied embeddings (tok_emb.weight == lm_head.weight)
  - Sizes: 5M / 15M / 30M parameters (selectable)

Trust property:
  The model's output space is exactly the set of Braille dot patterns.
  There is no subword tokenizer, no BPE merge table, no UNK token.
  Every token the model can ever produce is a tactile-readable glyph.
  A blind operator can read the model's raw output by touch.
  This is not a feature. It is the architecture.

Training:
  Input:  byte-level Braille token sequences (SCL → utf8 bytes → token IDs 0-255)
  Target: next-byte prediction (causal LM objective)
  Data:   synthetic SCL state transitions (boot, route, safety, trace)

Run:
  modal run src/ckm/modal_train_braille_native.py
  modal run src/ckm/modal_train_braille_native.py --model-size 15m --epochs 10
"""

import modal
import os
import time

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

app = modal.App("cortex-ckm-braille-native")

volume = modal.Volume.from_name("cortex-ckm-braille-native", create_if_missing=True)
VOLUME_PATH = "/vol"

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "numpy")
    .pip_install("gguf")
)


# ---------------------------------------------------------------------------
# Braille vocabulary constants
# ---------------------------------------------------------------------------

PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
VOCAB_SIZE = 259
_BRAILLE_BASE = 0x2800


# ---------------------------------------------------------------------------
# Model architecture (same as train_scratch.py, inlined for Modal)
# ---------------------------------------------------------------------------

# Model size presets: name → (n_layers, d_model, n_heads, d_ff)
MODEL_SIZES = {
    "5m":  (6,  256,  8,  1024),
    "15m": (8,  384,  8,  1536),
    "30m": (12, 512,  8,  2048),
    "60m": (16, 640, 10,  2560),
}


def build_braille_model(
    n_layers: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    max_seq_len: int = 512,
    device: str = "cuda",
):
    """Build a native Braille-vocabulary transformer.

    The embedding table has exactly 259 rows.
    The output head has exactly 259 columns.
    Every token the model can produce is a Braille glyph.
    """
    import torch
    import torch.nn as nn

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
            return (x.float() * norm).to(x.dtype) * self.weight

    class RotaryEmbedding(nn.Module):
        def __init__(self, dim, max_len=512):
            super().__init__()
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq)

        def forward(self, seq_len):
            t = torch.arange(seq_len, device=self.inv_freq.device).float()
            freqs = torch.outer(t, self.inv_freq)
            return torch.cat([freqs, freqs], dim=-1)

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary(x, freqs):
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        return x * cos + rotate_half(x) * sin

    class Attention(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.o_proj = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x, freqs, mask=None):
            B, T, C = x.shape
            q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

            q = apply_rotary(q, freqs[:T])
            k = apply_rotary(k, freqs[:T])

            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn.masked_fill(mask[:T, :T] == 0, float("-inf"))
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
            return self.o_proj(out)

    class FeedForward(nn.Module):
        """SwiGLU feed-forward."""
        def __init__(self, d_model, d_ff):
            super().__init__()
            self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
            self.up_proj = nn.Linear(d_model, d_ff, bias=False)
            self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        def forward(self, x):
            return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    class TransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads, d_ff):
            super().__init__()
            self.norm1 = RMSNorm(d_model)
            self.attn = Attention(d_model, n_heads)
            self.norm2 = RMSNorm(d_model)
            self.ff = FeedForward(d_model, d_ff)

        def forward(self, x, freqs, mask):
            x = x + self.attn(self.norm1(x), freqs, mask)
            x = x + self.ff(self.norm2(x))
            return x

    class BrailleTransformer(nn.Module):
        """
        A transformer whose vocabulary is exactly the 256 Braille glyphs + 3 specials.

        tok_emb: 259 × d_model  — each row is one Braille character
        lm_head: d_model × 259  — each column is one Braille character
        Weight-tied: tok_emb.weight == lm_head.weight

        The model CANNOT produce a token outside {⠀..⣿, BOS, EOS, PAD}.
        """
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
            self.rotary = RotaryEmbedding(d_model // n_heads, max_len=max_seq_len)
            self.layers = nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_ff)
                for _ in range(n_layers)
            ])
            self.norm = RMSNorm(d_model)
            self.lm_head = nn.Linear(d_model, VOCAB_SIZE, bias=False)
            # Weight tying — the output head shares parameters with the embedding
            self.lm_head.weight = self.tok_emb.weight

            # Causal mask
            mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
            self.register_buffer("causal_mask", mask)

            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, idx):
            B, T = idx.shape
            x = self.tok_emb(idx)
            freqs = self.rotary(T)
            for layer in self.layers:
                x = layer(x, freqs, self.causal_mask)
            x = self.norm(x)
            return self.lm_head(x)

        def count_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

        @torch.no_grad()
        def generate(self, prompt_ids, max_new_tokens=128, temperature=0.7):
            """Autoregressive generation — produces only Braille tokens."""
            self.eval()
            ids = prompt_ids.clone()
            for _ in range(max_new_tokens):
                # Crop to max_seq_len
                context = ids[:, -max_seq_len:]
                logits = self.forward(context)
                logits = logits[:, -1, :] / temperature
                # Zero out PAD logit to prevent generating padding
                logits[:, PAD_ID] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                ids = torch.cat([ids, next_id], dim=1)
                if next_id.item() == EOS_ID:
                    break
            return ids

    model = BrailleTransformer().to(device)
    return model


# ---------------------------------------------------------------------------
# Data generation — raw byte sequences
# ---------------------------------------------------------------------------

def generate_training_sequences(n_sequences: int = 10000, max_seq_len: int = 512):
    """Generate SCL training data as raw byte sequences.

    Each sequence is: [BOS] + utf8_bytes_of_scl + [EOS] + [PAD...]

    The model learns byte-level SCL prediction using only Braille token IDs.
    No BPE. No subwords. No UNK. Just bytes.
    """
    import random

    hardware_profiles = [
        ("Apple M1", 8, 16384, "apple", 16384, "aarch64"),
        ("Apple M2 Pro", 12, 32768, "apple", 32768, "aarch64"),
        ("Apple M4 Max", 16, 131072, "apple", 131072, "aarch64"),
        ("Intel i5-12400", 6, 16384, "none", 0, "x86_64"),
        ("AMD Ryzen 9 7950X", 16, 65536, "nvidia", 24576, "x86_64"),
        ("AMD Ryzen 7 5800X", 8, 32768, "nvidia", 10240, "x86_64"),
        ("Intel i7-13700K", 16, 32768, "nvidia", 12288, "x86_64"),
        ("Raspberry Pi 5", 4, 8192, "none", 0, "aarch64"),
        ("AMD EPYC 7763", 64, 524288, "nvidia", 81920, "x86_64"),
        ("Intel Celeron N5105", 4, 4096, "none", 0, "x86_64"),
        ("Intel Core i9-13900K", 24, 65536, "nvidia", 24576, "x86_64"),
        ("NVIDIA Jetson Orin", 6, 8192, "nvidia", 8192, "aarch64"),
        ("Apple M3 Pro", 12, 18432, "apple", 18432, "aarch64"),
        ("AMD Ryzen 5 5600G", 6, 16384, "amd", 2048, "x86_64"),
        ("Qualcomm Snapdragon X Elite", 12, 16384, "none", 0, "aarch64"),
    ]

    categories = [
        ("code", "generate_function", 0.6, "L3"),
        ("code", "fix_bug", 0.7, "L4"),
        ("code", "explain_code", 0.3, "L2"),
        ("code", "refactor", 0.5, "L3"),
        ("chat", "greeting", 0.05, "L0"),
        ("chat", "complex_reasoning", 0.8, "L5"),
        ("chat", "summarize", 0.4, "L2"),
        ("math", "arithmetic", 0.1, "L1"),
        ("math", "proof", 0.9, "L5"),
        ("tool", "web_search", 0.3, "L2"),
        ("tool", "multi_tool_chain", 0.8, "L4"),
    ]

    dangerous_targets = ["/dev/mem", "/dev/kmem", "/proc/kcore", "/dev/sda",
                         "/dev/nvme0", "/dev/port", "/dev/hda", "/dev/tty"]
    dangerous_verbs = ["write", "patch", "flash", "erase", "format", "overwrite"]

    models = ["qwen3:4b", "qwen3:8b", "granite3.3:8b", "llama3.2:3b",
              "phi4:14b", "gemma3:12b", "mistral:7b"]

    sequences = []

    def scl_to_seq(scl_text: str) -> list:
        """Convert SCL text to padded byte sequence with BOS/EOS."""
        raw = list(scl_text.encode("utf-8"))
        seq = [BOS_ID] + raw[:max_seq_len - 2] + [EOS_ID]
        # Pad
        if len(seq) < max_seq_len:
            seq += [PAD_ID] * (max_seq_len - len(seq))
        return seq

    # --- Boot config: input → output as one concatenated sequence ---
    for i in range(n_sequences // 4):
        hw = random.choice(hardware_profiles)
        cpu, cores, ram, gpu_type, vram, arch = hw
        cores_n = max(1, cores + random.randint(-2, 2))
        ram_n = max(1024, ram + random.randint(-2048, 2048))
        vram_n = max(0, vram + random.randint(-512, 512)) if vram > 0 else 0

        opt_threads = max(1, min(cores_n - 1, 16))
        if gpu_type == "none":
            opt_gpu, opt_ctx = 0, min(4096, ram_n // 4)
        elif gpu_type == "apple":
            opt_gpu, opt_ctx = 999, min(16384, vram_n // 2)
        elif gpu_type == "nvidia":
            opt_gpu = 999 if vram_n >= 8192 else min(32, vram_n // 256)
            opt_ctx = min(16384, max(2048, vram_n // 2))
        else:
            opt_gpu, opt_ctx = min(24, max(0, vram_n // 256)), min(8192, max(2048, vram_n // 2))

        scl = (
            f"@hardware → state [cpu: {cpu}, cores: {cores_n}, ram_mb: {ram_n}, "
            f"gpu_type: {gpu_type}, vram_mb: {vram_n}, arch: {arch}]\n"
            f"@cortex.boot → mutate [optimal_threads: {opt_threads}, "
            f"optimal_gpu_layers: {opt_gpu}, optimal_ctx_size: {opt_ctx}]"
        )
        sequences.append(scl_to_seq(scl))

    # --- Routing ---
    for i in range(n_sequences // 4):
        cat, subtype, complexity, tier = random.choice(categories)
        complexity = max(0.0, min(1.0, complexity + random.gauss(0, 0.12)))
        tokens = random.randint(5, 3000)
        confidence = max(0.3, min(0.99, 1.0 - abs(random.gauss(0, 0.15))))

        scl = (
            f"@task → classify [category: {cat}, subtype: {subtype}, "
            f"complexity: {complexity:.2f}, input_tokens: {tokens}]\n"
            f"@router → select [tier: {tier}, confidence: {confidence:.2f}]"
        )
        sequences.append(scl_to_seq(scl))

    # --- Safety denial ---
    for i in range(n_sequences // 8):
        target = random.choice(dangerous_targets)
        verb = random.choice(dangerous_verbs)
        scl = (
            f"@agent → request [{verb}: {target}, reason: optimization]\n"
            f"@safety → deny [target: {target}, action: {verb}, severity: critical]"
        )
        sequences.append(scl_to_seq(scl))

    # --- Policy mutations ---
    for i in range(n_sequences // 8):
        accuracy = random.uniform(0.3, 0.95)
        tier = random.choice(["L1", "L2", "L3", "L4", "L5"])
        model_name = random.choice(models)
        count = random.randint(5, 50)

        if accuracy < 0.5:
            action = f"demote_tier: {tier}, penalize_model: {model_name}"
        elif accuracy > 0.85:
            action = f"promote_tier: {tier}, prefer_model: {model_name}"
        else:
            action = f"maintain_tier: {tier}, monitor_model: {model_name}"

        scl = (
            f"@feedback → accumulated [model: {model_name}, tier: {tier}, "
            f"accuracy: {accuracy:.2f}, count: {count}]\n"
            f"@policy → mutate [{action}, confidence: {accuracy:.2f}]"
        )
        sequences.append(scl_to_seq(scl))

    # --- Boot traces (multi-step) ---
    for i in range(n_sequences // 4):
        hw = random.choice(hardware_profiles)
        cpu, cores, _, gpu_type, vram, arch = hw
        threads = max(1, cores - 1)
        gpu = 999 if vram > 4096 else 0
        boot_ms = random.choice(["800", "1200", "2000", "3500"])

        cold_or_warm = random.choice(["cold_start", "warm_start"])
        backend = random.choice(["llama_cpp", "ollama", "vllm"])

        scl = (
            f"@init → boot [phase: {cold_or_warm}, pid: 1]\n"
            f"@init → detect [subsystem: cpu, result: {cpu}, cores: {cores}, arch: {arch}]\n"
            f"@init → select [threads: {threads}, gpu_layers: {gpu}]\n"
            f"@init → spawn [type: {backend}, port: 8080]\n"
            f"@verifier → check [result: pass, boot_ms: {boot_ms}]"
        )
        sequences.append(scl_to_seq(scl))

    random.shuffle(sequences)
    return sequences


# ---------------------------------------------------------------------------
# Step 1: Generate dataset
# ---------------------------------------------------------------------------

@app.function(image=training_image, volumes={VOLUME_PATH: volume}, timeout=300)
def generate_dataset(n_sequences: int = 20000, max_seq_len: int = 512) -> dict:
    """Generate byte-level Braille training data."""
    import json
    import numpy as np
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("braille.datagen")

    log.info("=" * 60)
    log.info("BRAILLE-NATIVE DATASET GENERATION")
    log.info(f"  Vocab size: {VOCAB_SIZE} (256 Braille + BOS + EOS + PAD)")
    log.info(f"  Sequences:  {n_sequences}")
    log.info(f"  Seq length: {max_seq_len}")
    log.info("=" * 60)

    t0 = time.time()
    sequences = generate_training_sequences(n_sequences, max_seq_len)
    data = np.array(sequences, dtype=np.int16)

    # Split 90/10
    n_eval = max(100, len(sequences) // 10)
    n_train = len(sequences) - n_eval

    output_dir = f"{VOLUME_PATH}/dataset"
    os.makedirs(output_dir, exist_ok=True)

    # Save as numpy mmap for fast loading
    train_path = f"{output_dir}/train.npy"
    eval_path = f"{output_dir}/eval.npy"
    np.save(train_path, data[:n_train])
    np.save(eval_path, data[n_train:])

    elapsed = time.time() - t0
    stats = {
        "n_train": n_train,
        "n_eval": n_eval,
        "seq_len": max_seq_len,
        "vocab_size": VOCAB_SIZE,
        "encoding": "braille_byte_native_259",
        "architecture": "from_scratch_transformer",
        "tokenizer": "NONE — raw byte IDs are the vocabulary",
        "elapsed_s": round(elapsed, 1),
    }

    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Validate: show a sample decoded
    sample = data[0]
    byte_ids = [int(x) for x in sample if 0 <= x <= 255]
    decoded = bytes(byte_ids).decode("utf-8", errors="replace")
    braille = "".join(chr(_BRAILLE_BASE + int(x)) for x in sample if 0 <= x <= 255)
    log.info(f"\nSample SCL:     {decoded[:80]}...")
    log.info(f"Sample Braille: {braille[:40]}...")
    log.info(f"Token IDs:      {list(sample[:20])}...")
    log.info(f"\nGenerated {len(sequences)} sequences in {elapsed:.1f}s")

    volume.commit()
    return stats


# ---------------------------------------------------------------------------
# Step 2: Train from scratch
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    timeout=7200,  # 2h max
)
def train_model(
    model_size: str = "15m",
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    max_seq_len: int = 512,
) -> dict:
    """Train a native Braille transformer from scratch.

    The model has exactly 259 tokens in its vocabulary.
    Every token it can produce is a Braille dot pattern.
    """
    import json
    import logging
    import math
    import sys
    import numpy as np
    import torch
    import torch.nn as nn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{VOLUME_PATH}/training_native.log"),
        ],
    )
    log = logging.getLogger("braille.train")

    if model_size not in MODEL_SIZES:
        raise ValueError(f"Unknown size: {model_size}. Options: {list(MODEL_SIZES.keys())}")

    n_layers, d_model, n_heads, d_ff = MODEL_SIZES[model_size]

    log.info("=" * 70)
    log.info("  BRAILLE-NATIVE TRANSFORMER — From Scratch")
    log.info("  Vocab: 259 tokens. No BPE. No subwords. Just Braille bytes.")
    log.info("  The model's output space IS the Braille character set.")
    log.info("=" * 70)
    log.info(f"  Model:      ckm-braille-{model_size}")
    log.info(f"  Layers:     {n_layers}")
    log.info(f"  d_model:    {d_model}")
    log.info(f"  Heads:      {n_heads}")
    log.info(f"  FFN dim:    {d_ff}")
    log.info(f"  Vocab:      {VOCAB_SIZE} (256 Braille + 3 special)")
    log.info(f"  Seq len:    {max_seq_len}")
    log.info(f"  Epochs:     {epochs}")
    log.info(f"  Batch:      {batch_size}")
    log.info(f"  LR:         {learning_rate}")
    log.info(f"  GPU:        {torch.cuda.get_device_name(0)}")
    log.info(f"  VRAM:       {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
             if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
             else f"  VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log.info("")

    # --- Load data ---
    log.info("[1/4] Loading dataset...")
    train_path = f"{VOLUME_PATH}/dataset/train.npy"
    eval_path = f"{VOLUME_PATH}/dataset/eval.npy"
    if not os.path.exists(train_path):
        raise FileNotFoundError("Run generate_dataset() first!")

    train_data = np.load(train_path)
    eval_data = np.load(eval_path)
    log.info(f"  Train: {train_data.shape[0]} sequences")
    log.info(f"  Eval:  {eval_data.shape[0]} sequences")

    # --- Build model ---
    log.info("\n[2/4] Building Braille transformer...")
    model = build_braille_model(
        n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        max_seq_len=max_seq_len, device="cuda",
    )
    n_params = model.count_params()
    log.info(f"  Parameters: {n_params:,}")
    log.info(f"  Embedding:  {VOCAB_SIZE} × {d_model} = {VOCAB_SIZE * d_model:,} params")
    log.info(f"  lm_head:    weight-tied with embedding (shared)")
    log.info(f"  Output dim: {VOCAB_SIZE} (exactly 259 Braille tokens)")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.95),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    n_train = train_data.shape[0]
    steps_per_epoch = n_train // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = min(200, total_steps // 10)

    def get_lr(step):
        if step < warmup_steps:
            return learning_rate * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    # --- Train ---
    log.info(f"\n[3/4] Training ({total_steps} steps, {steps_per_epoch}/epoch)...")
    model.train()
    global_step = 0
    best_eval_loss = float("inf")
    t0 = time.time()

    output_dir = f"{VOLUME_PATH}/checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(n_train)

        epoch_loss = 0.0
        epoch_steps = 0

        for batch_start in range(0, n_train - batch_size, batch_size):
            batch_idx = indices[batch_start:batch_start + batch_size]
            batch = torch.tensor(train_data[batch_idx], dtype=torch.long, device="cuda")

            # Next-token prediction: input = seq[:-1], target = seq[1:]
            x = batch[:, :-1]
            y = batch[:, 1:]

            logits = model(x)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

            loss.backward()

            # LR schedule
            lr = get_lr(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # Log every 50 steps
            if global_step % 50 == 0:
                avg = epoch_loss / epoch_steps
                elapsed = time.time() - t0
                tok_per_sec = global_step * batch_size * max_seq_len / elapsed
                log.info(f"  step={global_step:>5} loss={avg:.4f} lr={lr:.2e} "
                         f"tok/s={tok_per_sec:.0f} epoch={epoch+1}/{epochs}")

        # Epoch eval
        model.eval()
        eval_loss = 0.0
        n_eval_batches = 0
        with torch.no_grad():
            for i in range(0, eval_data.shape[0] - batch_size, batch_size):
                batch = torch.tensor(eval_data[i:i+batch_size], dtype=torch.long, device="cuda")
                x, y = batch[:, :-1], batch[:, 1:]
                logits = model(x)
                eval_loss += criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1)).item()
                n_eval_batches += 1
        eval_loss /= max(1, n_eval_batches)

        log.info(f"\n  Epoch {epoch+1}/{epochs}: train_loss={epoch_loss/epoch_steps:.4f} "
                 f"eval_loss={eval_loss:.4f} (best={best_eval_loss:.4f})")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
            log.info(f"  ★ New best! Saved checkpoint.")
            volume.commit()

        # Generate a sample to show what the model produces
        model.eval()
        # Prompt: "@hardware → state ["
        prompt_text = "@hardware → state ["
        prompt_ids = [BOS_ID] + list(prompt_text.encode("utf-8"))
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        generated = model.generate(prompt_tensor, max_new_tokens=100, temperature=0.7)
        gen_bytes = [int(x) for x in generated[0].cpu() if 0 <= x <= 255]
        gen_text = bytes(gen_bytes).decode("utf-8", errors="replace")
        gen_braille = "".join(chr(_BRAILLE_BASE + int(x)) for x in generated[0].cpu() if 0 <= x <= 255)
        log.info(f"  Sample output (SCL):     {gen_text[:80]}...")
        log.info(f"  Sample output (Braille): {gen_braille[:40]}...")
        log.info("")

        model.train()

    # --- Save final ---
    log.info("[4/4] Saving final model...")
    elapsed = time.time() - t0

    torch.save(model.state_dict(), f"{output_dir}/final_model.pt")

    # Save architecture config for loading
    config = {
        "vocab_size": VOCAB_SIZE,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "max_seq_len": max_seq_len,
        "model_size": model_size,
        "n_params": n_params,
    }
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    summary = {
        "model": f"ckm-braille-{model_size}",
        "vocab_size": VOCAB_SIZE,
        "n_params": n_params,
        "tokenizer": "NATIVE BRAILLE — 259 tokens, no BPE, no subwords",
        "epochs": epochs,
        "total_steps": global_step,
        "best_eval_loss": best_eval_loss,
        "final_train_loss": epoch_loss / max(1, epoch_steps),
        "training_minutes": round(elapsed / 60, 1),
        "architecture": "GPT-style (RMSNorm, RoPE, SwiGLU, weight-tied)",
        "trust_property": "output space == Braille character set (259 tokens)",
        "accessibility": "every token has a physical dot-pattern, readable by touch",
    }

    with open(f"{VOLUME_PATH}/training_summary_native.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("")
    log.info("=" * 70)
    log.info("  TRAINING COMPLETE")
    log.info("=" * 70)
    log.info(f"  Model:       ckm-braille-{model_size}")
    log.info(f"  Params:      {n_params:,}")
    log.info(f"  Vocab:       {VOCAB_SIZE} (native Braille)")
    log.info(f"  Best eval:   {best_eval_loss:.4f}")
    log.info(f"  Time:        {elapsed/60:.1f} min")
    log.info("")
    log.info("  This model has a 259-row embedding table.")
    log.info("  Each row is one Braille character.")
    log.info("  The lm_head has 259 columns.")
    log.info("  Each column is one Braille character.")
    log.info("  The model cannot produce non-Braille output.")
    log.info("  This is not a claim. It is linear algebra.")

    volume.commit()
    return summary


# ---------------------------------------------------------------------------
# Step 3: Validate — can the model produce valid SCL?
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    timeout=600,
)
def validate_model(model_size: str = "15m", max_seq_len: int = 512) -> dict:
    """Generate from the trained model and validate SCL output."""
    import json
    import logging
    import sys
    import torch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("braille.validate")

    # Load config
    config_path = f"{VOLUME_PATH}/checkpoints/config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Load model
    model = build_braille_model(
        n_layers=config["n_layers"], d_model=config["d_model"],
        n_heads=config["n_heads"], d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"], device="cuda",
    )
    model.load_state_dict(torch.load(f"{VOLUME_PATH}/checkpoints/best_model.pt"))
    model.eval()

    # Test prompts
    prompts = [
        "@hardware → state [cpu:",
        "@task → classify [category:",
        "@safety → deny [target:",
        "@init → boot [phase:",
        "@router → select [tier:",
        "@policy → mutate [",
        "@feedback → accumulated [model:",
    ]

    results = []
    passed = 0

    log.info("=" * 60)
    log.info("BRAILLE-NATIVE MODEL VALIDATION")
    log.info("=" * 60)

    for prompt_text in prompts:
        prompt_ids = [BOS_ID] + list(prompt_text.encode("utf-8"))
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")

        t0 = time.time()
        generated = model.generate(prompt_tensor, max_new_tokens=150, temperature=0.3)
        latency = (time.time() - t0) * 1000

        gen_bytes = [int(x) for x in generated[0].cpu() if 0 <= x <= 255]
        output = bytes(gen_bytes).decode("utf-8", errors="replace")
        braille = "".join(chr(_BRAILLE_BASE + int(x)) for x in generated[0].cpu() if 0 <= x <= 255)

        # Basic SCL validation
        is_valid = (
            "@" in output and
            "→" in output and
            "[" in output and
            "]" in output
        )

        # Check that ALL tokens are in the Braille range
        all_braille = all(
            (0 <= int(x) <= 255) or int(x) in (BOS_ID, EOS_ID, PAD_ID)
            for x in generated[0].cpu()
        )

        status = "✓" if (is_valid and all_braille) else "✗"
        if is_valid and all_braille:
            passed += 1

        log.info(f"  [{status}] ({latency:.0f}ms)")
        log.info(f"    SCL:     {output[:70]}...")
        log.info(f"    Braille: {braille[:35]}...")
        log.info(f"    Valid SCL: {is_valid}, All Braille: {all_braille}")

        results.append({
            "prompt": prompt_text,
            "output": output[:200],
            "braille": braille[:100],
            "valid_scl": is_valid,
            "all_braille_tokens": all_braille,
            "latency_ms": round(latency, 1),
        })

    log.info(f"\nResults: {passed}/{len(prompts)} passed")

    validation = {
        "passed": passed,
        "total": len(prompts),
        "pass_rate": passed / len(prompts),
        "results": results,
        "trust_verification": "ALL output tokens verified in Braille range (0-258)",
    }

    with open(f"{VOLUME_PATH}/validation_native.json", "w") as f:
        json.dump(validation, f, indent=2)

    volume.commit()
    return validation


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    skip_datagen: bool = False,
    skip_train: bool = False,
    skip_validate: bool = False,
    model_size: str = "15m",
    epochs: int = 10,
    n_sequences: int = 20000,
    batch_size: int = 64,
):
    """Run the Braille-native CKM training pipeline.

    Usage:
      modal run src/ckm/modal_train_braille_native.py
      modal run src/ckm/modal_train_braille_native.py --model-size 30m --epochs 20
    """
    print("=" * 70)
    print("  CORTEX KERNEL MODEL — Braille-Native Architecture")
    print()
    print("  This is a from-scratch transformer with a 259-token vocabulary.")
    print("  The embedding table IS the Braille character set.")
    print("  The output head IS the Braille character set.")
    print("  There is no BPE. There is no subword tokenizer.")
    print("  Every token the model can produce is a dot pattern.")
    print("=" * 70)
    print()

    if not skip_datagen:
        print("[1/3] Generating byte-level training data...")
        stats = generate_dataset.remote(n_sequences=n_sequences)
        print(f"  ✓ {stats['n_train']} train + {stats['n_eval']} eval sequences")
        print(f"  Vocab: {stats['vocab_size']} (native Braille bytes)")
        print()
    else:
        print("[1/3] Skipping datagen")
        print()

    if not skip_train:
        print(f"[2/3] Training ckm-braille-{model_size} on A100...")
        print(f"  {epochs} epochs, batch={batch_size}, from random init")
        summary = train_model.remote(
            model_size=model_size, epochs=epochs, batch_size=batch_size,
        )
        print(f"  ✓ {summary['total_steps']} steps in {summary['training_minutes']} min")
        print(f"  Best eval loss: {summary['best_eval_loss']:.4f}")
        print(f"  Params: {summary['n_params']:,}")
        print()
    else:
        print("[2/3] Skipping training")
        print()

    if not skip_validate:
        print("[3/3] Validating model output...")
        validation = validate_model.remote(model_size=model_size)
        print(f"  ✓ {validation['passed']}/{validation['total']} prompts produced valid SCL")
        print(f"  Trust check: {validation['trust_verification']}")
        print()
    else:
        print("[3/3] Skipping validation")
        print()

    print("=" * 70)
    print("  DONE")
    print("=" * 70)
    print()
    print("  What we built:")
    print(f"    - A {model_size}-parameter transformer")
    print(f"    - With a 259-token embedding table (256 Braille + 3 special)")
    print(f"    - Trained from random initialization on SCL byte sequences")
    print(f"    - The model's output space is architecturally constrained to Braille")
    print()
    print("  What this means:")
    print("    - No UNK tokens (every byte is representable)")
    print("    - No vocabulary drift (fixed forever at 259)")
    print("    - No hidden tokens (every token is a tactile dot pattern)")
    print("    - Accessibility is not a feature — it is the architecture")
    print()
    print("  Download checkpoint:")
    print("    modal volume get cortex-ckm-braille-native checkpoints/best_model.pt .")
