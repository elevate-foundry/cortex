"""
CKM From-Scratch Training — train a tiny transformer to speak SCL.

This is NOT fine-tuning. This trains a 1M-60M parameter transformer from
random initialization on the SCL grammar, using only the tokenized mmap
dataset and a simple causal language modeling objective.

Architecture: GPT-style decoder-only transformer
  - RMSNorm (pre-norm)
  - Rotary positional embeddings (RoPE)
  - Grouped-query attention (GQA for efficiency)
  - SwiGLU feed-forward
  - No dropout (tiny models, not overfitting on SCL)

The model learns to predict valid SCL state transitions, not prose.

Training loop:
  1. Load pre-tokenized mmap dataset
  2. Initialize model from ModelSpec
  3. Train with AdamW, cosine LR schedule
  4. Early stop when eval plateaus
  5. Save best checkpoint

Dependencies: torch only (no transformers, no HF, no accelerate)
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cortex.ckm.train_scratch")

# We import torch lazily to allow the module to be imported without torch
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

def build_model(
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    max_seq_len: int,
    device: str = "cpu",
    dtype=None,
):
    """Build a tiny GPT-style transformer for CKM."""
    torch = _get_torch()
    import torch.nn as nn

    if dtype is None:
        dtype = torch.float32

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
            self.max_len = max_len

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

    class CKMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.rotary = RotaryEmbedding(d_model // n_heads, max_len=max_seq_len)
            self.layers = nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_ff)
                for _ in range(n_layers)
            ])
            self.norm = RMSNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            # Weight tying
            self.lm_head.weight = self.tok_emb.weight

            # Causal mask
            mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
            self.register_buffer("causal_mask", mask)

            # Initialize weights
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
            logits = self.lm_head(x)
            return logits

        def count_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

    model = CKMModel().to(device)
    if dtype == torch.float16:
        model = model.half()
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    """Tracks training progress for early stopping and checkpointing."""
    epoch: int = 0
    step: int = 0
    best_eval_loss: float = float("inf")
    patience_counter: int = 0
    total_tokens: int = 0
    start_time: float = 0.0
    losses: list = None

    def __post_init__(self):
        if self.losses is None:
            self.losses = []


def train_ckm(
    dataset_dir: str,
    output_dir: str,
    device: str = "cpu",
    precision: str = "fp32",
    batch_size: int = 8,
    gradient_accumulation: int = 4,
    learning_rate: float = 3e-4,
    max_epochs: int = 5,
    time_budget_minutes: float = 10.0,
    early_stop_patience: int = 3,
    model_name: str = "ckm-5m",
    eval_every_steps: int = 50,
) -> dict:
    """
    Train a CKM model from scratch.

    Returns dict with:
      - model_path: path to saved model
      - final_loss: training loss
      - eval_loss: best eval loss
      - total_steps: steps completed
      - elapsed_minutes: training time
      - stopped_early: whether early stopping triggered
    """
    torch = _get_torch()
    import torch.nn as nn

    from .dataset import TokenizedDataset
    from .profile import MODEL_LADDER

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info("Loading tokenized dataset from %s", dataset_dir)
    dataset = TokenizedDataset.load(dataset_dir)
    logger.info("Dataset: %d sequences, seq_len=%d, vocab=%d",
                dataset.header.n_sequences, dataset.header.seq_len,
                dataset.header.vocab_size)

    # Split into train/eval (95/5)
    n_total = dataset.header.n_sequences
    n_eval = max(10, n_total // 20)
    n_train = n_total - n_eval
    train_indices = list(range(n_train))
    eval_indices = list(range(n_train, n_total))

    # Build model
    spec = MODEL_LADDER.get(model_name, MODEL_LADDER["ckm-5m"])
    logger.info("Building model: %s (%d params)", spec.name, spec.params)

    dtype = torch.float32
    if precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16

    model = build_model(
        vocab_size=dataset.header.vocab_size,
        d_model=spec.d_model,
        n_heads=spec.n_heads,
        n_layers=spec.n_layers,
        d_ff=spec.d_ff,
        max_seq_len=dataset.header.seq_len,
        device=device,
    )
    actual_params = model.count_params()
    logger.info("Model built: %d parameters", actual_params)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    # Cosine LR schedule
    total_steps_est = (n_train // batch_size) * max_epochs // gradient_accumulation
    warmup_steps = min(100, total_steps_est // 10)

    def get_lr(step):
        if step < warmup_steps:
            return learning_rate * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps_est - warmup_steps)
        return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD

    # Training state
    state = TrainingState(start_time=time.time())
    best_model_path = output_path / "best_model.pt"

    # AMP scaler for fp16
    use_amp = precision == "fp16" and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    logger.info("Starting training: %d epochs, batch=%d, accum=%d, lr=%.2e",
                max_epochs, batch_size, gradient_accumulation, learning_rate)
    logger.info("Time budget: %.1f minutes", time_budget_minutes)

    # Training loop
    model.train()
    accum_loss = 0.0
    accum_steps = 0
    import random

    for epoch in range(max_epochs):
        state.epoch = epoch
        random.shuffle(train_indices)

        for batch_start in range(0, n_train, batch_size):
            # Check time budget
            elapsed = (time.time() - state.start_time) / 60
            if elapsed >= time_budget_minutes:
                logger.info("Time budget exhausted (%.1f min)", elapsed)
                break

            # Get batch
            batch_idx = train_indices[batch_start:batch_start + batch_size]
            if len(batch_idx) < batch_size:
                continue
            batch_tokens = dataset.get_batch(batch_idx)
            x = torch.tensor(batch_tokens, dtype=torch.long, device=device)

            # Forward
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(x[:, :-1])
                    loss = criterion(
                        logits.reshape(-1, logits.size(-1)),
                        x[:, 1:].reshape(-1),
                    )
                    loss = loss / gradient_accumulation
                scaler.scale(loss).backward()
            else:
                logits = model(x[:, :-1])
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    x[:, 1:].reshape(-1),
                )
                loss = loss / gradient_accumulation
                loss.backward()

            accum_loss += loss.item()
            accum_steps += 1

            # Optimizer step
            if accum_steps >= gradient_accumulation:
                # LR schedule
                lr = get_lr(state.step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()
                state.step += 1
                state.losses.append(accum_loss)
                state.total_tokens += batch_size * dataset.header.seq_len * gradient_accumulation
                accum_loss = 0.0
                accum_steps = 0

                # Logging
                if state.step % 10 == 0:
                    avg_loss = sum(state.losses[-10:]) / min(10, len(state.losses))
                    tokens_per_sec = state.total_tokens / (time.time() - state.start_time)
                    logger.info(
                        "step=%d epoch=%d loss=%.4f lr=%.2e tok/s=%.0f",
                        state.step, epoch, avg_loss, lr, tokens_per_sec,
                    )

                # Eval
                if state.step % eval_every_steps == 0:
                    eval_loss = _evaluate(model, dataset, eval_indices, batch_size, device, criterion)
                    logger.info("  eval_loss=%.4f (best=%.4f)", eval_loss, state.best_eval_loss)

                    if eval_loss < state.best_eval_loss:
                        state.best_eval_loss = eval_loss
                        state.patience_counter = 0
                        # Save best model
                        torch.save(model.state_dict(), best_model_path)
                        logger.info("  → saved best model")
                    else:
                        state.patience_counter += 1
                        if state.patience_counter >= early_stop_patience:
                            logger.info("Early stopping at step %d", state.step)
                            break

                    model.train()

        # Check if we broke out due to time or early stop
        elapsed = (time.time() - state.start_time) / 60
        if elapsed >= time_budget_minutes or state.patience_counter >= early_stop_patience:
            break

    # Final save
    elapsed_minutes = (time.time() - state.start_time) / 60
    final_loss = sum(state.losses[-10:]) / max(1, min(10, len(state.losses)))

    # Save final model if no best was saved
    if not best_model_path.exists():
        torch.save(model.state_dict(), best_model_path)

    # Save training metadata
    metadata = {
        "model_name": model_name,
        "params": actual_params,
        "vocab_size": dataset.header.vocab_size,
        "seq_len": dataset.header.seq_len,
        "d_model": spec.d_model,
        "n_heads": spec.n_heads,
        "n_layers": spec.n_layers,
        "d_ff": spec.d_ff,
        "final_loss": final_loss,
        "best_eval_loss": state.best_eval_loss,
        "total_steps": state.step,
        "total_tokens": state.total_tokens,
        "elapsed_minutes": round(elapsed_minutes, 2),
        "stopped_early": state.patience_counter >= early_stop_patience,
        "device": device,
        "precision": precision,
        "dataset_dir": dataset_dir,
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    dataset.close()
    logger.info("Training complete: %d steps, %.2f min, loss=%.4f",
                state.step, elapsed_minutes, final_loss)

    return metadata


def _evaluate(model, dataset, eval_indices, batch_size, device, criterion) -> float:
    """Run evaluation on held-out data."""
    torch = _get_torch()
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for i in range(0, len(eval_indices), batch_size):
            batch_idx = eval_indices[i:i+batch_size]
            if len(batch_idx) < 2:
                continue
            batch_tokens = dataset.get_batch(batch_idx)
            x = torch.tensor(batch_tokens, dtype=torch.long, device=device)
            logits = model(x[:, :-1])
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                x[:, 1:].reshape(-1),
            )
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(1, n_batches)
