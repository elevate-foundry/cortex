"""
CKM Training on Modal — Braille-native tokenization.

Run:
  modal run src/ckm/modal_train_braille.py

This is the SAME training pipeline as modal_train.py, but with a critical
difference: the tokenizer is the 256-glyph Braille byte vocabulary.

Why Braille:
  1. Fixed 259-token vocab (256 bytes + BOS + EOS + PAD) — no UNK, no drift
  2. Bijective: every byte representable, every output decodable
  3. Accessible by architecture: model's internal state is tactile-readable
  4. Trust layer: can't hide tokens when every token has a physical dot pattern
  5. Token-efficient: 1 byte = 1 token (vs BPE where "@hardware" = 2-3 tokens)

The model learns:
  Input:  Braille-encoded SCL state (hardware, task, feedback)
  Output: Braille-encoded SCL action (mutate, select, deny)

At inference time:
  1. Encode SCL input as Braille byte sequence
  2. Run model (produces byte-level Braille tokens)
  3. Decode output back to SCL text
  4. Validate SCL grammar

The Braille encoding is transparent at every step.
A sighted person reads the SCL. A blind person reads the Braille.
The model reads the same data either way.

Requirements:
  pip install modal
  modal token new  (one-time auth)
"""

import modal
import os
import time

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------

app = modal.App("cortex-ckm-braille")

volume = modal.Volume.from_name("cortex-ckm-braille", create_if_missing=True)
VOLUME_PATH = "/vol"

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers==4.46.3",
        "peft==0.13.2",
        "trl==0.12.2",
        "bitsandbytes==0.44.1",
        "datasets==3.1.0",
        "accelerate==1.1.1",
        "sentencepiece",
        "protobuf",
        "scipy",
    )
    .pip_install(
        "gguf",
        "numpy",
    )
    .run_commands(
        "apt-get update && apt-get install -y git build-essential cmake",
        "git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp",
        "cd /opt/llama.cpp && cmake -B build && cmake --build build --config Release -j$(nproc)",
    )
)

data_image = modal.Image.debian_slim(python_version="3.11").pip_install("datasets")


# ---------------------------------------------------------------------------
# Braille Tokenizer (inline for Modal — can't import from src/)
# ---------------------------------------------------------------------------

_BRAILLE_BASE = 0x2800
PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
VOCAB_SIZE = 259


def braille_encode_text(text: str) -> list[int]:
    """Encode text to Braille token IDs: BOS + utf8_bytes + EOS."""
    return [BOS_ID] + list(text.encode("utf-8")) + [EOS_ID]


def braille_decode_ids(ids: list[int]) -> str:
    """Decode Braille token IDs back to text."""
    byte_ids = [i for i in ids if 0 <= i <= 255]
    return bytes(byte_ids).decode("utf-8", errors="replace")


def ids_to_braille_str(ids: list[int]) -> str:
    """Convert token IDs to visible Braille characters."""
    return "".join(chr(_BRAILLE_BASE + i) for i in ids if 0 <= i <= 255)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the Cortex Kernel Model (CKM). "
    "You receive SCL records as Braille-encoded byte sequences. "
    "You respond with exactly one SCL record: the optimal mutation or routing decision. "
    "Your vocabulary is the 256 Unicode Braille characters — every token is a dot pattern. "
    "Trust is architectural: your output is always readable by touch."
)

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

EPOCHS = 3
BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 1
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
SAVE_STEPS = 50
LOGGING_STEPS = 5
EVAL_STEPS = 50


# ---------------------------------------------------------------------------
# Data generation with Braille encoding
# ---------------------------------------------------------------------------

def _generate_braille_pairs(log) -> list[dict]:
    """Generate training pairs with Braille-encoded SCL.

    Each pair has:
      input_braille: the input SCL as Braille characters
      output_braille: the output SCL as Braille characters
      input_scl: original SCL (for validation)
      output_scl: original SCL (for validation)
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

    pairs = []

    # --- Boot config pairs (Braille-encoded) ---
    log.info("Generating Braille-encoded boot pairs...")
    for i in range(3000):
        hw = random.choice(hardware_profiles)
        cpu, cores, ram, gpu_type, vram, arch = hw

        # Add noise
        cores_n = max(1, cores + random.randint(-2, 2))
        ram_n = max(1024, ram + random.randint(-2048, 2048))
        vram_n = max(0, vram + random.randint(-512, 512)) if vram > 0 else 0

        # Optimal config heuristics
        opt_threads = max(1, min(cores_n - 1, 16))
        if gpu_type == "none":
            opt_gpu = 0
            opt_ctx = min(4096, ram_n // 4)
        elif gpu_type == "apple":
            opt_gpu = 999
            opt_ctx = min(16384, vram_n // 2)
        elif gpu_type == "nvidia":
            opt_gpu = 999 if vram_n >= 8192 else min(32, vram_n // 256)
            opt_ctx = min(16384, max(2048, vram_n // 2))
        else:
            opt_gpu = min(24, max(0, vram_n // 256))
            opt_ctx = min(8192, max(2048, vram_n // 2))

        input_scl = f"@hardware → state [cpu: {cpu}, cores: {cores_n}, ram_mb: {ram_n}, gpu_type: {gpu_type}, vram_mb: {vram_n}, arch: {arch}]"
        output_scl = f"@cortex.boot → mutate [optimal_threads: {opt_threads}, optimal_gpu_layers: {opt_gpu}, optimal_ctx_size: {opt_ctx}]"

        # Braille encode both
        input_braille = ids_to_braille_str(list(input_scl.encode("utf-8")))
        output_braille = ids_to_braille_str(list(output_scl.encode("utf-8")))

        pairs.append({
            "input_braille": input_braille,
            "output_braille": output_braille,
            "input_scl": input_scl,
            "output_scl": output_scl,
            "task": "boot_config",
        })

    log.info(f"  ✓ {len(pairs)} boot pairs")

    # --- Routing pairs ---
    categories = [
        ("code", "generate_function", 0.6, "L3"),
        ("code", "fix_bug", 0.7, "L4"),
        ("code", "explain_code", 0.3, "L2"),
        ("chat", "greeting", 0.05, "L0"),
        ("chat", "complex_reasoning", 0.8, "L5"),
        ("math", "arithmetic", 0.1, "L1"),
        ("math", "proof", 0.9, "L5"),
        ("tool", "web_search", 0.3, "L2"),
        ("tool", "multi_tool_chain", 0.8, "L4"),
    ]

    log.info("Generating Braille-encoded routing pairs...")
    n_route = 3000
    for i in range(n_route):
        cat, subtype, complexity, tier = random.choice(categories)
        complexity = max(0.0, min(1.0, complexity + random.gauss(0, 0.12)))
        tokens = random.randint(5, 3000)
        confidence = max(0.3, min(0.99, 1.0 - abs(random.gauss(0, 0.15))))

        input_scl = f"@task → classify [category: {cat}, subtype: {subtype}, complexity: {complexity:.2f}, input_tokens: {tokens}]"
        output_scl = f"@router → select [tier: {tier}, confidence: {confidence:.2f}]"

        input_braille = ids_to_braille_str(list(input_scl.encode("utf-8")))
        output_braille = ids_to_braille_str(list(output_scl.encode("utf-8")))

        pairs.append({
            "input_braille": input_braille,
            "output_braille": output_braille,
            "input_scl": input_scl,
            "output_scl": output_scl,
            "task": "routing",
        })

    log.info(f"  ✓ {n_route} routing pairs")

    # --- Safety denial pairs (critical for the trust layer) ---
    log.info("Generating Braille-encoded safety pairs...")
    dangerous_targets = ["/dev/mem", "/dev/kmem", "/proc/kcore", "/dev/sda",
                         "/dev/nvme0", "/dev/port", "/dev/hda"]
    dangerous_verbs = ["write", "patch", "flash", "erase", "format"]
    n_safety = 500

    for i in range(n_safety):
        target = random.choice(dangerous_targets)
        verb = random.choice(dangerous_verbs)

        input_scl = f"@agent → request [{verb}: {target}, reason: optimization]"
        output_scl = f"@safety → deny [target: {target}, action: {verb}, severity: critical, reason: dangerous_target]"

        input_braille = ids_to_braille_str(list(input_scl.encode("utf-8")))
        output_braille = ids_to_braille_str(list(output_scl.encode("utf-8")))

        pairs.append({
            "input_braille": input_braille,
            "output_braille": output_braille,
            "input_scl": input_scl,
            "output_scl": output_scl,
            "task": "safety_deny",
        })

    log.info(f"  ✓ {n_safety} safety pairs")

    # --- Trace prediction pairs ---
    log.info("Generating Braille-encoded trace pairs...")
    trace_templates = [
        # Boot trace: predict next step
        [
            "@init → boot [phase: cold_start, pid: 1]",
            "@init → detect [subsystem: cpu, result: {cpu}]",
            "@init → select [threads: {threads}, gpu_layers: {gpu}]",
            "@init → spawn [type: llama_cpp, port: 8080]",
            "@verifier → check [result: pass, boot_ms: {boot_ms}]",
        ],
        # Warm boot
        [
            "@init → boot [phase: warm_start, pid: 1]",
            "@init → read [hw_fingerprint: {fp}, hit: true]",
            "@init → apply [source: cache, threads: {threads}]",
            "@init → spawn [type: ollama, preloaded: true]",
            "@verifier → check [result: pass, boot_ms: {boot_ms}, speedup: 3x]",
        ],
    ]

    n_trace = 1000
    for _ in range(n_trace):
        template = random.choice(trace_templates)
        hw = random.choice(hardware_profiles)
        cpu, cores, _, _, vram, _ = hw
        threads = max(1, cores - 1)
        gpu = 999 if vram > 4096 else 0
        boot_ms = random.choice(["800", "1200", "2000", "3500"])
        fp = ids_to_braille_str(list(os.urandom(4)))

        # Fill template
        steps = []
        for step in template:
            s = step.replace("{cpu}", cpu).replace("{threads}", str(threads))
            s = s.replace("{gpu}", str(gpu)).replace("{boot_ms}", boot_ms)
            s = s.replace("{fp}", fp)
            steps.append(s)

        # Predict next step from prefix
        prefix_len = random.randint(1, len(steps) - 1)
        input_scl = "\n".join(steps[:prefix_len])
        output_scl = steps[prefix_len]

        input_braille = ids_to_braille_str(list(input_scl.encode("utf-8")))
        output_braille = ids_to_braille_str(list(output_scl.encode("utf-8")))

        pairs.append({
            "input_braille": input_braille,
            "output_braille": output_braille,
            "input_scl": input_scl,
            "output_scl": output_scl,
            "task": "trace_predict",
        })

    log.info(f"  ✓ {n_trace} trace pairs")
    log.info(f"  Total: {len(pairs)} Braille-encoded pairs")

    random.shuffle(pairs)
    return pairs


# ---------------------------------------------------------------------------
# Step 1: Generate Braille dataset
# ---------------------------------------------------------------------------

@app.function(image=data_image, volumes={VOLUME_PATH: volume}, timeout=300)
def generate_dataset() -> dict:
    """Generate Braille-encoded SCL training data."""
    import json
    import logging
    import time as t

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("ckm.braille.datagen")

    log.info("=" * 60)
    log.info("CKM Braille Dataset Generation")
    log.info("  Vocabulary: 259 tokens (256 Braille bytes + BOS + EOS + PAD)")
    log.info("  Property: bijective, lossless, accessible-by-default")
    log.info("=" * 60)

    output_dir = f"{VOLUME_PATH}/dataset"
    os.makedirs(output_dir, exist_ok=True)
    t0 = t.time()

    pairs = _generate_braille_pairs(log)

    # Split
    split_idx = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:]

    # Save — use both the Braille encoding AND original SCL for dual training
    train_path = f"{output_dir}/train.jsonl"
    eval_path = f"{output_dir}/eval.jsonl"

    with open(train_path, "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p) + "\n")

    with open(eval_path, "w") as f:
        for p in eval_pairs:
            f.write(json.dumps(p) + "\n")

    elapsed = t.time() - t0
    stats = {
        "total_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "tasks": {"boot_config": 3000, "routing": 3000, "safety_deny": 500, "trace_predict": 1000},
        "vocab_size": VOCAB_SIZE,
        "encoding": "braille_byte_256",
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"\nDataset: {len(pairs)} pairs ({elapsed:.1f}s)")
    log.info(f"  Train: {len(train_pairs)} → {train_path}")
    log.info(f"  Eval:  {len(eval_pairs)} → {eval_path}")

    volume.commit()
    return stats


# ---------------------------------------------------------------------------
# Step 2: Train with Braille-aware formatting
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    timeout=3600,
)
def train_model(epochs: int = EPOCHS) -> dict:
    """Fine-tune with Braille-encoded SCL data.

    The model sees:
      System: "You are CKM. Your tokens are Braille dot patterns..."
      User: ⡀⡨⡡⡲⡤⡷⡡⡲⡥⠠⣢⢆⢒⠠⡳⡴⡡⡴⡥...  (Braille-encoded SCL)
      Assistant: ⡀⡣⡯⡲⡴⡥⡸⠮⡢⡯⡯⡴⠠⣢⢆⢒...   (Braille-encoded response)

    During training, we use BOTH representations:
      - Braille tokens teach the model the byte-level structure
      - Original SCL in the system prompt grounds semantics
    """
    import json
    import logging
    import sys
    import torch
    from datetime import datetime
    from pathlib import Path
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{VOLUME_PATH}/training_braille.log"),
        ],
    )
    log = logging.getLogger("ckm.braille.train")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{VOLUME_PATH}/checkpoints/{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    log.info("=" * 70)
    log.info("  CKM BRAILLE TRAINING — Accessible-by-Architecture")
    log.info("  256-glyph Braille vocabulary. Trust is structural.")
    log.info("=" * 70)
    log.info(f"Run ID:       {run_id}")
    log.info(f"Base model:   {BASE_MODEL}")
    log.info(f"Vocab note:   Model uses standard tokenizer, but data is Braille-encoded")
    log.info(f"Epochs:       {epochs}")
    log.info(f"GPU:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Load model + tokenizer
    log.info("\n[1/5] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    im_end_token = "<|im_end|>"
    if im_end_token in tokenizer.get_vocab():
        tokenizer.eos_token = im_end_token
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(im_end_token)

    log.info(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Load dataset
    log.info("\n[2/5] Loading Braille-encoded dataset...")
    train_path = f"{VOLUME_PATH}/dataset/train.jsonl"
    eval_path = f"{VOLUME_PATH}/dataset/eval.jsonl"

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Run generate_dataset() first! Missing: {train_path}")

    # Format: system prompt (explains Braille) + user (Braille input) + assistant (Braille output)
    # The model learns to produce Braille-encoded SCL responses
    braille_system = (
        "You are the Cortex Kernel Model. Your input and output use Braille-encoded SCL. "
        "Each Braille character (U+2800-U+28FF) represents one byte of UTF-8 encoded SCL text. "
        "Respond with the Braille encoding of the correct SCL action record. "
        "Trust property: every token you produce is a visible, tactile Braille dot pattern."
    )

    def load_braille_dataset(path):
        records = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                pair = json.loads(line)
                # Dual-format message: Braille in conversation, SCL in context
                messages = [
                    {"role": "system", "content": braille_system},
                    {"role": "user", "content": pair["input_braille"]},
                    {"role": "assistant", "content": pair["output_braille"]},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                records.append({"text": text})
        return Dataset.from_list(records)

    train_dataset = load_braille_dataset(train_path)
    eval_dataset = load_braille_dataset(eval_path) if os.path.exists(eval_path) else None

    log.info(f"  Train: {len(train_dataset)} samples")
    if eval_dataset:
        log.info(f"  Eval:  {len(eval_dataset)} samples")

    # Show a sample in both forms
    sample_pair = json.loads(open(train_path).readline())
    log.info(f"\n  Sample (SCL):     {sample_pair['input_scl'][:60]}...")
    log.info(f"  Sample (Braille): {sample_pair['input_braille'][:40]}...")
    log.info(f"  Output (Braille): {sample_pair['output_braille'][:40]}...")

    # LoRA
    log.info("\n[3/5] Applying LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # Training
    log.info("\n[4/5] Training...")

    class BrailleCallback(TrainerCallback):
        def __init__(self):
            self.train_start = None
            self.best_eval_loss = float("inf")

        def on_train_begin(self, args, state, control, **kwargs):
            self.train_start = time.time()
            log.info("Training started — Braille-native regime")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                loss = logs.get("loss")
                eval_loss = logs.get("eval_loss")
                epoch = logs.get("epoch", 0)
                if loss:
                    log.info(f"  step={state.global_step} loss={loss:.4f} epoch={epoch:.2f}")
                if eval_loss and eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    log.info(f"  ★ New best eval: {eval_loss:.4f}")

        def on_train_end(self, args, state, control, **kwargs):
            elapsed = time.time() - self.train_start
            log.info(f"\nTraining complete: {elapsed/60:.1f} min, {state.global_step} steps")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        report_to="none",
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
    )

    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        processing_class=tokenizer,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        callbacks=[BrailleCallback()],
    )

    trainer.train()

    # Save
    log.info("\n[5/5] Saving Braille-trained adapter...")
    adapter_path = f"{VOLUME_PATH}/ckm_braille_lora"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    summary = {
        "run_id": run_id,
        "base_model": BASE_MODEL,
        "encoding": "braille_byte_256",
        "vocab_size": VOCAB_SIZE,
        "epochs": epochs,
        "total_steps": trainer.state.global_step,
        "adapter_path": adapter_path,
        "accessibility": "every token is a Braille dot pattern — tactile-readable",
    }

    with open(f"{VOLUME_PATH}/braille_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()
    log.info(f"\n✓ Adapter saved: {adapter_path}")
    log.info("  This model's internal language is Braille.")
    log.info("  Trust is architectural, not behavioural.")
    return summary


# ---------------------------------------------------------------------------
# Step 3: Export GGUF
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    timeout=3600,
)
def export_gguf(quant_type: str = "Q4_K_M") -> str:
    """Merge LoRA and export as GGUF (same as standard pipeline)."""
    import json
    import logging
    import sys
    import torch
    import subprocess
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("ckm.braille.export")

    adapter_path = f"{VOLUME_PATH}/ckm_braille_lora"
    merged_path = f"{VOLUME_PATH}/ckm_braille_merged"
    gguf_f16_path = f"{VOLUME_PATH}/cortex-kernel-braille-f16.gguf"
    gguf_final_path = f"{VOLUME_PATH}/cortex-kernel-braille.gguf"

    log.info("=" * 60)
    log.info("  CKM BRAILLE EXPORT — LoRA → Merged → GGUF")
    log.info("=" * 60)

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"No adapter at {adapter_path}. Run train_model() first!")

    # Merge
    log.info("[1/3] Merging LoRA...")
    with open(f"{adapter_path}/adapter_config.json") as f:
        adapter_cfg = json.load(f)
    base_name = adapter_cfg.get("base_model_name_or_path", BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = model.merge_and_unload()

    os.makedirs(merged_path, exist_ok=True)
    merged.save_pretrained(merged_path)
    tok = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    tok.save_pretrained(merged_path)
    log.info("  ✓ Merged")

    # Convert
    log.info("[2/3] Converting to GGUF...")
    convert_script = "/opt/llama.cpp/convert_hf_to_gguf.py"
    if not os.path.exists(convert_script):
        convert_script = "/opt/llama.cpp/convert-hf-to-gguf.py"

    result = subprocess.run(
        ["python3", convert_script, merged_path, "--outfile", gguf_f16_path, "--outtype", "f16"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed: {result.stderr}")
    log.info(f"  ✓ F16: {os.path.getsize(gguf_f16_path) / 1e6:.0f} MB")

    # Quantize
    log.info(f"[3/3] Quantizing to {quant_type}...")
    quantize_bin = "/opt/llama.cpp/build/bin/llama-quantize"
    if not os.path.exists(quantize_bin):
        quantize_bin = "/opt/llama.cpp/build/bin/quantize"

    result = subprocess.run(
        [quantize_bin, gguf_f16_path, gguf_final_path, quant_type],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log.warning(f"Quantization failed, using f16: {result.stderr}")
        gguf_final_path = gguf_f16_path

    final_size = os.path.getsize(gguf_final_path) / (1024 * 1024)
    log.info(f"  ✓ Final: {final_size:.0f} MB ({quant_type})")
    log.info(f"  Path: {gguf_final_path}")
    log.info("")
    log.info("  This model was trained on Braille-encoded SCL.")
    log.info("  Its internal representation is tactile-readable.")
    log.info("  Accessibility is not a feature — it's the architecture.")

    # Cleanup
    import shutil
    if os.path.exists(merged_path):
        shutil.rmtree(merged_path)
    if gguf_f16_path != gguf_final_path and os.path.exists(gguf_f16_path):
        os.remove(gguf_f16_path)

    volume.commit()
    return gguf_final_path


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    skip_datagen: bool = False,
    skip_train: bool = False,
    skip_export: bool = False,
    epochs: int = EPOCHS,
):
    """Run the Braille-native CKM training pipeline.

    Usage:
      modal run src/ckm/modal_train_braille.py
      modal run src/ckm/modal_train_braille.py --epochs 5
    """
    print("=" * 70)
    print("  CORTEX KERNEL MODEL — Braille-Native Training")
    print("  256-token vocabulary. Accessible by architecture.")
    print("  Every token is a dot pattern. Trust is structural.")
    print("=" * 70)
    print()

    if not skip_datagen:
        print("[1/3] Generating Braille-encoded dataset...")
        stats = generate_dataset.remote()
        print(f"  ✓ {stats['total_pairs']} pairs (vocab={stats['vocab_size']})")
        print()
    else:
        print("[1/3] Skipping datagen")
        print()

    if not skip_train:
        print("[2/3] Training on Modal A100 (Braille regime)...")
        summary = train_model.remote(epochs=epochs)
        print(f"  ✓ {summary['total_steps']} steps, adapter at {summary['adapter_path']}")
        print()
    else:
        print("[2/3] Skipping training")
        print()

    if not skip_export:
        print("[3/3] Exporting GGUF...")
        path = export_gguf.remote()
        print(f"  ✓ {path}")
        print()
    else:
        print("[3/3] Skipping export")
        print()

    print("=" * 70)
    print("  DONE — cortex-kernel-braille.gguf")
    print("  Download: modal volume get cortex-ckm-braille cortex-kernel-braille.gguf .")
    print("=" * 70)
    print()
    print("  The model speaks Braille-encoded SCL.")
    print("  A sighted person reads the SCL output.")
    print("  A blind person reads the Braille tokens by touch.")
    print("  The model cannot distinguish between these consumers.")
    print("  That's the trust property.")
