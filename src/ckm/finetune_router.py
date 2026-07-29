"""
CortexRouter-0.6B — LoRA fine-tuning pipeline.

Takes Qwen3-0.6B (a general-purpose 0.6B LLM) and fine-tunes it into a
routing specialist using LoRA adapters on routing telemetry data.

The trained model outputs structured routing decisions:
  Input:  user prompt (or first 256 tokens)
  Output: {"tier": "L3", "model": "qwen3:8b", "confidence": 0.85, "reason": "code task, moderate complexity"}

Training data sources:
  1. Live daemon telemetry (JSONL from /v1/usage audit log)
  2. Distillation traces (from src/ckm/distill_traces.py)
  3. Synthetic routing decisions (from frontier model labeling)

Pipeline:
  1. Load base model (Qwen3-0.6B from Ollama or HuggingFace)
  2. Apply LoRA adapters (rank=16, alpha=32, target: q_proj, v_proj, o_proj)
  3. Train on routing JSONL with causal LM objective
  4. Merge LoRA → full model
  5. Quantize to GGUF (Q4_K_M)
  6. Register with Ollama as cortex-router:0.6b

Dependencies: torch, peft, transformers, bitsandbytes (optional)
Alternative: unsloth for 2x training speed on consumer hardware

Usage:
  python -m src ckm finetune-router \\
    --base-model Qwen/Qwen3-0.6B \\
    --data data/routing_telemetry.jsonl \\
    --output models/cortex-router-0.6b \\
    --epochs 3 --lr 2e-4 --rank 16
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cortex.ckm.finetune_router")


# ---------------------------------------------------------------------------
# Training data format
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """You are CortexRouter, a routing model that decides which AI model and tier should handle a user request.

Given a user message, output a JSON routing decision:
{"tier": "L0-L7", "model": "model_id", "confidence": 0.0-1.0, "category": "category", "reason": "brief explanation"}

Tiers:
- L0-L2: Reflex (simple lookups, greetings, one-word answers) → local small models
- L3-L4: Standard (code, analysis, multi-step) → local medium models or cloud
- L5-L6: Expert (complex reasoning, long context) → cloud large models
- L7: Frontier (novel problems, research, safety-critical) → best available model

Categories: code, chat, classify, creative, math, science, vision, safety, unknown"""


@dataclass
class RoutingSample:
    """One training sample for the router."""
    prompt: str              # user's message (truncated to ~256 tokens)
    tier: str                # L0-L7
    model: str               # model_id that handled it
    confidence: float        # routing confidence
    category: str            # task category
    reason: str = ""         # why this routing (optional)
    latency_ms: float = 0.0  # actual latency (for quality signal)
    cost_usd: float = 0.0    # actual cost
    quality: float = 0.0     # 0-1 quality score (from feedback or heuristic)

    def to_training_pair(self) -> dict:
        """Convert to input/output pair for fine-tuning."""
        return {
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": json.dumps({
                    "tier": self.tier,
                    "model": self.model,
                    "confidence": round(self.confidence, 2),
                    "category": self.category,
                    "reason": self.reason or f"{self.category} task → {self.tier}",
                })},
            ]
        }

    def to_jsonl(self) -> str:
        """Serialize to JSONL line."""
        return json.dumps(self.to_training_pair())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_routing_samples(path: Path) -> list[RoutingSample]:
    """Load routing samples from JSONL file.
    
    Supports two formats:
    1. Raw telemetry: {"prompt": "...", "tier": "L3", "model": "...", ...}
    2. Chat format: {"messages": [...]} (already formatted)
    """
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            
            if "messages" in data:
                # Already in chat format — extract fields
                assistant_msg = ""
                user_msg = ""
                for msg in data["messages"]:
                    if msg["role"] == "user":
                        user_msg = msg["content"]
                    elif msg["role"] == "assistant":
                        assistant_msg = msg["content"]
                try:
                    routing = json.loads(assistant_msg)
                    samples.append(RoutingSample(
                        prompt=user_msg,
                        tier=routing.get("tier", "L2"),
                        model=routing.get("model", ""),
                        confidence=routing.get("confidence", 0.5),
                        category=routing.get("category", "unknown"),
                        reason=routing.get("reason", ""),
                    ))
                except (json.JSONDecodeError, KeyError):
                    continue
            else:
                # Raw telemetry format
                samples.append(RoutingSample(
                    prompt=data.get("prompt", ""),
                    tier=data.get("tier", data.get("routed_tier", "L2")),
                    model=data.get("model", data.get("actual_model", "")),
                    confidence=data.get("confidence", 0.5),
                    category=data.get("category", "unknown"),
                    reason=data.get("reason", ""),
                    latency_ms=data.get("latency_ms", 0.0),
                    cost_usd=data.get("cost_usd", 0.0),
                    quality=data.get("quality", 0.0),
                ))
    
    logger.info("Loaded %d routing samples from %s", len(samples), path)
    return samples


def export_audit_to_training_data(memory_db_path: Path, output_path: Path) -> int:
    """Export daemon audit log to router training JSONL.
    
    Reads the SQLite audit log and converts each request into a routing sample.
    """
    import sqlite3
    
    conn = sqlite3.connect(str(memory_db_path))
    conn.row_factory = sqlite3.Row
    
    cursor = conn.execute("""
        SELECT routed_tier, actual_model, category, confidence,
               tokens_prompt, tokens_completion, latency_ms, cost_usd
        FROM audit_log
        WHERE routed_tier != '' AND actual_model != ''
        ORDER BY created_at DESC
    """)
    
    count = 0
    with open(output_path, "w") as f:
        for row in cursor:
            # We don't have the original prompt in audit_log,
            # but we have all routing metadata. This produces "label-only"
            # samples that need to be joined with the actual prompts from
            # the thread/message history.
            sample = RoutingSample(
                prompt="",  # needs to be filled from message history
                tier=row["routed_tier"],
                model=row["actual_model"],
                confidence=row["confidence"],
                category=row["category"],
                latency_ms=row["latency_ms"],
                cost_usd=row["cost_usd"] if row["cost_usd"] else 0.0,
            )
            if sample.prompt or sample.tier:
                f.write(sample.to_jsonl() + "\n")
                count += 1
    
    conn.close()
    logger.info("Exported %d training samples to %s", count, output_path)
    return count


# ---------------------------------------------------------------------------
# LoRA Fine-tuning
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    base_model: str = "Qwen/Qwen3-0.6B"       # HuggingFace model ID
    rank: int = 16                              # LoRA rank
    alpha: int = 32                             # LoRA alpha (scaling)
    dropout: float = 0.05                       # LoRA dropout
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "o_proj",           # attention projections
    ])
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 4              # effective batch = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    max_seq_len: int = 512
    output_dir: str = "models/cortex-router-0.6b"
    use_4bit: bool = True                       # QLoRA (saves VRAM)
    device: str = "auto"                        # auto, cuda, mps, cpu


def finetune_router(
    data_path: Path,
    config: Optional[LoRAConfig] = None,
) -> Path:
    """
    Fine-tune Qwen3-0.6B into CortexRouter using LoRA.
    
    Returns path to the merged model directory.
    """
    if config is None:
        config = LoRAConfig()
    
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # --- Check dependencies ---
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
        )
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
    except ImportError as e:
        logger.error(
            "Fine-tuning requires: pip install torch transformers peft\n"
            "For QLoRA: pip install bitsandbytes\n"
            "For 2x speed: pip install unsloth\n"
            f"Missing: {e}"
        )
        raise

    t0 = time.time()
    logger.info("=== CortexRouter-0.6B Fine-tuning ===")
    logger.info("Base model: %s", config.base_model)
    logger.info("Data: %s", data_path)
    logger.info("Output: %s", output_path)
    
    # --- Load training data ---
    samples = load_routing_samples(data_path)
    if len(samples) < 100:
        logger.warning("Only %d samples — recommend at least 1000 for good results", len(samples))
    
    # Convert to chat format
    training_data = [s.to_training_pair() for s in samples]
    
    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- Load model ---
    model_kwargs = {"trust_remote_code": True}
    
    if config.use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except ImportError:
            logger.warning("bitsandbytes not available — loading in full precision")
            config.use_4bit = False
    
    if config.device == "auto":
        if torch.cuda.is_available():
            device_map = "auto"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_map = {"": "mps"}
            config.use_4bit = False  # bitsandbytes doesn't support MPS
        else:
            device_map = {"": "cpu"}
            config.use_4bit = False
    else:
        device_map = {"": config.device}
    
    model_kwargs["device_map"] = device_map
    
    logger.info("Loading base model (%s)...", "4bit" if config.use_4bit else "fp16/fp32")
    model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    
    # --- Apply LoRA ---
    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=config.target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # --- Tokenize dataset ---
    def tokenize_chat(example):
        """Tokenize a chat-format training example."""
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=config.max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        tokens["labels"] = tokens["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in tokens.items()}
    
    # Build dataset
    from torch.utils.data import Dataset as TorchDataset
    
    class RouterDataset(TorchDataset):
        def __init__(self, data, tokenize_fn):
            self.data = data
            self.tokenize_fn = tokenize_fn
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.tokenize_fn(self.data[idx])
    
    dataset = RouterDataset(training_data, tokenize_chat)
    
    # Split 90/10
    split = int(len(dataset) * 0.9)
    train_dataset = torch.utils.data.Subset(dataset, range(split))
    eval_dataset = torch.utils.data.Subset(dataset, range(split, len(dataset)))
    
    logger.info("Training: %d samples, Eval: %d samples", len(train_dataset), len(eval_dataset))
    
    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        report_to="none",
        remove_unused_columns=False,
    )
    
    # --- Train ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    logger.info("Starting LoRA fine-tuning...")
    trainer.train()
    
    # --- Save LoRA adapter ---
    adapter_path = output_path / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info("LoRA adapter saved: %s", adapter_path)
    
    # --- Merge LoRA into base model ---
    logger.info("Merging LoRA into base model...")
    from peft import PeftModel
    
    # Reload base in full precision for merge
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=torch.float16,
    )
    merged_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged_model = merged_model.merge_and_unload()
    
    merged_path = output_path / "merged"
    merged_model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))
    
    elapsed = time.time() - t0
    logger.info("=== Fine-tuning complete (%.1f min) ===", elapsed / 60)
    logger.info("Merged model: %s", merged_path)
    
    # Save training metadata
    meta = {
        "base_model": config.base_model,
        "training_samples": len(samples),
        "epochs": config.epochs,
        "lora_rank": config.rank,
        "lora_alpha": config.alpha,
        "training_time_seconds": elapsed,
        "merged_path": str(merged_path),
    }
    (output_path / "training_meta.json").write_text(json.dumps(meta, indent=2))
    
    return merged_path


# ---------------------------------------------------------------------------
# GGUF Export + Ollama Registration
# ---------------------------------------------------------------------------

def export_to_gguf(
    model_path: Path,
    output_path: Optional[Path] = None,
    quantization: str = "Q4_K_M",
) -> Path:
    """
    Export a HuggingFace model to GGUF format for Ollama.
    
    Requires llama.cpp's convert script or the `gguf` package.
    """
    if output_path is None:
        output_path = model_path.parent / f"cortex-router-0.6b-{quantization.lower()}.gguf"
    
    import subprocess
    
    # Try llama.cpp conversion first
    convert_script = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # Try common alternative locations
        for alt in [
            Path("/usr/local/bin/convert_hf_to_gguf.py"),
            Path.home() / "Projects" / "llama.cpp" / "convert_hf_to_gguf.py",
        ]:
            if alt.exists():
                convert_script = alt
                break
    
    if convert_script.exists():
        logger.info("Converting to GGUF using llama.cpp...")
        
        # Step 1: Convert to f16 GGUF
        f16_path = model_path.parent / "cortex-router-0.6b-f16.gguf"
        subprocess.run([
            "python3", str(convert_script),
            str(model_path),
            "--outtype", "f16",
            "--outfile", str(f16_path),
        ], check=True)
        
        # Step 2: Quantize
        quantize_bin = convert_script.parent / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            quantize_bin = convert_script.parent / "llama-quantize"
        
        if quantize_bin.exists():
            subprocess.run([
                str(quantize_bin),
                str(f16_path),
                str(output_path),
                quantization,
            ], check=True)
            f16_path.unlink()  # Clean up f16
        else:
            # Can't quantize, just use f16
            f16_path.rename(output_path)
            logger.warning("llama-quantize not found — using f16 (larger file)")
    else:
        logger.error(
            "llama.cpp not found. Install it:\n"
            "  git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp\n"
            "  cd ~/llama.cpp && make -j\n"
            "Then re-run this export."
        )
        raise FileNotFoundError("llama.cpp convert script not found")
    
    logger.info("GGUF exported: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path


def create_ollama_modelfile(gguf_path: Path, output_path: Optional[Path] = None) -> Path:
    """Generate an Ollama Modelfile for the router model."""
    if output_path is None:
        output_path = gguf_path.parent / "Modelfile"
    
    content = (
        f"FROM {gguf_path.name}\n\n"
        f"SYSTEM \"{ROUTER_SYSTEM_PROMPT}\"\n\n"
        "PARAMETER temperature 0\n"
        "PARAMETER num_predict 64\n"
    )

    output_path.write_text(content)
    logger.info("Modelfile written: %s", output_path)
    return output_path


def register_with_ollama(gguf_path: Path, model_name: str = "cortex-router:0.6b") -> bool:
    """Register the GGUF model with local Ollama."""
    import subprocess

    modelfile_path = create_ollama_modelfile(gguf_path)

    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            cwd=str(gguf_path.parent),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            logger.info("Registered with Ollama: %s", model_name)
            return True
        else:
            logger.error("Ollama create failed: %s", result.stderr)
            return False
    except FileNotFoundError:
        logger.error("Ollama not found — install from https://ollama.com")
        return False
    except subprocess.TimeoutExpired:
        logger.error("Ollama create timed out")
        return False


# ---------------------------------------------------------------------------
# Full pipeline: data → train → export → register
# ---------------------------------------------------------------------------

def full_pipeline(
    data_path: Path,
    config: Optional[LoRAConfig] = None,
    quantization: str = "Q4_K_M",
    register: bool = True,
) -> dict:
    """
    Run the full CortexRouter training pipeline.

    1. Fine-tune Qwen3-0.6B with LoRA on routing data
    2. Merge LoRA into base model
    3. Export to GGUF
    4. Register with Ollama

    Returns metadata dict with paths and stats.
    """
    if config is None:
        config = LoRAConfig()

    # Step 1+2: Fine-tune and merge
    merged_path = finetune_router(data_path, config)

    # Step 3: Export to GGUF
    gguf_path = export_to_gguf(merged_path, quantization=quantization)

    # Step 4: Register with Ollama
    registered = False
    if register:
        registered = register_with_ollama(gguf_path)

    return {
        "merged_model": str(merged_path),
        "gguf_path": str(gguf_path),
        "quantization": quantization,
        "registered": registered,
        "model_name": "cortex-router:0.6b",
    }
