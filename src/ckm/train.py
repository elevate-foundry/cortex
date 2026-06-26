"""
CKM Training Pipeline — Fine-tune a 0.3B model to speak SCL.

The Cortex Kernel Model (CKM) is a small LLM that:
  - Takes SCL records as input
  - Produces SCL records as output
  - Runs at boot time in <100ms on CPU
  - Replaces all heuristics in the boot/routing system

Training flow:
  1. Generate dataset (data_generator.py → JSONL)
  2. Select base model (Qwen2.5-0.3B or SmolLM2-135M)
  3. Fine-tune with LoRA (4-bit quantized, 1 epoch)
  4. Export to GGUF (for llama.cpp inference at boot)
  5. Package into initramfs

The model is tiny enough to:
  - Fit in L1 cache on most CPUs
  - Run inference in <50ms without GPU
  - Live in the initramfs (< 200MB quantized)

Usage:
  python -m src.ckm.train --dataset /path/to/ckm_training.jsonl --output /path/to/ckm.gguf
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cortex.ckm.train")


# ---------------------------------------------------------------------------
# Base model selection
# ---------------------------------------------------------------------------

BASE_MODELS = {
    "qwen2.5-0.3b": {
        "hf_id": "Qwen/Qwen2.5-0.3B",
        "params": "0.3B",
        "context": 32768,
        "good_for": "best quality at 0.3B size",
    },
    "smollm2-135m": {
        "hf_id": "HuggingFaceTB/SmolLM2-135M",
        "params": "135M",
        "context": 8192,
        "good_for": "fastest inference, fits in 100MB GGUF",
    },
    "smollm2-360m": {
        "hf_id": "HuggingFaceTB/SmolLM2-360M",
        "params": "360M",
        "context": 8192,
        "good_for": "balanced speed/quality",
    },
}

DEFAULT_BASE = "qwen2.5-0.3b"


# ---------------------------------------------------------------------------
# Dataset formatting for fine-tuning
# ---------------------------------------------------------------------------

def format_for_sft(jsonl_path: Path) -> list[dict]:
    """Convert JSONL training data to SFT (Supervised Fine-Tuning) format.

    Each pair becomes a conversation:
      system: "You are the Cortex Kernel Model. Respond in SCL only."
      user: <input_scl>
      assistant: <output_scl>
    """
    SYSTEM_PROMPT = (
        "You are the Cortex Kernel Model (CKM). "
        "You receive SCL records describing hardware state or request classification. "
        "You respond with exactly one SCL record: the optimal mutation or routing decision. "
        "Never output anything except valid SCL. "
        "Format: @anchor → verb [key: value, key: value]"
    )

    conversations = []
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            pair = json.loads(line)
            conv = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": pair["input"]},
                    {"role": "assistant", "content": pair["output"]},
                ],
                "quality": pair.get("quality", 1.0),
            }
            conversations.append(conv)

    return conversations


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

def get_training_config(
    base_model: str = DEFAULT_BASE,
    output_dir: str = "./ckm_output",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_seq_length: int = 512,
) -> dict:
    """Generate training configuration for the CKM fine-tune.

    Uses LoRA (low-rank adaptation) for efficient fine-tuning:
      - Only trains ~2% of parameters
      - Fits in 8GB VRAM (or CPU with patience)
      - Completes in <1 hour on consumer GPU
    """
    model_info = BASE_MODELS.get(base_model, BASE_MODELS[DEFAULT_BASE])

    return {
        "model": {
            "base": model_info["hf_id"],
            "params": model_info["params"],
        },
        "lora": {
            "r": lora_r,
            "alpha": lora_alpha,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
            "dropout": 0.05,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": 4,
            "learning_rate": learning_rate,
            "lr_scheduler": "cosine",
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
            "max_seq_length": max_seq_length,
            "fp16": True,
        },
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
        },
        "output": {
            "dir": output_dir,
            "save_steps": 100,
            "logging_steps": 10,
        },
    }


# ---------------------------------------------------------------------------
# Training script (generates the actual training code)
# ---------------------------------------------------------------------------

def generate_training_script(
    config: dict,
    dataset_path: str,
    output_path: str = "./train_ckm.py",
) -> str:
    """Generate a standalone Python training script.

    This can be run on any machine with a GPU and the right packages.
    It produces a LoRA adapter that gets merged and exported to GGUF.
    """
    script = f'''#!/usr/bin/env python3
"""
Auto-generated CKM training script.
Run: pip install transformers peft trl bitsandbytes datasets
Then: python train_ckm.py
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Configuration
BASE_MODEL = "{config['model']['base']}"
DATASET_PATH = "{dataset_path}"
OUTPUT_DIR = "{config['output']['dir']}"
MAX_SEQ_LENGTH = {config['training']['max_seq_length']}

SYSTEM_PROMPT = (
    "You are the Cortex Kernel Model (CKM). "
    "You receive SCL records describing hardware state or request classification. "
    "You respond with exactly one SCL record: the optimal mutation or routing decision. "
    "Never output anything except valid SCL. "
    "Format: @anchor → verb [key: value, key: value]"
)


def load_dataset():
    """Load JSONL training data into HuggingFace Dataset."""
    conversations = []
    with open(DATASET_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            pair = json.loads(line)
            # Format as chat template
            text = (
                f"<|im_start|>system\\n{{SYSTEM_PROMPT}}<|im_end|>\\n"
                f"<|im_start|>user\\n{{pair['input']}}<|im_end|>\\n"
                f"<|im_start|>assistant\\n{{pair['output']}}<|im_end|>"
            )
            conversations.append({{"text": text, "quality": pair.get("quality", 1.0)}})
    return Dataset.from_list(conversations)


def main():
    print(f"Loading base model: {{BASE_MODEL}}")
    print(f"Dataset: {{DATASET_PATH}}")
    print(f"Output: {{OUTPUT_DIR}}")

    # Quantization config (4-bit for efficient training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    lora_config = LoraConfig(
        r={config['lora']['r']},
        lora_alpha={config['lora']['alpha']},
        target_modules={config['lora']['target_modules']},
        lora_dropout={config['lora']['dropout']},
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = load_dataset()
    print(f"Training samples: {{len(dataset)}}")

    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs={config['training']['epochs']},
        per_device_train_batch_size={config['training']['batch_size']},
        gradient_accumulation_steps={config['training']['gradient_accumulation_steps']},
        learning_rate={config['training']['learning_rate']},
        lr_scheduler_type="{config['training']['lr_scheduler']}",
        warmup_ratio={config['training']['warmup_ratio']},
        weight_decay={config['training']['weight_decay']},
        fp16={config['training']['fp16']},
        logging_steps={config['output']['logging_steps']},
        save_steps={config['output']['save_steps']},
        save_total_limit=2,
        report_to="none",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
    )

    print("Starting training...")
    trainer.train()

    # Save LoRA adapter
    adapter_path = Path(OUTPUT_DIR) / "ckm_lora"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"LoRA adapter saved to: {{adapter_path}}")
    print()
    print("Next steps:")
    print(f"  1. Merge: python -m src.ckm.export --adapter {{adapter_path}} --output ckm.gguf")
    print(f"  2. Test:  python -m src.ckm.inference --model ckm.gguf --input '@hardware → state [...]'")
    print(f"  3. Deploy: copy ckm.gguf to /mnt/cortex/models/")


if __name__ == "__main__":
    main()
'''
    Path(output_path).write_text(script)
    return script


# ---------------------------------------------------------------------------
# GGUF export
# ---------------------------------------------------------------------------

def generate_export_script(output_path: str = "./export_ckm.sh") -> str:
    """Generate shell script to merge LoRA + quantize to GGUF.

    GGUF is what llama.cpp uses — this makes the model bootable.
    """
    script = '''#!/bin/bash
# Export CKM to GGUF for boot-time inference
# Requires: llama.cpp (for convert/quantize), Python (for merge)

set -e

ADAPTER_DIR="${1:-./ckm_output/ckm_lora}"
OUTPUT="${2:-./ckm.gguf}"
QUANT="${3:-Q4_K_M}"  # Good balance of size vs quality

echo "=== CKM Export Pipeline ==="
echo "  Adapter: $ADAPTER_DIR"
echo "  Output:  $OUTPUT"
echo "  Quant:   $QUANT"
echo ""

# Step 1: Merge LoRA into base model
echo "[1/3] Merging LoRA adapter..."
python3 -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    '$(cat $ADAPTER_DIR/adapter_config.json | python3 -c \"import json,sys; print(json.load(sys.stdin)[\\\"base_model_name_or_path\\\"])\")',
    torch_dtype=torch.float16,
    device_map='cpu',
)
model = PeftModel.from_pretrained(base, '$ADAPTER_DIR')
merged = model.merge_and_unload()
merged.save_pretrained('./ckm_merged')
AutoTokenizer.from_pretrained('$ADAPTER_DIR').save_pretrained('./ckm_merged')
print('Merged model saved to ./ckm_merged')
"

# Step 2: Convert to GGUF
echo "[2/3] Converting to GGUF..."
python3 llama.cpp/convert_hf_to_gguf.py ./ckm_merged --outfile ./ckm_f16.gguf --outtype f16

# Step 3: Quantize
echo "[3/3] Quantizing to $QUANT..."
llama.cpp/build/bin/llama-quantize ./ckm_f16.gguf "$OUTPUT" "$QUANT"

# Summary
SIZE=$(du -h "$OUTPUT" | cut -f1)
echo ""
echo "=== Done ==="
echo "  Model: $OUTPUT ($SIZE)"
echo "  Deploy: cp $OUTPUT /mnt/cortex/models/cortex-kernel.gguf"
echo "  Test:   llama-cli -m $OUTPUT -p '@hardware → state [cpu: Apple M1, ram_mb: 16384, gpu_type: apple]'"

# Cleanup
rm -rf ./ckm_merged ./ckm_f16.gguf
'''
    Path(output_path).write_text(script)
    return script


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CKM Training Pipeline")
    sub = parser.add_subparsers(dest="command")

    # Generate dataset
    gen = sub.add_parser("generate", help="Generate training dataset")
    gen.add_argument("--output", default="./ckm_training.jsonl")
    gen.add_argument("--boot-count", type=int, default=1000)
    gen.add_argument("--route-count", type=int, default=2000)
    gen.add_argument("--boot-telemetry", help="Path to boot_deltas.scl for real data")

    # Train
    train = sub.add_parser("train", help="Generate training script")
    train.add_argument("--dataset", required=True)
    train.add_argument("--base-model", default=DEFAULT_BASE, choices=BASE_MODELS.keys())
    train.add_argument("--output-dir", default="./ckm_output")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--script-output", default="./train_ckm.py")

    # Export
    export = sub.add_parser("export", help="Generate GGUF export script")
    export.add_argument("--output", default="./export_ckm.sh")

    args = parser.parse_args()

    if args.command == "generate":
        from .data_generator import CKMDataset, generate_boot_pairs
        from ..scl.delta import DeltaStream, Delta
        from ..scl.parser import parse_document

        dataset = CKMDataset(output_dir=str(Path(args.output).parent))

        # Add synthetic data
        added = dataset.add_synthetic(boot_count=args.boot_count, route_count=args.route_count)
        print(f"Generated {added} synthetic training pairs")

        # Add real boot telemetry if available
        if args.boot_telemetry and Path(args.boot_telemetry).exists():
            text = Path(args.boot_telemetry).read_text()
            doc = parse_document(text)
            stream = DeltaStream()
            for record in doc.records:
                if record.relation.verb == "mutate":
                    delta = Delta.from_scl(record)
                    stream.append(delta)
            boot_added = dataset.add_boot_telemetry(stream)
            print(f"Added {boot_added} real boot telemetry pairs")

        path = dataset.save(filename=Path(args.output).name)
        print(f"\nDataset saved: {path}")
        print(json.dumps(dataset.stats(), indent=2))

    elif args.command == "train":
        config = get_training_config(
            base_model=args.base_model,
            output_dir=args.output_dir,
            epochs=args.epochs,
        )
        generate_training_script(config, args.dataset, args.script_output)
        print(f"Training script generated: {args.script_output}")
        print(f"  Base model: {config['model']['base']} ({config['model']['params']})")
        print(f"  Run: python {args.script_output}")

    elif args.command == "export":
        generate_export_script(args.output)
        print(f"Export script generated: {args.output}")
        print(f"  Run: bash {args.output} <adapter_dir> <output.gguf>")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
