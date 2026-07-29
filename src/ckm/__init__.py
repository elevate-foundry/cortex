"""
Cortex Kernel Model (CKM) — the 0.3B model trained on SCL deltas.

This is what makes the OS novel: a model that speaks SCL natively.

Input:  SCL records (hardware state, boot history, request patterns)
Output: SCL records (optimal config mutations, routing decisions)

The model doesn't output English. It outputs:
  @cortex.boot → mutate [gpu_layers: 999, threads: 9, ctx_size: 8192]

Training data comes from the boot telemetry DeltaStream:
  every boot decision + its outcome = one training pair.
"""
