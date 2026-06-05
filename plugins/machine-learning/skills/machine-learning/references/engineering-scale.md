# Engineering & Scale

Making training and inference fast, memory-feasible, numerically stable, and correct at scale. Architectures are
in the model references; this is the systems layer. Current to 2026 (PyTorch 2.10–2.12, FSDP2, `torch.compile`).

---

## 1. Frameworks and their idioms

- **PyTorch** (research default). Eager by default; wrap the model in **`torch.compile`** for 20–50% speedups
  (graph capture + fusion + autotuned kernels) — default-on for serious training as of 2.x. Mind: first-call
  compile latency, graph breaks (avoid data-dependent Python control flow in hot paths), and recompiles on
  changing shapes (pad/bucket sequence lengths, or use dynamic shapes). Use `torch.profiler` to find
  bottlenecks.
- **JAX + Flax NNX / Equinox** (TPU & large-scale / functional). Pure functions + `jit`/`grad`/`vmap`/`pmap`/
  `shard_map` compose beautifully; **NNX** is the recommended Flax API (Pythonic, mutable-feeling) over Linen.
  Excellent for research that needs per-example gradients, ensembles via `vmap`, or first-class parallelism;
  steeper functional-purity learning curve. MaxText/Levanter for foundation models.
- **Choosing:** PyTorch by default (ecosystem, `transformers`, community); JAX for TPUs, research-grade
  parallelism, or heavy functional transforms. Don't switch frameworks mid-project without a real reason.

## 2. Memory: where it goes and how to fit more

Training memory = **parameters + gradients + optimizer state + activations + workspace**. For AdamW in fp32,
params+grads+optimizer ≈ **16 bytes/param** (2 momentvectors + master copy); activations scale with batch ×
sequence × width × depth and often dominate. Levers, cheapest first:
- **Mixed precision (bf16)** — ~½ the memory and ~2× throughput on modern GPUs; **bf16 > fp16** (same exponent
  range as fp32 → no loss-scaling needed, far fewer NaNs). Use fp16 only on hardware without bf16, with a
  `GradScaler`. Keep a master fp32 copy of weights / norm accumulations.
- **Gradient accumulation** — simulate a large batch in micro-batches (more steps, same effective batch). Free
  except wall-clock.
- **Gradient (activation) checkpointing** — don't store activations; recompute them in the backward pass. Trades
  ~30% extra compute for large activation-memory savings; essential for big models / long sequences.
- **Optimizer-state sharding / 8-bit optimizers** — `bitsandbytes` 8-bit Adam, Adafactor (no second moment),
  or shard optimizer state across GPUs (ZeRO/FSDP §4).
- **Efficient attention** — FlashAttention / SDPA avoids materializing the $L\times L$ attention matrix (memory
  $O(L)$ not $O(L^2)$); use for any long sequence.
- **Quantization for fine-tuning** — QLoRA (4-bit frozen base + LoRA adapters) fine-tunes large models on one
  GPU (see [representation-learning.md](representation-learning.md)).
- **Offloading** — CPU/NVMe offload of params/optimizer (ZeRO-Offload) when you're truly memory-bound; slow,
  last resort.

## 3. Speed: throughput and the input pipeline

- **The GPU should never wait on data.** Profile utilization; if it's low, the bottleneck is usually the
  **dataloader**, not the model. Fix with enough `num_workers`, `pin_memory=True`, `prefetch_factor`,
  `non_blocking=True` transfers, efficient decoding, and pre-tokenized/sharded data (WebDataset, Mosaic
  Streaming, Parquet/Arrow). Overlap H2D copy with compute.
- **`torch.compile`** for kernel fusion + autotuning; **fused optimizers** (`fused=True` AdamW); **channels-last**
  memory format for conv nets; **TF32** matmuls on Ampere+ (`torch.set_float32_matmul_precision('high')`).
- **Sequence packing** (concatenate short sequences to fill the context, with proper masking via `varlen_attn`)
  removes padding waste in LM training.
- **Profile, don't guess:** `torch.profiler` / Nsight / the JAX profiler. Optimize the measured bottleneck.
  Compute-bound, memory-bandwidth-bound, and comms-bound problems need different fixes; the **roofline** tells
  you which you're in.

## 4. Distributed training (multi-GPU / multi-node)

Pick the minimal parallelism that fits; complexity compounds.
- **Data Parallel (DDP / DistributedDataParallel)** — replicate the model on each GPU, split the batch,
  all-reduce gradients. The default and the **first thing to reach for**; near-linear scaling when the model
  fits on one GPU. (Never use the old single-process `DataParallel`.) Use `DistributedSampler`; remember the
  effective batch and LR scaling.
- **Sharded data parallel (FSDP / ZeRO)** — when the model **doesn't fit** on one GPU, shard parameters,
  gradients, and optimizer state across GPUs and gather on demand. **FSDP2** (PyTorch) and DeepSpeed **ZeRO**
  (stages 1/2/3) are the standard; ZeRO-3 ≈ full FSDP. The default path for large models.
- **Tensor (model) parallel (TP)** — split individual large layers (matmuls) across GPUs; high communication →
  keep **within a node** (fast NVLink). Megatron-style.
- **Pipeline parallel (PP)** — split layers into stages across devices; micro-batch to keep the pipeline full
  (mind the "bubble").
- **Context/sequence parallel** — split the sequence dimension for very long context (Ring/Ulysses attention).
- **3D (and beyond) parallelism** — combine DP × TP × PP (× sequence × expert) for the largest models; e.g., TP
  within node, PP across nodes, DP/FSDP outermost. This is how trillion-parameter models train. **MoE adds
  expert parallelism** (experts sharded across devices, all-to-all routing).
- **Correctness traps:** scale LR with the global batch and warm up; ensure BatchNorm→SyncBatchNorm (or use
  batch-independent norms); seed per rank for data, identically for init; gradient-accumulate before the
  reduce; checkpoint/restore sharded state correctly; verify loss matches single-GPU on a tiny config before
  scaling. **A distributed bug often shows as "trains but slightly worse" — validate against single-GPU.**
- **Tooling:** `torchrun`/`accelerate` for launch; `accelerate`/Lightning/DeepSpeed/`torchtitan` for
  orchestration; NCCL for comms (watch interconnect — NVLink intra-node, InfiniBand inter-node; comms are
  often the scaling ceiling).

## 5. Numerical stability (silent killer of training)

- **Use log-space:** log-sum-exp for softmax/log-likelihood; never `log(softmax)` naively; cross-entropy from
  logits (fused, stable), not from probabilities.
- **bf16 for range, fp32 for reductions:** keep loss, normalization statistics, and large reductions in fp32;
  bf16 elsewhere. fp16 needs loss scaling to avoid gradient underflow.
- **Guard against NaN/Inf:** gradient clipping (global norm), sane init + warmup, QK-norm / z-loss for LLM
  stability, and input sanitization (no inf/NaN in data). A single bad batch or `log(0)` can poison a run.
- **Watch for spikes:** loss spikes mid-training usually mean LR too high for current curvature, a bad batch, or
  missing clip — lower LR / clip / re-warmup, and consider skipping/clamping outlier batches.
- **Determinism vs. speed:** deterministic kernels reproduce runs exactly but cost throughput (see
  [experimentation-reproducibility.md](experimentation-reproducibility.md) §3).

## 6. Inference & serving optimization

(LLM-specific decoding in [transformers-llms.md](transformers-llms.md) §10.)
- **Quantization:** post-training INT8/INT4/FP8 (GPTQ, AWQ, FP8) for big memory/throughput wins; quantization-
  aware training when PTQ loses too much. Calibrate and **measure quality loss**, don't assume it's free.
- **Compilation / kernels:** `torch.compile`, TensorRT/TensorRT-LLM, ONNX Runtime, custom CUDA/Triton kernels
  for hot ops.
- **Batching & caching:** continuous/in-flight batching (vLLM, TGI), paged KV cache, prefix caching for shared
  prompts.
- **Distillation & pruning:** smaller student models, structured pruning, speculative decoding (draft+verify) to
  cut latency.
- **Right-size the deployment:** match precision, batch size, and hardware to the latency/throughput/cost SLA;
  a distilled or quantized model often meets the SLA at a fraction of the cost.

## 7. Hardware mental model

- **GPUs** (NVIDIA dominant; AMD ROCm maturing): know your card's memory (fits the model?), bf16/FP8 support,
  and memory bandwidth (often the real limit). Tensor cores want shapes that are multiples of 8/16.
- **TPUs:** great for JAX/large-scale; think in terms of the XLA compiler and sharding (`jit` + `shard_map`).
- **Interconnect is destiny at scale:** NVLink (intra-node) ≫ PCIe; InfiniBand (inter-node). Communication, not
  FLOPs, is usually the scaling bottleneck — design parallelism around the network topology.
- **The roofline / arithmetic intensity** decides whether you're compute- or bandwidth-bound and therefore which
  optimization helps. Measure FLOPs/byte before optimizing.

## 8. Reach-for table

| Problem | Reach for |
|---|---|
| Model fits on 1 GPU, want more throughput | DDP + `torch.compile` + bf16 + fused optimizer |
| Out of memory, model still fits | Grad checkpointing, bf16, grad accumulation, 8-bit optimizer |
| Model doesn't fit on 1 GPU | FSDP2 / DeepSpeed ZeRO-3 |
| Single layer too big for a GPU | Tensor parallel (within node) |
| Very deep model across many nodes | Pipeline parallel (+ DP/FSDP) |
| Trillion-scale | 3D parallelism (+ expert parallel for MoE) |
| GPU underutilized | Fix the dataloader; profile before tuning the model |
| Long-context training | FlashAttention + sequence packing (+ context parallel) |
| Fine-tune a large model on 1 GPU | QLoRA (4-bit + LoRA) |
| Fast/cheap serving | Quantization + vLLM/TGI + speculative decoding |

**Canonical references:** PyTorch FSDP & `torch.compile` docs; Rajbhandari et al. 2020 (ZeRO); Shoeybi et al.
2019 (Megatron-LM tensor parallelism); Dao et al. 2022 (FlashAttention); Micikevicius et al. 2018 (mixed
precision); JAX/XLA & Flax NNX docs; the "How to Scale Your Model" / Megatron & DeepSpeed engineering writeups.
