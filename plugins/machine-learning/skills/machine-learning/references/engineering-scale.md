# Engineering & Scale

Making training and inference fast, memory-feasible, numerically stable, and correct at scale. Architectures are
in the model references; this is the systems layer. Current to 2026 (PyTorch 2.10–2.12, FSDP2, `torch.compile`).

---

## 1. Frameworks and their idioms

- **PyTorch** (research default; 2.11–2.12 current in mid-2026). Eager by default; wrap the model in
  **`torch.compile`** for 20–50% speedups (TorchDynamo graph capture → TorchInductor fusion + autotuned
  Triton/CUDA kernels) — default-on for serious training as of 2.x. Mind: first-call compile latency,
  **graph breaks** (data-dependent Python control flow, `.item()`/`.tolist()`, printing, unsupported ops force
  a fall-back to eager — inspect with `TORCH_LOGS=graph_breaks`), and **recompiles** on changing shapes (pad/
  bucket sequence lengths, or `torch.compile(dynamic=True)` to compile shape-symbolic guards once;
  `TORCH_LOGS=recompiles` to diagnose). Modes: default; `mode="reduce-overhead"` (wraps in **CUDA graphs** to
  kill per-kernel launch overhead — great for small-batch inference and tight training steps, but needs static
  shapes/addresses); `mode="max-autotune"` (longer compile, best kernels). **Regional compilation** —
  `torch.compile` a single repeated block (e.g. one transformer layer) instead of the whole model — slashes
  compile time and avoids whole-model graph breaks; the standard pattern for big stacks. 2.12 adds a unified
  `torch.accelerator.Graph` capture API and MX-format export. Use `torch.profiler` to find bottlenecks.
- **JAX + Flax NNX / Equinox** (TPU & large-scale / functional). Pure functions + `jit`/`grad`/`vmap`/`pmap`/
  `shard_map` compose beautifully; **NNX** is the recommended Flax API (Pythonic, mutable-feeling) over Linen.
  Excellent for research that needs per-example gradients, ensembles via `vmap`, or first-class parallelism;
  steeper functional-purity learning curve. MaxText/Levanter for foundation models.
- **Choosing:** PyTorch by default (ecosystem, `transformers`, community); JAX for TPUs, research-grade
  parallelism, or heavy functional transforms. Don't switch frameworks mid-project without a real reason.

## 2. Memory: where it goes and how to fit more

Training memory = **parameters + gradients + optimizer state + activations + workspace**. The **model-state**
bill for AdamW with bf16 compute + fp32 master weights is **~16–18 bytes/param**: 2 B (bf16 param) + 2 B (bf16
grad) + 4 B (fp32 master) + 4 B (Adam $m$) + 4 B (Adam $v$) = 16 (older recipes that keep an fp32 grad copy hit
18). A 7 B model is thus ~112 GB of *state alone* — already over one 80 GB GPU before a single activation, which
is why sharding (§4) is mandatory past a few billion params. Optimizer choice moves this number: SGD-momentum
≈ 12 B/param, 8-bit Adam ≈ 10 B, Adafactor/Lion ≈ 10–12 B (one state, not two). **Activations** scale with
batch × sequence × width × depth and often dominate at long context. The Korthikanti et al. estimate for one
transformer layer (no recompute) is ≈ $sbh(34 + 5\,\frac{as}{h})$ bytes — the $5as/h$ term is the attention
score matrix, quadratic in sequence length $s$, which is exactly what FlashAttention removes from memory.
Levers, cheapest first:
- **Mixed precision (bf16)** — ~½ the memory and ~2× throughput on modern GPUs; **bf16 > fp16** (same exponent
  range as fp32 → no loss-scaling needed, far fewer NaNs). Use fp16 only on hardware without bf16, with a
  `GradScaler`. Keep a master fp32 copy of weights / norm accumulations. (See [deep-learning.md](deep-learning.md)
  §6 for the autocast mental model.)
- **FP8 training (Hopper/Blackwell)** — the matmuls run in 8-bit (E4M3 fwd, often E5M2 grad) for another ~1.3–2×
  over bf16; only the GEMMs are fp8, everything else stays bf16/fp32. The whole game is **scaling** because fp8's
  dynamic range is tiny: **per-tensor delayed scaling** (amax tracked over a history window — fastest, used on
  Hopper via Transformer Engine) vs **fine-grained block scaling** — **MXFP8** (a shared scale per 32-element
  block, hardware-native on Blackwell) and DeepSeek-V3's 1×128/128×128 tile scaling with high-precision CUDA-core
  accumulation, which held the loss gap to <0.25% vs bf16 on a frontier-scale run. Defaults: try bf16 first, add
  fp8 only when you can validate the loss curve against a bf16 baseline; keep the LM head, embeddings, and norms
  out of fp8.
- **Gradient accumulation** — simulate a large batch in micro-batches (more steps, same effective batch). Free
  except wall-clock.
- **Gradient (activation) checkpointing** — don't store activations; recompute them in the backward pass. Trades
  compute for activation memory: full (every-layer) recompute is the classic ~$\sqrt{N}$-memory / +33%-compute
  point, but **selective recomputation** (Korthikanti et al. — store the cheap-to-keep activations, recompute
  only the memory-heavy attention block) recovers most of the saving for far less than 33% overhead and is the
  modern default. In PyTorch use `torch.utils.checkpoint` (`use_reentrant=False`) or FSDP2's wrapper; checkpoint
  *per transformer block*, not per op.
- **Optimizer-state sharding / 8-bit optimizers** — `bitsandbytes` 8-bit Adam, Adafactor (no second moment),
  or shard optimizer state across GPUs (ZeRO/FSDP §4).
- **Efficient attention** — FlashAttention / SDPA tiles the attention computation and uses the online-softmax
  trick so the $L\times L$ score matrix is never materialized in HBM (memory $O(L)$ not $O(L^2)$); it is an
  *exact* (not approximate) attention and is faster because attention is memory-bandwidth-bound, not
  compute-bound. **FlashAttention-3** (2024) adds Hopper warp-specialization (TMA producer/consumer pipelines),
  matmul–softmax overlap, and FP8 — ~75% H100 utilization in bf16 (≈840 TFLOP/s), up to ~1.3 PFLOP/s in fp8;
  FA-4 lands with newer PyTorch. Always call it through `F.scaled_dot_product_attention` (auto-selects the best
  backend) or the `flash-attn` package; use the **varlen** API with sequence packing (below) for ragged
  batches.
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
  you which you're in — compare your kernel's arithmetic intensity (FLOPs/byte) against the GPU's ridge point
  (peak FLOP/s ÷ peak GB/s; ≈ 295 for an H100 in bf16). Below the ridge you're bandwidth-bound (fusion, larger
  tiles, fewer passes help); above it you're compute-bound (need better matmul shapes/precision).
- **Measure efficiency with MFU (Model FLOPs Utilization),** the fraction of peak FLOP/s your run actually
  uses. Estimate model FLOPs as **$C \approx 6N$ per token** (forward+backward, dense; $2N$ fwd + $4N$ bwd),
  plus the attention term $12\,L\,h\,s$ per token that matters at long context, so
  $\text{MFU} = \frac{6N \cdot (\text{tokens/s})}{\text{peak FLOP/s}}$. **Targets:** well-tuned dense LLM
  pretraining hits ~40–55% MFU; <30% means you're leaving half the cluster idle — look at the dataloader,
  comms overlap, recompute overhead, or small/odd matmul shapes. MoE and very-long-context runs report lower
  MFU and need their own FLOP accounting. (Distinguish MFU from *HFU*, hardware FLOPs utilization, which counts
  recomputation as "useful" and so reads higher.)

## 4. Distributed training (multi-GPU / multi-node)

Pick the minimal parallelism that fits; complexity compounds.
- **Data Parallel (DDP / DistributedDataParallel)** — replicate the model on each GPU, split the batch,
  all-reduce gradients. The default and the **first thing to reach for**; near-linear scaling when the model
  fits on one GPU. (Never use the old single-process `DataParallel`.) Use `DistributedSampler`; remember the
  effective batch and LR scaling.
- **Sharded data parallel (FSDP / ZeRO)** — when the model **doesn't fit** on one GPU, shard parameters,
  gradients, and optimizer state across GPUs and gather on demand. The **ZeRO** stages name what you shard:
  **1** = optimizer state, **2** = +gradients, **3** = +parameters (gathered just-in-time per layer in forward/
  backward, then freed). ZeRO-3 ≈ full FSDP. **FSDP2** (PyTorch, now the recommended API; FSDP1 is legacy)
  represents each parameter as a **DTensor** sharded on dim-0 instead of FSDP1's one flat 1-D buffer — so
  per-parameter semantics, clean composition with TP/PP (it's the basis of `torchtitan`'s ND parallelism),
  sharded checkpoints that need no communication, and deterministic memory release (no `record_stream`, lower
  peak). Use `fully_shard` on each transformer block + the root; this is the default path for large models.
  ZeRO-3/FSDP communication is ~1.5× DDP's (extra all-gather of params) — overlap it (below) or it dominates.
- **Tensor (model) parallel (TP)** — split individual large layers (matmuls) across GPUs (column-parallel then
  row-parallel, an all-reduce per block); high communication → keep **within a node** (fast NVLink). Megatron-
  style. **Async-TP / `torch.symm_mem`** (PyTorch symmetric memory) decomposes and overlaps the TP collectives
  with the matmul to hide much of that cost.
- **Pipeline parallel (PP)** — split layers into stages across devices; micro-batch to keep the pipeline full.
  The idle time is the **bubble**, fraction ≈ $(P-1)/(m+P-1)$ for $P$ stages and $m$ micro-batches → use many
  micro-batches. Interleaved/1F1B schedules (Megatron) and **zero-bubble / DualPipe** (DeepSeek-V3, overlaps the
  fwd/bwd of paired micro-batches to nearly eliminate the bubble) are the current schedulers.
- **Context/sequence parallel** — split the sequence dimension for very long context. **Ring attention** passes
  K/V blocks around a ring of GPUs overlapped with compute (near-infinite context); **Ulysses** all-to-alls the
  head dimension. **Sequence parallelism** (Megatron) additionally shards the norm/dropout activations along
  sequence to cut the activation memory TP leaves behind — pairs naturally with selective recompute.
- **3D (and beyond) parallelism** — combine DP × TP × PP (× sequence × expert) for the largest models; e.g., TP
  within node (NVLink), PP across nodes, DP/FSDP outermost. This is how trillion-parameter models train. **MoE
  adds expert parallelism** (experts sharded across devices, **all-to-all** routing twice per layer) — the
  all-to-all is the new bottleneck, which is why DeepSeek-V3 co-designed DualPipe to hide it and used an
  **auxiliary-loss-free** bias-based load balancer to keep experts balanced without the usual quality hit.
- **Correctness traps:** scale LR with the global batch and warm up; ensure BatchNorm→SyncBatchNorm (or use
  batch-independent norms); seed per rank for data, identically for init; gradient-accumulate before the
  reduce; checkpoint/restore sharded state correctly; verify loss matches single-GPU on a tiny config before
  scaling. **A distributed bug often shows as "trains but slightly worse" — validate against single-GPU.**
- **Communication overlap is the lever at scale.** Once the model is sharded, the wall-clock is set by how much
  collective time you can hide behind compute. DDP overlaps gradient all-reduce with backward automatically
  (bucketed); FSDP overlaps the next layer's param all-gather with the current layer's compute (tune
  `limit_all_gathers`/prefetch); async-TP and DualPipe overlap TP/PP comms explicitly. Use **gradient
  accumulation** to amortize the reduce over more compute, and prefer **bf16 gradient reduction** to halve
  comms volume. The diagnostic: a comms-bound run shows low SM occupancy with NCCL kernels on the timeline —
  fix the overlap, not the model.
- **Tooling:** `torchrun`/`accelerate` for launch; `accelerate`/Lightning/DeepSpeed/`torchtitan` (the PyTorch-
  native reference for FSDP2 + TP + PP + fp8) for orchestration; NCCL for comms (watch interconnect — NVLink/
  NVSwitch intra-node ≫ PCIe; InfiniBand/RoCE inter-node; comms are often the scaling ceiling). Set
  `NCCL_DEBUG=INFO` once to confirm it picked the fast transport, not a fallback.

## 5. Numerical stability (silent killer of training)

- **Use log-space:** log-sum-exp for softmax/log-likelihood; never `log(softmax)` naively; cross-entropy from
  logits (fused, stable), not from probabilities.
- **bf16 for range, fp32 for reductions:** keep loss, normalization statistics, and large reductions in fp32;
  bf16 elsewhere. fp16 needs loss scaling to avoid gradient underflow.
- **Guard against NaN/Inf:** gradient clipping (global norm), sane init + warmup, QK-norm / z-loss for LLM
  stability, and input sanitization (no inf/NaN in data). A single bad batch or `log(0)` can poison a run.
  `torch.autograd.set_detect_anomaly(True)` localizes the first NaN (slow — debugging only); for production,
  log pre-clip grad norm and skip/clamp batches whose norm exceeds a threshold.
- **fp8 needs scale hygiene:** the failure mode is amax tracking lagging a sudden activation spike, so a value
  saturates E4M3's small range → Inf → NaN. Delayed scaling trades a little accuracy for speed by using a
  history-window amax; if a run goes unstable in fp8 but is fine in bf16, suspect the scaling, not the model —
  switch that tensor to current/finer-grained (block/MX) scaling or keep it in bf16.
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

**Custom kernels (Triton) — when and how.** Reach for a hand-written kernel only after profiling says a hot op
is *memory-bound and fusible* and `torch.compile` didn't fuse it (check the Inductor output first — it often
already emits Triton). The win is **fusion**: collapse a chain of pointwise/reduction ops (e.g. RMSNorm,
SwiGLU, fused cross-entropy, rotary) into one kernel so the data is read from HBM once instead of per-op,
eliminating launch overhead and intermediate round-trips — the **Liger-Kernel** library packages exactly these
for LLM training (~20% throughput, ~60% activation-memory cut). Triton writes block-level Python (you tile the
problem; the compiler handles intra-block vectorization/coalescing), which is far more productive than CUDA and
usually within ~10% of it. Practical guidance: tile sizes powers of two, 32–256, multiples of 128 bytes for
coalescing; use `@triton.autotune` to sweep `BLOCK_SIZE`/`num_warps`/`num_stages` (each combination is a
distinct compiled binary, so autotune compiles+benchmarks them — cache the result, it's not free); keep
reductions/accumulators in fp32 even with bf16 I/O. For matmul-heavy ops, prefer the vendor library (cuBLAS/
CUTLASS) — don't out-write the GEMM. Match a reference implementation numerically before trusting it.
- **Batching & caching:** continuous/in-flight batching (vLLM, TGI), paged KV cache, prefix caching for shared
  prompts.
- **Distillation & pruning:** smaller student models, structured pruning, speculative decoding (draft+verify) to
  cut latency.
- **Right-size the deployment:** match precision, batch size, and hardware to the latency/throughput/cost SLA;
  a distilled or quantized model often meets the SLA at a fraction of the cost.

## 7. Hardware mental model

- **GPUs** (NVIDIA dominant; AMD ROCm maturing): know your card's memory (fits the model?), bf16/FP8 support,
  and memory bandwidth (often the real limit). Rough current landmarks: **H100** 80 GB HBM3 (~3.35 TB/s), ~990
  bf16 TFLOP/s, fp8; **H200** 141 GB; **B200/Blackwell** ~190 GB + native MXFP8/FP4 and much higher bf16/fp8
  throughput. Tensor cores want shapes that are multiples of 8/16 (and ideally 64/128 for fp8/large GEMMs) —
  a hidden size of 4097 silently loses tensor-core efficiency. Mind that *advertised* peak FLOP/s often assumes
  2:4 structured sparsity; halve it for dense.
- **TPUs:** great for JAX/large-scale; think in terms of the XLA compiler and sharding (`jit` + `shard_map`),
  the MXU systolic array (wants large matmuls), and the 2D/3D torus interconnect (ICI) — shard so collectives
  stay on-mesh.
- **Interconnect is destiny at scale:** NVLink/NVSwitch (intra-node, ~900 GB/s on Hopper) ≫ PCIe (~64 GB/s);
  InfiniBand/RoCE (inter-node, ~400 Gb/s/NIC). The ~10–20× intra/inter gap is *why* you keep high-comms TP
  inside a node and put low-comms DP/PP across nodes. Communication, not FLOPs, is usually the scaling
  bottleneck — design parallelism around the network topology.
- **The roofline / arithmetic intensity** decides whether you're compute- or bandwidth-bound and therefore which
  optimization helps. Measure FLOPs/byte before optimizing. The same op flips regimes with batch size: a matmul
  with batch 1 (decode) is bandwidth-bound (weights stream once per token), the same matmul with batch 256 is
  compute-bound — which is the whole reason inference batches.

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
| Long-context training | FlashAttention + sequence packing (+ context/sequence parallel) |
| Fine-tune a large model on 1 GPU | QLoRA (4-bit + LoRA) |
| Squeeze >1.3× over bf16 on Hopper/Blackwell | fp8 GEMMs (delayed or MX/block scaling), validate loss vs bf16 |
| MFU stuck <30% | Profile: fix dataloader, comms overlap, matmul shapes, recompute overhead — in that order |
| Comms dominating at scale | Overlap (FSDP prefetch, async-TP, DualPipe), bf16 grad reduction, keep TP intra-node |
| Pick HPs once, train big | μP / μTransfer (see [deep-learning.md](deep-learning.md) §3) |
| Hot fusible op `torch.compile` missed | Custom Triton kernel (Liger-style fusion); else leave it |
| Fast/cheap serving | Quantization + vLLM/TGI + speculative decoding |

**Canonical references:** PyTorch FSDP2 / `torch.compile` / `torch.distributed` docs and the `torchtitan` repo;
Rajbhandari et al. 2020 (ZeRO) & ZeRO-Offload/Infinity; Shoeybi et al. 2019 (Megatron-LM tensor parallelism) &
Korthikanti et al. 2023 (sequence parallelism + selective recompute); Dao et al. 2022 (FlashAttention-2) &
Shah et al. 2024 (FlashAttention-3); Liu et al. 2023 (Ring Attention); Micikevicius et al. 2018 (mixed
precision); NVIDIA Transformer Engine / FP8 (delayed & MXFP8) docs; DeepSeek-AI 2024 (V3 technical report —
fp8, DualPipe, aux-loss-free MoE); Hwu/PaLM (MFU); the NVIDIA & "How to Scale Your Model" (Google DeepMind)
scaling guides; Liger-Kernel & Triton docs; JAX/XLA & Flax NNX docs.
