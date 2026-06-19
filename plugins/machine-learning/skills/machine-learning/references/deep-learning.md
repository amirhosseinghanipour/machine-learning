# Deep Learning: Architectures, Training, and Debugging

Core neural-net building blocks, the training recipe that actually works, and the debugging ladder that
saves weeks. Transformers/LLMs are in [transformers-llms.md](transformers-llms.md); generative models in
[generative-models.md](generative-models.md); scaling/distributed in [engineering-scale.md](engineering-scale.md).

---

## 1. Architecture families and their inductive biases

The architecture *is* the prior — it encodes which input variations should and shouldn't change the output.

- **MLP / fully-connected.** No structural prior; universal approximator but data-hungry. Use for tabular
  (though GBMs usually win), as heads/projections, and inside other architectures.
- **CNN (convolutional).** Prior: **translation equivariance** + **locality** + **weight sharing**. The right
  bias for grids (images, spectrograms, some sequences). Receptive field grows with depth/stride/dilation.
  Modern image CNNs (ConvNeXt) remain competitive with ViTs, especially at small/medium data where the
  convolutional prior pays off. Key pieces: conv → norm → activation, pooling/stride for downsampling, 1×1
  convs for channel mixing, depthwise-separable convs for efficiency (MobileNet).
- **RNN / LSTM / GRU.** Prior: sequential, recurrent state. Largely superseded by transformers for most
  sequence tasks, but still relevant for streaming/online, very long or low-compute settings, and as the
  conceptual ancestor of state-space models (see SSMs in [transformers-llms.md](transformers-llms.md)).
  LSTM/GRU gating mitigates vanishing gradients; still struggles with very long dependencies.
- **Transformer.** Prior: **permutation-equivariant content-based mixing** (attention) + position info added
  explicitly. The dominant general-purpose architecture across modalities. Full treatment in
  [transformers-llms.md](transformers-llms.md).
- **Residual networks** are the meta-architecture: $\mathbf{x}_{l+1}=\mathbf{x}_l+f(\mathbf{x}_l)$. Skip
  connections make very deep nets trainable (gradient highway), turn layers into refinements of an identity,
  and are inside virtually every modern architecture (ResNet, Transformer blocks, U-Nets). If a deep net won't
  train, the first question is whether it has residual connections and normalization.
- **Encoder / decoder / encoder-decoder / U-Net.** Choose by the input→output structure: encoders for
  representation (classification, retrieval), decoders for autoregressive generation, encoder-decoder for
  seq2seq (translation), U-Nets (encoder-decoder + skip connections) for dense prediction (segmentation) and
  as the diffusion backbone.

**CNN design notes that still matter.** The ConvNeXt recipe (2022) showed a pure ConvNet matches ViTs when
given the *training recipe*, not the architecture, of modern transformers: large 7×7 depthwise kernels,
inverted bottleneck (expand → depthwise → project), fewer-but-wider stages, GELU, LayerNorm (not BatchNorm),
AdamW + heavy augmentation + stochastic depth + long schedules. The lesson generalizes: when a "new
architecture" beats an old one, first check whether the *recipe* (optimizer, augmentation, schedule length,
regularization) was held fixed — it usually wasn't. Effective receptive field grows as $O(\sqrt{\text{depth}})$
for stacked small kernels, so dilation/striding or a few large kernels buy global context far more cheaply
than depth alone.

**RNN renaissance.** The "RNNs are dead" claim is now partly false: linear-recurrent / SSM models (Mamba-2,
the matrix-form S6) and **minGRU/minLSTM** (gates made input-independent so the recurrence parallelizes via a
prefix scan) recover most of the LSTM's modeling power while training at transformer-like parallelism. They are
the conceptual bridge to the SSMs in [transformers-llms.md](transformers-llms.md); a linear RNN with a
data-dependent diagonal transition *is* a selective SSM.

## 2. The components that make training work

- **Activations.** ReLU (default, simple, can "die" — dead units output constant 0); GELU/SiLU(Swish) (smooth,
  standard in transformers); Leaky/PReLU (avoid dead units); GLU/SwiGLU (gated, strong in transformer MLPs —
  SwiGLU is the modern default FFN). Avoid sigmoid/tanh in deep hidden layers (saturate → vanishing gradients);
  keep them for gates/outputs. **SwiGLU bookkeeping:** a gated FFN has 3 weight matrices, not 2, so to hold
  parameter count fixed against a ReLU/GELU FFN you shrink the hidden width to $\tfrac{2}{3}\times$ (the common
  "$\tfrac{8}{3}d$" hidden size). Practical defaults: GELU/SiLU for vision and general nets; SwiGLU for
  transformer FFNs; plain ReLU only where you want exact sparsity or maximum kernel simplicity.
- **Normalization** (stabilizes and accelerates training by controlling activation statistics):
  - **BatchNorm** — normalizes across the batch; great for CNNs/vision; **breaks with small batches** and
    couples examples (problematic for some tasks, RL, and variable-length). Train/eval behave differently
    (running stats) — a classic bug source.
  - **LayerNorm** — normalizes per-example across features; the **transformer standard**, batch-size
    independent. **RMSNorm** (no mean-centering, $\mathbf{x}/\sqrt{\text{mean}(\mathbf{x}^2)+\epsilon}\cdot\gamma$)
    is the modern, cheaper default in LLMs (Llama, Qwen, etc.) — drops the mean subtraction and bias, ~10–20%
    cheaper, no measured quality loss. Keep the norm and its $\epsilon$ in fp32 even under bf16 (see
    [engineering-scale.md](engineering-scale.md) §5).
  - **GroupNorm** — batch-independent, good for small-batch vision/detection/segmentation; the diffusion U-Net
    default.
  - **Pre-norm vs post-norm:** pre-norm (norm inside the residual branch, $\mathbf{x}+f(\text{norm}(\mathbf{x}))$)
    is far more stable for deep transformers and is now standard — it keeps a clean identity path so gradients
    don't have to pass through a norm. Its cost is that the residual stream's variance grows with depth; the
    fixes seen in current LLMs are **QK-norm** (normalize Q and K before attention — kills attention-logit
    blowup, used in many 2024–25 models) and a **final pre-output norm**. **Post-norm** can edge out pre-norm
    in final quality at moderate depth but needs warmup/careful init to train. Hybrids exist: **DeepNorm**
    (scaled post-norm, trains 1000-layer transformers) and **"sandwich"/peri-norm** (norm before *and* after the
    sublayer) are used when pushing depth.
- **Initialization.** Match init to activation to keep variance stable across depth: **Kaiming/He** for ReLU-
  family ($\text{Var}=2/n_\text{in}$), **Xavier/Glorot** for tanh ($\text{Var}=2/(n_\text{in}+n_\text{out})$).
  Residual nets often scale/zero-init the residual branch (e.g., zero-init final norm γ, or scale the residual
  projection by $1/\sqrt{2L}$ as in GPT-2) so the net starts near identity — a residual block initialized to
  the identity map cannot hurt the forward pass and lets the network "grow into" depth. Bad init = immediate
  plateau or explosion. For principled width/depth-invariant init+LR, see **μP** in §3.
- **Regularization** (control variance — see when to use it via the bias/variance diagnosis in
  [foundations.md](foundations.md)):
  - **Weight decay** (≈ $\ell_2$; use **decoupled** AdamW) — the most reliable knob.
  - **Dropout** — strong for MLPs/RNNs; less used in large transformers (where data scale regularizes), but
    standard in fine-tuning and attention/residual dropout in some recipes.
  - **Data augmentation** — usually the highest-leverage regularizer; it directly injects the invariances you
    want (see modality-specific augmentations in [data.md](data.md)). MixUp/CutMix, RandAugment for vision.
  - **Label smoothing** (typ. 0.1) — softens targets, improves calibration, but *can hurt* knowledge
    distillation (it erases the inter-class similarity structure the teacher provides); skip it when you'll
    distill from the model.
  - **Stochastic depth / DropPath** — randomly drop whole residual branches in training (rate ramped with
    depth); the key regularizer that makes very deep vision transformers/ConvNeXt train well.
  - **Early stopping** (on validation), **EMA of weights** (a free, reliable boost — keep an exponential moving
    average of params for evaluation; decay ~0.999–0.9999, the de-facto standard for diffusion and many vision
    models). EMA also smooths the noise that a constant-LR / WSD-stable phase leaves behind.
  - **Spectral/gradient penalties** for stability (GANs, certified robustness).
  - **A note on the modern regime:** in large-scale pretraining the dataset is effectively seen ~once, so the
    classical overfitting these tools fight barely occurs — weight decay (often acting more as an optimization/
    conditioning knob than a true prior) and a touch of dropout in fine-tuning are usually all that's used.
    Match the regularizer to the regime (underparameterized vs interpolating — see
    [foundations.md](foundations.md)); aggressive dropout on a one-epoch LLM run just wastes capacity.

## 3. Optimizers, learning rates, and schedules (the recipe)

(Theory in [foundations.md](foundations.md) §4; this is the practitioner's recipe.)

- **Optimizer:** **AdamW** is the default for transformers and most deep nets ($\beta_1{=}0.9$, $\beta_2{=}0.95$
  for LLMs / $0.999$ otherwise, $\epsilon{=}10^{-8}$, decoupled weight decay ~0.1 for LLMs). *Decoupled* matters:
  AdamW applies $\theta \leftarrow \theta - \eta\lambda\theta$ separately from the adaptive step, so weight
  decay is true $\ell_2$ shrinkage independent of the gradient scale — plain "Adam + L2 in the loss" is *not*
  equivalent and underperforms. Note the AdamW decay is coupled to LR; the effective shrinkage per step is
  $\eta\lambda$, so re-tune $\lambda$ if you change the schedule peak. **SGD+momentum** still edges out Adam on
  some vision CNNs and can generalize slightly better — try it for ConvNets. Optimizers worth knowing in 2026:
  - **Lion** (sign-based, $\text{sign}(\beta_1 m + (1{-}\beta_1)g)$) — half Adam's optimizer memory (one state
    not two), competitive on large-batch vision/LM; needs ~3–10× *smaller* LR and *larger* weight decay than
    AdamW because the update magnitude is fixed at $\pm1$ per coordinate.
  - **Adafactor** — factorizes the second-moment matrix into row/column statistics → near-zero optimizer memory;
    the historical large-model choice (T5, PaLM) when optimizer state didn't fit. Slightly worse than Adam
    per-step; pairs with relative-step LR.
  - **Shampoo / SOAP** — approximate full-matrix (Kronecker-factored) preconditioning. SOAP (Vyas et al.,
    ICLR 2025) = "run Adam in the eigenbasis of Shampoo's preconditioner"; it adds only one extra
    hyperparameter (preconditioning-update frequency) and in the large-batch regime cuts steps ~40% and
    wall-clock ~35% vs AdamW. Distributed Shampoo is in production at scale; the cost is the periodic
    eigendecomposition and extra state.
  - **Muon** (Jordan 2024; scaled to trillion-param **Kimi K2** as **MuonClip**) — for **2D weight matrices
    only**: take SGD-momentum, then **orthogonalize the update** via a few (≈5) Newton–Schulz iterations
    (a quintic polynomial run in bf16) so the applied update is approximately the polar factor (semi-orthogonal).
    Intuition: it equalizes the update's singular values, taking a large step in *every* direction rather than
    collapsing onto the dominant one. Empirically ~1.3–2× more token-efficient than AdamW at fixed compute.
    Practical wiring: **Muon on hidden matmul weights, AdamW on embeddings, the LM head, norms, and all 1-D
    params**; scale Muon's per-matrix LR so its update RMS matches AdamW's; add weight decay. At scale it needs
    stabilization (Kimi's **QK-Clip** to bound attention logits) — MuonClip reportedly eliminated loss spikes
    across a trillion-token run. This is the most significant optimizer shift of 2024–25; reach for it on a
    pretraining run where AdamW is the incumbent.
  - **Schedule-free** (Defazio et al. 2024, won the 2024 AlgoPerf self-tuning track) — folds Polyak/Primal
    averaging into the optimizer so you need **no LR decay schedule and no preset run length**; strong at small/
    medium batch but tends to fall behind WSD/cosine at very large batch. Useful when total steps are unknown
    (continual / open-ended training).

  Don't hand-roll an optimizer unless you must; do swap AdamW→Muon/SOAP when a pretraining budget justifies the
  validation.
- **Learning rate is the single most important hyperparameter.** Find it with an **LR-range test** (sweep LR
  up, watch loss). Too high → diverge/NaN; too low → crawl. Typical AdamW LRs: 1e-3 to 3e-4 (from scratch,
  small/medium models), ~1e-4 to 2e-4 (large-LM pretraining), 1e-5 to 5e-5 (fine-tuning large models). LR
  scales **down** as model width grows under standard parametrization — which motivates μP below.
- **μP (maximal update parametrization) — tune small, transfer large.** Under standard init/LR, the optimal LR
  drifts with width, so the LR you find on a 100M model is wrong for a 10B model. μP rescales per-layer init
  variances and LRs so that the optimal HPs become **width- (and, with depth-μP, depth-) invariant**: sweep on
  a cheap proxy model, then transfer the *same* HPs to the target with no re-sweep. This is now standard
  practice for serious pretraining (it's how you avoid burning the compute budget on HP search at scale).
  **u-μP** (ICLR 2025) combines μP with unit scaling for simpler defaults and easier FP8. Caveat: μP needs care
  (coordinate-check the activations across widths to confirm the transfer actually holds before trusting it).
- **Schedule:** **linear warmup** (critical for transformers/Adam — a few hundred to a few thousand steps; it
  lets Adam's second-moment estimate stabilize before large steps and prevents the early-step blowup) then
  **cosine decay** to ~10% of peak. Alternatives: **linear decay**; **WSD (warmup-stable-decay)** — warmup →
  long constant-LR "stable" phase → short (~10–20% of steps) decay/cooldown to near-zero. WSD's advantage is
  *flexibility*: you can branch a decay from any stable-phase checkpoint, so run length need not be fixed in
  advance, and you can resume/extend. The loss drops sharply during the cooldown ("river-valley" picture —
  the stable phase travels along a valley, the cooldown descends into it). Empirically WSD ≈ cosine at matched
  budget. Weight-averaging the stable phase (EMA/LAWA) can approximate the cooldown's gain without a separate
  decay.
- **Batch size:** larger = more stable, more parallel, but with diminishing returns past a **critical batch
  size** (McCandlish et al.) — beyond it, doubling batch no longer halves steps-to-target. The critical batch
  size *grows* as the loss falls during training, so a fixed large batch is wasteful early and good late
  (batch-size warmup helps). Scale LR with batch size: **linear** for SGD, **square-root** is the better rule
  for Adam-family. Use **gradient accumulation** to simulate large batches on limited memory.
- **Gradient clipping** (by global norm, e.g., 1.0) prevents loss spikes — standard for transformers/RNNs.
  Log the *pre-clip* grad norm; a sustained rise foreshadows divergence and a sudden spike marks the bad batch.
- **Mixed precision** (bf16 preferred over fp16; fp16 needs loss scaling): ~2× speed/memory, near-free. See
  [engineering-scale.md](engineering-scale.md).

A reliable from-scratch transformer recipe: AdamW ($\beta_2{=}0.95$, wd 0.1), gradient clip 1.0, linear warmup
(~1–2% of steps) → cosine decay to 10%, bf16, the largest batch under the critical size, LR found by range test
(or transferred via μP). Start there, then ablate. For a large pretraining budget, the current frontier swap is
Muon(+AdamW on 1-D params) with WSD and μP-transferred HPs.

## 4. The debugging ladder (do these in order)

Most "the model doesn't work" problems are caught by climbing this ladder. **Do not** tune hyperparameters
before the lower rungs pass.

1. **Overfit a single batch to ~0 loss.** If you can't, you have a *bug* (wrong loss, detached graph, label
   misalignment, frozen params, broken data path) — not a modeling problem. This catches 80% of issues in
   minutes.
2. **Check the loss at init.** It should equal the theoretical value for random predictions (e.g.,
   $-\log(1/C)=\ln C$ for $C$-class cross-entropy). A wrong init loss reveals scaling/label/softmax bugs.
3. **Verify shapes and the data pipeline.** Visualize/inspect a batch *as the model sees it* (after all
   transforms). Confirm labels align with inputs, no off-by-one, no accidental shuffle of X vs y. Print
   min/max/mean of inputs.
4. **Run a label-shuffled / random-feature control.** The model should *not* be able to fit shuffled labels
   better than chance on a held-out set — if it can, you have leakage or an eval bug.
5. **Watch gradient and activation statistics.** Grad norms should be finite, non-zero, and stable; dead ReLUs
   (always-zero activations), exploding/vanishing grads, and saturated units all show up here. Log them.
6. **Scale data up gradually.** Small subset → confirm it learns and overfits → full data. Loss curves should
   be smooth-ish; spikes mean LR/clip/data issues.
7. **Only now tune** LR, schedule, regularization, architecture — using validation, with the rigor checklist
   from SKILL.md.

**Symptom → cause quick map:**
- Loss is NaN: LR too high, fp16 overflow (use bf16), bad/inf inputs, log(0). Lower LR, clip, sanitize data.
- Train loss won't go down: bug (rung 1 fails), LR too low/high, dead units, bad init/no norm.
- Train good, val bad: overfitting (regularize/more data/augment) or **leakage in the other direction** (val is
  easier than it should be — audit splits).
- Loss spikes mid-training: LR too high for current curvature, bad batch, missing grad clip; lower LR/clip/warmup.
  For LLMs specifically, attention-logit blowup is a common culprit — add **QK-norm** or **z-loss**
  (a small penalty on the log-partition $\log\sum e^{z_i}$ that keeps logits bounded). MuonClip's QK-Clip is the
  same idea for Muon.
- Val loss noisy/non-monotonic: small val set, high LR, or BatchNorm train/eval mismatch.

**Gradient pathologies — name them precisely.**
- **Vanishing/exploding gradients.** In a plain deep stack the gradient is a product of $L$ Jacobians; its norm
  scales like $\prod \|J_l\|$, so it shrinks or blows up exponentially in depth. The structural fixes (not LR
  band-aids) are **residual connections** (turn the product into a sum of paths) + **normalization** (keep
  per-layer Jacobian scale ≈ 1) + **variance-preserving init**. If a deep net won't train, check these *before*
  touching the optimizer.
- **Dead ReLUs.** A unit whose pre-activation is always negative gets zero gradient forever. Caused by too-high
  LR (large negative bias drift) or bad init; mitigate with smaller LR, GELU/SiLU/LeakyReLU, or proper init.
  Monitor the fraction of always-zero activations.
- **Rank collapse / over-smoothing.** Deep attention without residuals/norm drives all tokens toward the same
  representation (token uniformity); in GNNs the analog is over-smoothing. Residual + norm + enough heads
  prevents it; it's a real failure mode in very deep or under-normalized stacks.
- **Loss-of-plasticity / dead-unit creep** in long or continual training: a growing fraction of units saturate
  and stop adapting. Weight decay, occasional re-initialization of dead units (continual-backprop), or shrink-
  and-perturb help; relevant to RL and lifelong setups ([learning-paradigms.md](learning-paradigms.md)).

## 5. Architecture design principles

- **Start from a known-good architecture** for your modality and modify incrementally. Don't design from
  scratch; the field has strong priors baked into reference models (ResNet/ConvNeXt for vision, a standard
  pre-norm transformer for sequences, U-Net for dense prediction/diffusion).
- **Width vs. depth:** depth adds expressivity/abstraction but needs residuals+norm to train; width is easier
  to optimize and parallelize. Scale both per scaling laws.
- **Bottlenecks and skip connections** structure information flow (autoencoders, U-Nets, inverted residuals).
- **Parameter sharing** (conv, recurrence, weight tying input/output embeddings) is regularization + efficiency.
- **Heads:** keep task-specific heads small; put capacity in the shared backbone. Linear-probe a frozen
  backbone to measure representation quality (see [representation-learning.md](representation-learning.md)).
- **Match capacity to data and compute** (scaling laws, [transformers-llms.md](transformers-llms.md)); an
  oversized model on small data just overfits or wastes compute.

## 6. Practical training loop (PyTorch idiom)

```python
model.train()
for step, (x, y) in enumerate(loader):
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  # mixed precision
        logits = model(x)
        loss = loss_fn(logits, y) / accum_steps
    loss.backward()                                   # grads accumulate
    if (step + 1) % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip
        optimizer.step()
        scheduler.step()                              # per-step warmup+cosine
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)                             # EMA of weights for eval
```

Validate with `model.eval()` + `torch.no_grad()` (or `inference_mode`), use the EMA/averaged weights, and
log everything to your tracker (see [experimentation-reproducibility.md](experimentation-reproducibility.md)).
Wrap the model in `torch.compile` for speed and use `DistributedSampler`/FSDP for scale
([engineering-scale.md](engineering-scale.md)).

**Mixed precision in one paragraph** (full treatment in [engineering-scale.md](engineering-scale.md) §2,§5).
Autocast runs the matmul/conv-heavy forward in low precision while keeping a master fp32 copy of the weights
and doing the optimizer update in fp32. **Prefer bf16**: it shares fp32's exponent range, so gradients don't
underflow and you need *no* loss scaling — fewer NaNs, simpler code. Use **fp16 only** on hardware without
bf16, and then with a `GradScaler` (it multiplies the loss up so small gradients survive fp16's narrow range,
then unscales before the step). Always keep **reductions, the loss, softmax/cross-entropy, and normalization
statistics in fp32** — that's where low precision bites. The bf16 mantissa is only 8 bits (~2–3 decimal
digits), so large running sums (EMA, accumulators, very long residual streams) can stagnate; keep those fp32.

## 7. Transfer learning is usually the right default

Training from scratch is rarely optimal outside research on the pretraining recipe itself. Start from a
pretrained backbone (ImageNet/DINOv2 for vision, a pretrained LM for text, a foundation model for your
modality) and fine-tune or use PEFT/LoRA. Covered in [representation-learning.md](representation-learning.md).

**Canonical references:** Goodfellow, Bengio & Courville *Deep Learning*; Prince *Understanding Deep Learning*
(2023, the current go-to text — free PDF); Murphy *PML* vol. 2; He et al. 2015 (ResNet) & 2016 (identity
mappings/pre-activation); Ioffe & Szegedy 2015 (BatchNorm); Ba et al. 2016 (LayerNorm); Zhang & Sennrich 2019
(RMSNorm); Liu et al. 2022 (ConvNeXt); Kingma & Ba 2015 (Adam); Loshchilov & Hutter 2019 (AdamW); Chen et al.
2023 (Lion); Vyas et al. 2025 (SOAP); Jordan et al. 2024 + Kimi Team 2025 (Muon / MuonClip); Defazio et al.
2024 (Schedule-Free); Yang et al. 2022 (μP / μTransfer) & u-μP (ICLR 2025); Hu et al. 2024 (WSD / MiniCPM);
McCandlish et al. 2018 (critical batch size); Karpathy's "A Recipe for Training Neural Networks" (the
debugging-ladder ethos).
