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

## 2. The components that make training work

- **Activations.** ReLU (default, simple, can "die" — dead units output constant 0); GELU/SiLU(Swish) (smooth,
  standard in transformers); Leaky/PReLU (avoid dead units); GLU/SwiGLU (gated, strong in transformer MLPs —
  SwiGLU is the modern default FFN). Avoid sigmoid/tanh in deep hidden layers (saturate → vanishing gradients);
  keep them for gates/outputs.
- **Normalization** (stabilizes and accelerates training by controlling activation statistics):
  - **BatchNorm** — normalizes across the batch; great for CNNs/vision; **breaks with small batches** and
    couples examples (problematic for some tasks, RL, and variable-length). Train/eval behave differently
    (running stats) — a classic bug source.
  - **LayerNorm** — normalizes per-example across features; the **transformer standard**, batch-size
    independent. **RMSNorm** (no mean-centering) is the modern, cheaper default in LLMs.
  - **GroupNorm** — batch-independent, good for small-batch vision/detection/segmentation.
  - **Pre-norm vs post-norm:** pre-norm (norm inside the residual branch) is far more stable for deep
    transformers and is now standard.
- **Initialization.** Match init to activation to keep variance stable across depth: **Kaiming/He** for ReLU-
  family, **Xavier/Glorot** for tanh. Residual nets often scale/zero-init the residual branch (e.g., zero-init
  final norm γ) so the net starts near identity. Bad init = immediate plateau or explosion.
- **Regularization** (control variance — see when to use it via the bias/variance diagnosis in
  [foundations.md](foundations.md)):
  - **Weight decay** (≈ $\ell_2$; use **decoupled** AdamW) — the most reliable knob.
  - **Dropout** — strong for MLPs/RNNs; less used in large transformers (where data scale regularizes), but
    standard in fine-tuning and attention/residual dropout in some recipes.
  - **Data augmentation** — usually the highest-leverage regularizer; it directly injects the invariances you
    want (see modality-specific augmentations in [data.md](data.md)). MixUp/CutMix, RandAugment for vision.
  - **Label smoothing**, **stochastic depth**, **early stopping** (on validation), **EMA of weights** (a free,
    reliable boost — keep an exponential moving average of params for evaluation).
  - **Spectral/gradient penalties** for stability (GANs, certified robustness).

## 3. Optimizers, learning rates, and schedules (the recipe)

(Theory in [foundations.md](foundations.md) §4; this is the practitioner's recipe.)

- **Optimizer:** **AdamW** is the default for transformers and most deep nets ($\beta_1{=}0.9$, $\beta_2{=}0.95$
  for LLMs / $0.999$ otherwise, decoupled weight decay ~0.1 for LLMs). **SGD+momentum** still edges out Adam on
  some vision CNNs and can generalize slightly better — try it for ConvNets. **Lion**, **Adafactor**
  (memory-efficient, used for large models), **Shampoo/SOAP** (approximate second-order, increasingly used at
  scale) are worth knowing. Don't hand-roll an optimizer unless you must.
- **Learning rate is the single most important hyperparameter.** Find it with an **LR-range test** (sweep LR
  up, watch loss). Too high → diverge/NaN; too low → crawl. Typical AdamW LRs: 1e-3 to 3e-4 (from scratch),
  1e-5 to 5e-5 (fine-tuning large models).
- **Schedule:** **linear warmup** (critical for transformers/Adam — a few hundred to a few thousand steps)
  then **cosine decay** (or linear decay, or WSD: warmup-stable-decay for flexible run lengths). Warmup
  prevents early instability when Adam's variance estimates are noisy.
- **Batch size:** larger = more stable, more parallel, but with diminishing returns past a "critical batch
  size." Scale LR roughly linearly with batch size (with warmup). Use **gradient accumulation** to simulate
  large batches on limited memory.
- **Gradient clipping** (by global norm, e.g., 1.0) prevents loss spikes — standard for transformers/RNNs.
- **Mixed precision** (bf16 preferred over fp16; fp16 needs loss scaling): ~2× speed/memory, near-free. See
  [engineering-scale.md](engineering-scale.md).

A reliable from-scratch transformer recipe: AdamW, $\beta_2{=}0.95$, weight decay 0.1, gradient clip 1.0,
linear warmup → cosine decay, bf16, the largest batch that fits, LR found by range test. Start there, then ablate.

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
- Val loss noisy/non-monotonic: small val set, high LR, or BatchNorm train/eval mismatch.

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

## 7. Transfer learning is usually the right default

Training from scratch is rarely optimal outside research on the pretraining recipe itself. Start from a
pretrained backbone (ImageNet/DINOv2 for vision, a pretrained LM for text, a foundation model for your
modality) and fine-tune or use PEFT/LoRA. Covered in [representation-learning.md](representation-learning.md).

**Canonical references:** Goodfellow, Bengio & Courville *Deep Learning*; He et al. 2015 (ResNet); Ioffe &
Szegedy 2015 (BatchNorm); Ba et al. 2016 (LayerNorm); Kingma & Ba 2015 (Adam); Loshchilov & Hutter 2019
(AdamW); Karpathy's "A Recipe for Training Neural Networks" (the debugging-ladder ethos).
