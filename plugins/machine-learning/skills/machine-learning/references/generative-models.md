# Generative Models

How to model and sample from $p(\mathbf{x})$ (or $p(\mathbf{x}\mid\mathbf{c})$). Diffusion and flow matching
are the SOTA for continuous data (images, video, audio, molecules, 3D); autoregressive transformers dominate
discrete sequences (see [transformers-llms.md](transformers-llms.md)). Each family trades off **sample
quality, diversity (mode coverage), likelihood, and sampling speed** differently — pick by which you need.

---

## 1. The landscape and the core trade-off

| Family | Exact likelihood | Sample quality | Diversity | Sampling speed | Training |
|---|---|---|---|---|---|
| Autoregressive | ✅ exact | ✅ high | ✅ | ❌ slow (sequential) | stable |
| VAE | ⚠️ ELBO bound | ⚠️ blurry | ✅ | ✅ fast (1 step) | stable |
| Normalizing flow | ✅ exact | ⚠️ medium | ✅ | ✅ fast | stable, arch-constrained |
| GAN | ❌ none | ✅ sharp | ❌ mode collapse | ✅ fast (1 step) | unstable |
| Diffusion / flow matching | ⚠️ bound | ✅✅ SOTA | ✅✅ | ⚠️ multi-step (improving) | stable |

The 2020s arc: GANs (sharp but unstable, poor coverage) → diffusion (stable, SOTA quality + diversity, but
slow sampling) → **flow matching** (same quality, simpler/faster, the current default objective). The
remaining frontier is **few/single-step** sampling (consistency/distillation) to match GAN speed at diffusion
quality.

## 2. Autoregressive models

Factorize $p(\mathbf{x})=\prod_t p(x_t\mid x_{<t})$ and model each conditional (transformer/RNN/CNN). Exact
likelihood, stable training, SOTA for text/code and competitive for images (as discrete tokens via a VQ
tokenizer) and audio. Cost: **sequential sampling** is slow. Discrete diffusion and masked/parallel decoding
are active attempts to parallelize. For text, autoregressive transformers remain dominant.

## 3. Variational Autoencoders (VAEs)

Latent-variable model trained by maximizing the **ELBO**:
$\log p(\mathbf{x}) \ge \mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}[\log p_\theta(\mathbf{x}\mid\mathbf{z})]
- D_\text{KL}(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z}))$ — a reconstruction term minus a latent
regularizer. The **reparameterization trick** ($\mathbf{z}=\mu+\sigma\odot\epsilon$) makes it differentiable.
- **Strengths:** fast 1-step sampling, a usable latent space, stable training, principled.
- **Weaknesses:** blurry samples (Gaussian likelihood averages), posterior collapse (decoder ignores
  $\mathbf{z}$ — mitigate with KL warmup/free-bits/β-VAE), looser quality than diffusion.
- **Where they matter most now:** as the **latent-space encoder for latent diffusion** (Stable Diffusion's VAE),
  VQ-VAE/VQ-GAN tokenizers for discrete generative modeling, and representation learning (β-VAE for
  disentanglement). Rarely the final generator anymore, but everywhere as a component.

## 4. Generative Adversarial Networks (GANs)

A generator and discriminator play a minimax game; the generator learns to fool the discriminator. Produce
**sharp** samples and **fast 1-step** generation.
- **Failure modes:** training instability, **mode collapse** (generator covers few modes), no likelihood, hard
  to evaluate. Mitigations: WGAN-GP / spectral normalization (Lipschitz control), R1 regularization, careful
  architecture (StyleGAN), two-timescale updates.
- **Status (2026):** largely superseded by diffusion/flow for quality + coverage + stability, but still
  relevant for **real-time / single-step** generation, super-resolution, and as a **distillation target/adversarial
  loss** to sharpen few-step diffusion samples. StyleGAN-class models remain strong for narrow domains (faces).

## 5. Normalizing flows

Learn an **invertible** map $f$ between data and a simple base (Gaussian); change-of-variables gives **exact
likelihood**: $\log p(\mathbf{x}) = \log p(\mathbf{z}) + \log|\det J_f|$. Need tractable, invertible layers with
cheap Jacobian determinant (coupling layers: RealNVP/Glow; autoregressive flows: MAF/IAF). Exact density and
fast sampling, but architecturally constrained and historically behind on image quality. **Continuous
normalizing flows** (Neural ODEs) and their training via **flow matching** (below) revived the family and are
now central.

## 6. Diffusion models (the workhorse)

**Idea:** define a forward process that gradually adds Gaussian noise to data over "time" $t\in[0,1]$ until it's
pure noise; train a network to **reverse** it (denoise), then sample by starting from noise and denoising.
- **Training objective:** predict the noise/score. DDPM's simple loss is
  $\mathbb{E}_{t,\mathbf{x}_0,\epsilon}\,\|\epsilon - \epsilon_\theta(\mathbf{x}_t,t)\|^2$ — denoising score
  matching. The network learns the **score** $\nabla_\mathbf{x}\log p_t(\mathbf{x})$ (score-based view, Song &
  Ermon); the forward/reverse can be written as an **SDE** with an equivalent **probability-flow ODE**.
- **Why it works/wins:** stable regression-style training (no adversary), SOTA sample quality *and* mode
  coverage, scales, and conditions easily.
- **Samplers:** ancestral (DDPM, many steps) → **DDIM** (deterministic, fewer steps) → fast ODE/SDE solvers
  (DPM-Solver, Heun, etc.) that cut steps to ~10–30 with little quality loss.
- **Conditioning & guidance:** **classifier-free guidance** (train with and without the condition, extrapolate
  at sampling: $\epsilon = \epsilon_\varnothing + w(\epsilon_c - \epsilon_\varnothing)$) is the standard control
  knob trading diversity for fidelity/prompt-adherence. Cross-attention injects text/conditions.
- **Latent diffusion** (Stable Diffusion): run diffusion in a VAE latent space — far cheaper, the dominant
  recipe for high-res images, video, and audio. Backbone: U-Net historically, increasingly **DiT (Diffusion
  Transformer)** at scale.
- **Beyond images:** video, audio/speech (and TTS), 3D/molecules/proteins, and **discrete diffusion** for text.

## 7. Flow matching (the current default objective)

**Flow matching** (Lipman et al. 2022) trains a continuous normalizing flow by **regressing a velocity field**
that transports the base distribution to the data along a chosen probability path — *without* simulating the ODE
during training. With **conditional/rectified** flows (straight-line interpolant
$\mathbf{x}_t=(1-t)\mathbf{x}_0+t\mathbf{x}_1$), the target velocity is just $\mathbf{x}_1-\mathbf{x}_0$, giving
a dead-simple, stable regression loss.
- **Why it's winning:** simpler and more general than the diffusion derivation, **straighter** trajectories →
  **fewer sampling steps**, strong empirical quality, and a clean theory. Diffusion is recoverable as a
  *special case* (a particular noise schedule/path) under the unifying **Generator Matching / stochastic
  interpolants** framework. Many 2025–2026 image/video/audio systems are flow-matching-based.
- **Practical note:** flow matching and diffusion are close cousins; you can convert between them (e.g.,
  Diff2Flow). Default to a **flow-matching (rectified/conditional) objective** for new continuous generative
  work; reach for diffusion-specific machinery when you need its mature tooling/guidance.

## 8. Fast sampling: consistency & distillation

The cost of diffusion/flow is iterative sampling. The frontier compresses it:
- **Consistency models** (Song et al. 2023) learn to map any point on a trajectory directly to its origin,
  enabling **1–4 step** generation; trainable from scratch or distilled.
- **Distillation:** progressive distillation, adversarial/score distillation (e.g., turning a multi-step model
  into a few-step generator with a GAN-style loss). 2025–2026 work pushes high-quality **single-step** image/
  video/audio generation. When latency matters, distill.

## 9. Conditioning, control, and editing

- **Conditioning mechanisms:** cross-attention (text), concatenation, FiLM/adaptive norm, ControlNet (spatial
  conditions: edges, pose, depth), adapters/LoRA for cheap customization.
- **Inversion & editing:** DDIM inversion, prompt-to-prompt, and latent editing for image manipulation.
- **Personalization:** DreamBooth / textual inversion / LoRA fine-tunes for subject/style.

## 10. Evaluating generative models (hard — be careful)

There is **no single good metric**; report several and show samples honestly.
- **Images:** **FID** (Fréchet distance between Inception features of real vs. generated — lower better; the de
  facto standard but sensitive to the feature extractor, sample count, and preprocessing — **report all three**).
  **Inception Score** is largely deprecated. **Precision/Recall** and **Density/Coverage** separately measure
  fidelity vs. diversity (FID conflates them). **CLIPScore** for text-image alignment. Newer features (DINOv2-
  based FD) reduce Inception's biases.
- **Likelihood models:** bits-per-dimension (only comparable within the same data/preprocessing).
- **Text:** perplexity (within identical tokenizer), plus task/quality evals (see
  [evaluation-statistics.md](evaluation-statistics.md)); n-gram metrics (BLEU/ROUGE) are weak for open-ended
  generation. Diversity vs. quality trade-off (e.g., via temperature) must be reported jointly.
- **Audio/speech:** FAD (Fréchet Audio Distance), plus MOS/human eval; intelligibility (WER via ASR) for TTS.
- **Always:** include **human evaluation** for perceptual quality (with proper protocol — multiple raters,
  randomized, blind), check for **memorization/copying of training data** (a real risk and a legal/ethical one),
  and never cherry-pick samples as evidence. Report mean ± CI over multiple generation seeds.

## 11. Reach-for table

| Goal | Reach for |
|---|---|
| SOTA image/video/audio/3D generation | Latent diffusion or flow matching (DiT backbone) |
| New continuous generative model from scratch | Conditional/rectified **flow matching** |
| Discrete sequences (text/code) | Autoregressive transformer |
| Real-time / single-step generation | Consistency model or distilled few-step; GAN for narrow domains |
| Exact likelihood / density estimation | Normalizing flow or autoregressive |
| Latent space / tokenizer / representation | VAE / VQ-VAE |
| Text-conditioned control | Classifier-free guidance + cross-attention (+ ControlNet for spatial) |

**Canonical references:** Kingma & Welling 2013 (VAE); Goodfellow et al. 2014 (GAN); Rezende & Mohamed 2015,
Dinh et al. 2016 (flows); Ho et al. 2020 (DDPM); Song et al. 2020–21 (score-based SDE, DDIM); Rombach et al.
2022 (latent diffusion); Lipman et al. 2022 (flow matching); Liu et al. 2022 (rectified flow); Song et al. 2023
(consistency models); Peebles & Xie 2022 (DiT).
