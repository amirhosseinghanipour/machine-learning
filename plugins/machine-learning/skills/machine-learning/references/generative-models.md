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

**Unifying view (load this before arguing about which family is "best").** Diffusion, flow matching, score
matching, and stochastic interpolants are *the same object viewed through different time parameterizations and
noise schedules*. **Stochastic interpolants** (Albergo & Vanden-Eijnden 2023) and **Generator Matching**
(Holderrieth et al. 2024, ICLR 2025) make this precise: pick a probability path connecting noise $p_0$ to data
$p_1$, regress the generator (a velocity field for ODE transport, or a score/rate for SDE/jump processes) that
realizes it. DDPM is one schedule, rectified flow another; both admit a deterministic **probability-flow ODE**
and a family of equivalent SDEs at varying noise injection. Practically: design in the EDM/flow-matching
parameterization (clean signal/velocity prediction, sane preconditioning), then choose the sampler (ODE vs SDE,
step count) independently of how you trained. Generator Matching also explains *why* flow matching tends to be
more robust than ε-prediction diffusion: the regression target has bounded magnitude across all $t$, whereas
ε-prediction targets blow up as $t\to 0$.

## 2. Autoregressive models

Factorize $p(\mathbf{x})=\prod_t p(x_t\mid x_{<t})$ and model each conditional (transformer/RNN/CNN). Exact
likelihood, stable training, SOTA for text/code and competitive for images (as discrete tokens via a VQ
tokenizer) and audio. Cost: **sequential sampling** is slow. Discrete diffusion and masked/parallel decoding
are active attempts to parallelize. For text, autoregressive transformers remain dominant.
- **Order matters for images.** Raster-order next-token prediction on VQ tokens is weak; **next-scale
  prediction** (VAR, Tian et al. 2024, NeurIPS best-paper) predicts a coarse-to-fine pyramid of token maps and
  beat diffusion on ImageNet-256 (FID ≈1.8) with better scaling-law behavior — the strongest case that AR image
  models are competitive. **MAR** (Li et al. 2024) drops vector quantization entirely: model continuous tokens
  with a small per-token **diffusion loss** head, getting AR's flexibility without VQ's reconstruction ceiling.
- **The VQ bottleneck is the real cost.** A discrete tokenizer caps fidelity at its reconstruction quality;
  codebook collapse and low utilization are chronic. Mitigations: large codebooks with low-dim/$\ell_2$-normalized
  codes (ViT-VQGAN), **FSQ** (finite scalar quantization — no learned codebook, no collapse, simpler), or
  lookup-free quantization. If quality is the priority and AR isn't required, continuous latent diffusion still
  wins.

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
- **Defaults that matter for the LDM autoencoder (the failure point people skip).** A continuous KL-VAE for
  diffusion is trained with a **tiny KL weight** ($\sim10^{-6}$, so it is mostly an autoencoder), an LPIPS
  perceptual loss, and a **patch GAN** adversarial term to avoid blur. Crucially, its **latent variance is
  normalized to ≈1** (SD uses a fixed scale factor 0.18215) before diffusion trains on it — get this wrong and
  the noise schedule is silently mismatched. Higher latent channel count ($f8$/4ch → 16ch in SD3/FLUX) raises
  the reconstruction ceiling at the cost of a harder diffusion problem. The ELBO/likelihood is irrelevant here;
  reconstruction FID and rFID are what you tune.
- **Posterior-collapse diagnosis:** if KL per dim ≈ 0 and reconstructions ignore $\mathbf{z}$, the decoder is too
  strong relative to the KL pressure. Free-bits (floor the KL per dim at $\lambda$ nats), KL annealing, or a
  weaker/autoregressive-free decoder fix it; β-VAE with $\beta>1$ trades reconstruction for disentanglement, not
  the other way around.

## 4. Generative Adversarial Networks (GANs)

A generator and discriminator play a minimax game; the generator learns to fool the discriminator. Produce
**sharp** samples and **fast 1-step** generation.
- **Failure modes:** training instability, **mode collapse** (generator covers few modes), no likelihood, hard
  to evaluate. Mitigations: WGAN-GP / spectral normalization (Lipschitz control), R1 regularization, careful
  architecture (StyleGAN), two-timescale updates.
- **Status (2026):** largely superseded by diffusion/flow for quality + coverage + stability, but still
  relevant for **real-time / single-step** generation, super-resolution, and as a **distillation target/adversarial
  loss** to sharpen few-step diffusion samples. StyleGAN-class models remain strong for narrow domains (faces).
- **The quiet comeback as a loss, not a model.** The highest-impact GAN role in 2025–2026 is the **adversarial
  term in few/one-step distillation**: ADD/LADD (SDXL-Turbo), DMD2, and the GAN head in latent-consistency
  variants all use a discriminator to recover the sharpness that pure score/consistency distillation loses at
  1–4 steps. **R3GAN** (Huang et al. 2024) also showed a modernized, regularized minimax objective (RpGAN +
  R1+R2 gradient penalties) trains stably without the StyleGAN bag of tricks — evidence the instability was
  loss/regularization design, not GANs per se. **Diagnose mode collapse** with recall/coverage (§10), not FID
  alone — FID can look fine while diversity is gone.

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
- **What the network predicts (this choice matters more than the architecture).** Equivalent up to weighting:
  **ε-prediction** (DDPM default, but the target diverges at low noise → unstable near $t\to0$); **x0** (clean
  signal, good at low noise, poor at high noise); **v-prediction** $v=\alpha_t\epsilon-\sigma_t\mathbf{x}_0$
  (Salimans & Ho — well-behaved across all $t$, the right default for the cosine schedule and a prerequisite for
  good distillation); and **velocity for flow matching** ($\mathbf{x}_1-\mathbf{x}_0$, see §7). For pixel-space
  or high-res, prefer **v** or flow-matching velocity over raw ε.
- **EDM parameterization (the recommended training/sampling recipe; Karras et al. 2022, "Elucidating…").** Work
  in noise level $\sigma$ directly, with **preconditioning** so the network sees standardized in/outputs at every
  $\sigma$: $D_\theta(\mathbf{x};\sigma)=c_\text{skip}(\sigma)\mathbf{x}+c_\text{out}(\sigma)F_\theta(c_\text{in}(\sigma)\mathbf{x};\,c_\text{noise}(\sigma))$.
  Concrete defaults: $\sigma_\text{data}=0.5$, $\sigma_\text{min}=0.002$, $\sigma_\text{max}=80$, train-noise
  $\ln\sigma\sim\mathcal N(P_\text{mean}{=}{-}1.2,\,P_\text{std}{=}1.2)$, loss weight $\lambda(\sigma)=(\sigma^2+\sigma_\text{data}^2)/(\sigma\,\sigma_\text{data})^2$,
  $c_\text{noise}=\tfrac14\ln\sigma$. Sampling: a "karras" $\rho{=}7$ $\sigma$-schedule with **Heun (2nd-order)**
  reaches SOTA FID in ~**35 NFE** (18 steps). This recipe is the de-facto baseline; start here, not from raw DDPM.
- **EDM2 (Karras et al. CVPR 2024):** redesign layers to be **magnitude-preserving** (forced unit activation/weight
  norms, no learned scale drift), plus **post-hoc EMA** — sweep the EMA half-life *after* training from a few
  stored snapshots instead of guessing it up front (EMA length is one of the most impactful and most overlooked
  knobs; optimal length interacts with guidance). Set ImageNet-512 records at far lower cost.
- **Why it works/wins:** stable regression-style training (no adversary), SOTA sample quality *and* mode
  coverage, scales, and conditions easily.
- **Samplers (decoupled from training; pick by NFE budget):** ancestral DDPM (many steps, stochastic) → **DDIM**
  (deterministic, ~50 steps) → **DPM-Solver++(2M)** / **UniPC** (the practical defaults: ~15–20 and ~5–10 steps
  respectively; enable `lower_order_final`/`thresholding` for guided sampling) → **Heun/EDM** for highest quality
  at moderate NFE. Multistep (Adams-style) solvers reuse past evaluations and beat single-step at equal NFE.
  Rule of thumb: ODE/deterministic solvers for few-step and reproducibility; add SDE/stochasticity (EDM
  $S_\text{churn}$) only when you have steps to spare and want a small quality/diversity bump.
- **Conditioning & guidance:** **classifier-free guidance** (train with the condition dropped ~10–20% of the
  time, extrapolate at sampling: $\tilde\epsilon = \epsilon_\varnothing + w(\epsilon_c - \epsilon_\varnothing)$,
  typically $w\in[3,8]$ for text-to-image) is the standard control knob trading diversity for fidelity/prompt
  adherence. **Pitfalls and fixes (2024):** high $w$ oversaturates colors and collapses diversity — **apply
  guidance only on a middle noise interval** (Kynkäänniemi et al., NeurIPS 2024: off at the highest and lowest
  $\sigma$) to keep crispness without the artifacts; **autoguidance** (Karras et al. 2024) guides with a
  *smaller/less-trained* copy of the *same* model instead of the unconditional, improving FID *and* alignment;
  **CFG-Zero⋆** fixes flow-matching guidance at small $t$. Cross-attention injects text/conditions; classifier
  guidance (the original, needing a noisy classifier) is largely obsolete.
- **Latent diffusion** (Stable Diffusion): run diffusion in a VAE latent space — far cheaper, the dominant
  recipe for high-res images, video, and audio. Backbone: U-Net historically, increasingly **DiT (Diffusion
  Transformer)** at scale — DiT scales cleanly with compute (FLOPs ↔ FID), uses **adaLN-zero** to inject
  timestep/class, and is the backbone of SD3/FLUX/Sora-class systems. Resolution scaling needs schedule
  fixes: **SNR/timestep shift** (sample more high-noise steps as resolution grows; SD3's `shift`) or a
  zero-terminal-SNR schedule, else high-res images come out hazy/low-contrast.
- **Beyond images:** video, audio/speech (and TTS), 3D/molecules/proteins, and **discrete diffusion** for text —
  now scaled to LLM size: **SEDD** (score-entropy/ratio matching), **MDLM/MD4** (simplified masked/absorbing
  diffusion, a Rao-Blackwellized MLM-style loss), and **LLaDA/Dream** (8B-scale diffusion LLMs competitive with
  AR at similar scale, with parallel/any-order decoding). Still behind AR on raw perplexity-per-FLOP but
  attractive for parallel generation and infilling. See [transformers-llms.md](transformers-llms.md).

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
- **Defaults that actually move FID.** (1) **Timestep sampling is not uniform** in the systems that win: SD3
  and Movie Gen sample $t$ from a **logit-normal** distribution (concentrating training on the hard middle of the
  path), a larger effect than most architecture tweaks. (2) **Resolution/SNR shift** — at higher resolution,
  shift the schedule toward higher noise (SD3's $\text{shift}$ parameter); skipping this is the usual cause of
  washed-out high-res samples. (3) **Minibatch optimal-transport coupling** (pair noise↔data within the batch by
  OT instead of independently) straightens paths further and cuts NFE; cheap and worth it. (4) Sampling is just
  an ODE solve of the velocity — Euler is fine at moderate steps; rectified flow gets straight enough that
  **few-step Euler** is viable, and **reflow** (retrain on the model's own (noise, sample) pairs) straightens it
  toward 1-step.
- **Practical note:** flow matching and diffusion are close cousins; you can convert between them (e.g.,
  Diff2Flow). Default to a **flow-matching (rectified/conditional) objective with logit-normal $t$ and v/velocity
  parameterization** for new continuous generative work; reach for diffusion-specific machinery when you need its
  mature tooling/guidance. **Discrete and multimodal flow matching** (Discrete FM, generator matching on
  arbitrary state spaces) now extend the same recipe to text/graphs/proteins.

## 8. Fast sampling: consistency & distillation

The cost of diffusion/flow is iterative sampling. The frontier compresses it. Two families: **distillation**
(teacher → fast student) and **few-step training from scratch**.
- **Consistency models** (Song et al. 2023) learn a function that maps any point on a PF-ODE trajectory directly
  to its origin (self-consistency), enabling **1–4 step** generation; trainable from scratch (CT) or distilled
  (CD). Discrete-time CMs are finicky; **improved techniques** (Song & Dhariwal 2024) and especially the
  **continuous-time formulation sCM / TrigFlow** (Lu & Song 2024) fix the instability and close most of the gap
  to the teacher — sCM reaches ~10% of teacher FLOPs with FID within a few % of the teacher at 2 steps, and is
  the current go-to for principled few-step.
- **Variants worth knowing:** **LCM/LCM-LoRA** (consistency distillation in latent space; LoRA makes it a
  drop-in 4-step accelerator for any SD/SDXL checkpoint, the practical default for "make my model fast"),
  **TCD** (trajectory consistency, better at 4–8 steps), **CTM** (consistency-trajectory, unifies CM + score,
  SOTA few-step FID), and **multistep CM** (interpolates CM↔diffusion: trade 1→8 steps for quality).
- **Distillation without consistency:** **progressive distillation** (halve steps repeatedly), **DMD/DMD2**
  (distribution matching — match the student's output distribution to the teacher's score, + a GAN term),
  **ADD/LADD** (adversarial diffusion distillation — SDXL-Turbo/SD3-Turbo, 1–4 step), and **score distillation**
  (the SDS family, also the engine behind text-to-3D). 2025–2026 work delivers high-quality **single-step**
  image/video/audio. **Pitfall:** one-step distillation reliably loses **diversity/recall** (mode coverage) even
  when FID looks competitive — measure recall/coverage, not just FID, and keep ≥2 steps if diversity matters.
  When latency matters, distill; when quality/diversity matters most, keep the multi-step teacher.

## 9. Conditioning, control, and editing

- **Conditioning mechanisms:** cross-attention (text), concatenation, FiLM/adaptive norm, ControlNet (spatial
  conditions: edges, pose, depth), adapters/LoRA for cheap customization.
- **Inversion & editing:** DDIM inversion, prompt-to-prompt, and latent editing for image manipulation.
- **Personalization:** DreamBooth / textual inversion / LoRA fine-tunes for subject/style.

## 10. Evaluating generative models (hard — be careful)

There is **no single good metric**; report several and show samples honestly.
- **Images:** **FID** (Fréchet distance between Inception-V3 features of real vs. generated — lower better; the
  de facto standard but **deeply flawed**, so treat it as one signal among several). Hard rules: it is biased
  for small $N$ (use **≥50k** samples and report $N$), assumes Gaussian features (false), and is sensitive to
  the **exact** resize/crop/JPEG/dtype pipeline (mismatched preprocessing alone shifts FID by points — match the
  reference pipeline bit-for-bit and report it). Inception features over-weight ImageNet-class texture and
  correlate weakly with human judgment.
  - **Better feature extractors:** **FD-DINOv2** (Stein et al. 2023) correlates far better with human preference
    and is the recommended replacement/companion to FID; report it for any 2025+ work. **CMMD** (CLIP features +
    MMD, Jayasumana et al. 2024) is **unbiased, sample-efficient** (works with a few thousand images), and
    distribution-free — prefer it for in-training monitoring and low-data regimes where FID is unreliable.
  - **Decompose fidelity vs diversity:** **Precision/Recall** and **Density/Coverage** (FID conflates them; a
    mode-collapsed or memorizing model can post a good FID). Recall/coverage is how you catch the diversity loss
    that distillation and high CFG cause.
  - **Catch memorization/overfitting:** FID is blind to copying the training set. **Feature Likelihood Divergence
    (FLD)** explicitly penalizes memorization; also run nearest-neighbor checks against training data.
  - **Text–image alignment:** **CLIPScore** is the cheap default but saturates and is gameable; **human/learned
    preference** models (PickScore, ImageReward, HPSv2) and VQA-based metrics track prompt adherence better.
    **Inception Score is deprecated** (no reference distribution; ImageNet-bound).
- **Likelihood models:** bits-per-dimension (only comparable within the same data/preprocessing). Note diffusion
  "likelihoods" are ELBO bounds and **not comparable** to exact-likelihood (flow/AR) numbers.
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
| New continuous generative model from scratch | Conditional/rectified **flow matching** (logit-normal $t$, v/velocity, EDM preconditioning) |
| Discrete sequences (text/code) | Autoregressive transformer |
| Real-time / single-step generation | sCM/consistency or ADD/DMD2 distillation; LCM-LoRA to accelerate an existing SD model |
| Exact likelihood / density estimation | Normalizing flow or autoregressive |
| Latent space / tokenizer / representation | VAE (continuous) / VQ-VAE or **FSQ** (discrete, no codebook collapse) |
| Text-conditioned control | Classifier-free guidance (interval-limited) + cross-attention (+ ControlNet for spatial) |
| Reliable in-training / low-sample eval | **CMMD** (unbiased) and **FD-DINOv2**, plus recall/coverage — not FID alone |

**Canonical references:** Kingma & Welling 2013 (VAE); Goodfellow et al. 2014 (GAN); Rezende & Mohamed 2015,
Dinh et al. 2016 (flows); Ho et al. 2020 (DDPM); Song et al. 2020–21 (score-based SDE, DDIM); **Karras et al.
2022 (EDM — preconditioning/sampler defaults)** and Karras et al. 2024 (EDM2, post-hoc EMA, autoguidance);
Salimans & Ho 2022 (v-prediction, progressive distillation); Rombach et al. 2022 (latent diffusion); Peebles &
Xie 2022 (DiT); Lu et al. 2022 (DPM-Solver++); Zhao et al. 2023 (UniPC); Lipman et al. 2022 (flow matching);
Liu et al. 2022 (rectified flow); Albergo & Vanden-Eijnden 2023 (stochastic interpolants); **Holderrieth et al.
2024 (Generator Matching)**; Esser et al. 2024 (SD3 — rectified-flow transformer, logit-normal $t$); Song et al.
2023 + Lu & Song 2024 (consistency models, sCM/TrigFlow); Tian et al. 2024 (VAR), Li et al. 2024 (MAR);
Stein et al. 2023 (FD-DINOv2), Jayasumana et al. 2024 (CMMD); Kynkäänniemi et al. 2024 (interval guidance).
**Texts:** Murphy, *Probabilistic Machine Learning: Advanced Topics* (vol. 2); Tomczak, *Deep Generative
Modeling* (2nd ed.).
