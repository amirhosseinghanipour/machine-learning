# Foundations: Math, Optimization Theory, and Learning Theory

The minimum theory that changes what you *do* in practice. Skip the parts you know; the goal is
judgment, not a textbook. Notation: scalars $x$, vectors $\mathbf{x}$, matrices $X$, expectation
$\mathbb{E}$, parameters $\theta$, loss $L$, data distribution $\mathcal{D}$.

---

## 1. Linear algebra you actually use

- **Everything is a matvec.** A linear layer is $W\mathbf{x}+\mathbf{b}$; a conv is a structured
  (Toeplitz/banded, weight-shared) matvec; attention is $\text{softmax}(QK^\top/\sqrt{d})V$. Thinking in
  matvecs makes shapes, FLOPs, and memory obvious. FLOPs of a dense layer $\approx 2\,d_\text{in}d_\text{out}$
  per token; of a transformer block, dominated by $O(L^2 d)$ attention + $O(L d^2)$ MLP.
- **SVD is the master decomposition.** $X = U\Sigma V^\top$. PCA = SVD of centered data; low-rank
  approximation (Eckart–Young: truncated SVD is the optimal rank-$k$ approximation in Frobenius/spectral
  norm); LoRA = learned low-rank update; the condition number $\sigma_\max/\sigma_\min$ governs
  optimization difficulty and numerical stability.
- **Eigen-intuition for training.** Near a minimum, loss $\approx \tfrac12(\theta-\theta^*)^\top H(\theta-\theta^*)$.
  The Hessian $H$'s eigenvalues set the local geometry: gradient descent's stable step size is
  $\eta < 2/\lambda_\max$; convergence speed is governed by the condition number $\lambda_\max/\lambda_\min$.
  Ill-conditioning (a few huge eigenvalues) is *why* adaptive optimizers and normalization help. The
  deep-net Hessian spectrum is empirically **bulk-plus-outliers**: a dense bulk near zero plus a few large
  outlier eigenvalues roughly equal to the number of classes — most directions are flat, a handful are
  sharp. This is why a single global LR is a poor fit and why preconditioning (Adam, Shampoo) pays off.
- **Norms encode priors.** $\ell_2$ (ridge) shrinks smoothly; $\ell_1$ (lasso) induces sparsity (corners of
  the ball); nuclear norm induces low rank; spectral norm bounds Lipschitz constant (used for stability/GAN
  regularization). Choosing the penalty = choosing the prior.
- **Matrix calculus.** Know $\nabla_\mathbf{x}(\mathbf{a}^\top\mathbf{x})=\mathbf{a}$,
  $\nabla_\mathbf{x}(\mathbf{x}^\top A\mathbf{x})=(A+A^\top)\mathbf{x}$, and the chain rule for Jacobians.
  Autodiff handles the rest, but you need this to derive losses and sanity-check gradients.

## 2. Probability & statistics that inform modeling

- **The likelihood view.** A model defines $p_\theta(y\mid \mathbf{x})$. Training by minimizing
  cross-entropy/MSE *is* maximum likelihood. Regularization = a prior → MAP estimation. This single frame
  unifies most of supervised learning (see [classical-ml.md](classical-ml.md), [probabilistic-ml.md](probabilistic-ml.md)).
- **Bias–variance decomposition.** For squared loss, $\mathbb{E}[(y-\hat f)^2] = \text{Bias}^2 + \text{Var} +
  \sigma^2_\text{noise}$. The irreducible noise floor $\sigma^2$ caps any model. Diagnose which term dominates
  (high train+test error = bias; low train, high test = variance) before choosing a fix.
- **Concentration.** Sample means concentrate at rate $O(1/\sqrt{n})$ (CLT); this is why a 1,000-example
  test set gives a metric with std on the order of a few percent — small leaderboard gaps need significance
  tests. The toolbox: **Hoeffding** (bounded variables, gap $\le\sqrt{\log(2/\delta)/2n}$ — distribution-free
  but loose), **Bernstein/empirical-Bernstein** (variance-adaptive, much tighter when the metric's variance is
  small, e.g. near-0/1 accuracy), **McDiarmid/bounded-differences** (functions where any one sample changes the
  output little — the engine behind Rademacher generalization bounds), and **sub-Gaussian/sub-exponential** tail
  control for unbounded losses. A concrete sizing rule: to resolve a true accuracy difference $\Delta$ on a
  binary metric you need roughly $n\gtrsim p(1-p)\,(z_{1-\delta}/\Delta)^2$ test examples — to call a 1% gap at
  95% confidence around $p\approx0.9$ takes on the order of $10^3$–$10^4$ examples. See
  [evaluation-statistics.md](evaluation-statistics.md).
- **Distributions to know cold.** Gaussian (MSE, central limit, reparameterization), Bernoulli/Categorical
  (classification, cross-entropy), Beta/Dirichlet (conjugate priors, smoothing), Poisson (counts),
  Exponential family (the unifying form; natural gradients live here).
- **Estimators.** MLE is consistent and asymptotically efficient (attains the Cramér–Rao bound, with
  asymptotic covariance the inverse Fisher information $I(\theta)^{-1}$) but can overfit in finite samples and
  high dimensions; MAP adds a prior (= regularized MLE); the full Bayesian posterior quantifies uncertainty
  instead of collapsing to a point. Bias of an estimator vs. variance is the same trade-off as above —
  Stein's paradox shows that even for estimating a Gaussian mean in $\ge 3$ dimensions, the (unbiased) MLE is
  *inadmissible*: shrinkage (James–Stein, ridge) strictly dominates it. Regularization is not a hack; it is
  often provably better than the unbiased estimate.
- **Common statistical traps:** confusing $p(y\mid x)$ with $p(x\mid y)$ (base-rate / Simpson's paradox),
  ignoring that the **max** of many noisy scores is biased high (selection), and treating correlated samples
  as independent (inflates significance — see [evaluation-statistics.md](evaluation-statistics.md)).

## 3. Information theory

- **Entropy** $H(p)=-\sum p\log p$ = bits to encode samples from $p$; **cross-entropy**
  $H(p,q)=-\sum p\log q$ = bits when you use code $q$ for true $p$; **KL** $D_\text{KL}(p\|q)=H(p,q)-H(p)\ge 0$
  = excess bits. Minimizing cross-entropy loss = minimizing $D_\text{KL}(p_\text{data}\|p_\theta)$ (forward KL,
  mass-covering). Reverse KL (mode-seeking) shows up in variational inference and RLHF/DPO regularization.
- **Mutual information** $I(X;Y)=H(X)-H(X\mid Y)$ underlies contrastive learning (InfoNCE is an MI lower
  bound), the information bottleneck, and feature selection. MI is hard to estimate in high dimensions — treat
  neural MI estimates as bounds, not ground truth.
- **MDL / Occam.** The best model is the one that most compresses the data (model bits + residual bits).
  A useful prior for why simpler models that fit generalize better.
- **Perplexity** $=\exp(\text{cross-entropy})$ is the standard LM metric; it's the effective vocabulary
  branching factor. Lower is better but only comparable across identical tokenization — perplexity per *token*
  is not comparable across tokenizers, so report bits-per-byte (BPB) for cross-tokenizer comparisons.
- **Divergences beyond KL — pick deliberately.** KL is one member of the **f-divergence** family
  ($D_f(p\|q)=\mathbb{E}_q[f(p/q)]$): also $\chi^2$, total variation, Hellinger, Jensen–Shannon (the GAN
  divergence). f-divergences are blind to geometry — they ignore *how far apart* the supports are. **Integral
  probability metrics** (IPMs: Wasserstein/earth-mover, MMD, total variation) instead measure transport cost
  and stay finite and informative even for disjoint supports, which is exactly why Wasserstein (WGAN) and MMD
  fixed the vanishing-gradient pathology of JS-based GANs. Rule of thumb: use forward KL when you have samples
  from $p$ and want a density model (MLE), reverse KL for mode-seeking variational fits, and an IPM when
  comparing distributions with possibly disjoint support or when you need a true metric. See
  [generative-models.md](generative-models.md).

## 4. Optimization — the engine of ML

**Convex vs. non-convex.** Convex problems (linear/logistic regression, SVMs) have a unique global minimum
and reliable solvers. Deep nets are non-convex, but in the overparameterized regime most local minima are
nearly equivalent and SGD reliably finds good ones — the landscape is benign in practice. Don't import convex
intuitions ("we're stuck in a bad local minimum") uncritically; the usual failure is bad conditioning, dead
units, or wrong learning rate, not a literal bad minimum.

**Gradient descent and friends** (full treatment of the practical optimizers — Adam(W), schedules, warmup —
in [deep-learning.md](deep-learning.md)):
- **GD/SGD.** $\theta \leftarrow \theta - \eta\,\nabla L$. SGD's gradient noise is a *feature*: it regularizes
  and helps escape saddles. Batch size and learning rate trade off; the linear scaling rule
  ($\eta \propto$ batch size, with warmup) holds over a useful range.
- **Momentum** accumulates a velocity, accelerating along consistent directions and damping oscillation in
  high-curvature ones — effectively improving conditioning. Nesterov adds a lookahead.
- **Adaptive methods (Adam/AdamW).** Per-parameter step sizes from running gradient moments; robust to bad
  conditioning and the default for transformers. AdamW (decoupled weight decay) is the standard.
- **Second-order & natural gradient.** Newton ($H^{-1}\nabla$), K-FAC, Shampoo approximate curvature for faster
  convergence; natural gradient uses the Fisher information metric (the right geometry for probability models,
  and the basis of TRPO/PPO's trust region — see [reinforcement-learning.md](reinforcement-learning.md)).
- **Matrix-aware optimizers (the 2025 shift).** **Muon** (Jordan et al., 2024–25) applies a few Newton–Schulz
  iterations to orthogonalize the momentum update of each 2D weight matrix toward $UV^\top$ — a cheap
  spectral-norm-controlled step that empirically trains LLMs ~2× faster than AdamW at <1% overhead, and set
  speedrun records / scaled to frontier runs (e.g. Kimi K2). It is applied only to hidden matrices; embeddings,
  unembeddings, biases, and scalars stay on Adam. **Shampoo** and its distributed variant **SOAP** are the
  other practical curvature-aware family. The unifying idea — *condition the update by matrix structure, not
  just per-coordinate variance* — is the most consequential optimizer trend since AdamW. Full practical
  treatment in [deep-learning.md](deep-learning.md).
- **Convergence intuition.** Strongly convex + smooth → linear rate; convex → $O(1/t)$ (or $O(1/t^2)$ with
  acceleration); non-convex → you get convergence to a stationary point in expectation, nothing global. Step
  size must respect smoothness ($\eta \lesssim 1/L$); too large diverges, too small crawls.
- **Constrained / proximal.** $\ell_1$ via proximal/soft-thresholding; projected gradient for constraints;
  mirror descent for simplex/entropy geometry. Relevant for structured/regularized objectives.

**Saddle points, not minima,** are the obstacle in high dimensions (minima are exponentially rarer). SGD
noise and momentum handle them. **Sharp vs. flat minima:** flatter minima tend to generalize better; small
batch / higher LR / SAM (Sharpness-Aware Minimization) bias toward flatness. (Caveat: sharpness is not
reparameterization-invariant, so "flat = generalizes" is a useful heuristic, not a theorem — adaptive/relative
flatness measures are the more defensible version.)

**Edge of Stability (EoS) — the textbook step-size rule is violated on purpose.** Classical theory says
full-batch GD diverges once sharpness $\lambda_\max(H) > 2/\eta$. In real deep-net training (Cohen et al.,
2021) sharpness instead *rises until it hits $2/\eta$ and then hovers there* (the "edge of stability") while
the loss decreases non-monotonically. The practical consequences: (1) the largest usable LR is set by this
self-stabilization, not by the local Hessian at init, so LR-range tests work; (2) larger LR pushes training to
a *flatter* (lower-sharpness) region — part of why higher LR generalizes better; (3) loss spikes during stable
training are often EoS oscillation, not divergence — don't reflexively cut the LR. Adaptive optimizers show an
analogous "adaptive edge of stability."

**Practical optimization debugging order:** (1) Is the loss even decreasing on one batch? (2) Is the LR in a
sane range (loss explodes = too high; barely moves = too low — do an LR-range test)? (3) Gradient norms
finite and non-zero (no NaN/Inf, no dead ReLUs)? (4) Is the data/label pipeline correct? Most "optimization
problems" are data or LR problems.

## 5. Statistical learning theory — why generalization is possible

You don't need proofs, but these results shape intuition:

- **Empirical risk minimization (ERM).** We minimize training loss as a proxy for true risk
  $R(\theta)=\mathbb{E}_{(\mathbf{x},y)\sim\mathcal{D}}[L]$. Generalization gap = test − train risk. Theory
  bounds this gap.
- **Capacity controls the gap (classical view).** Uniform-convergence bounds (VC dimension, Rademacher
  complexity, covering numbers) say gap $\lesssim \sqrt{\text{capacity}/n}$. **Rademacher complexity**
  $\mathfrak{R}_n(\mathcal{F})=\mathbb{E}_\sigma\sup_{f}\frac1n\sum_i\sigma_i f(x_i)$ is the modern workhorse:
  it is data-dependent (measures how well the class fits *random* $\pm1$ labels — pure overfitting capacity),
  composes nicely (contraction lemma passes it through Lipschitz losses), and recovers margin bounds. The key
  practical reading: what controls the gap for linear/kernel/SVM models is **norm, not parameter count** — e.g.
  margin-normalized bounds scale like $\|\mathbf{w}\|\,R/(\gamma\sqrt{n})$, independent of dimension. This is
  why $\ell_2$ regularization and large margins generalize, and the seed of why overparameterization need not hurt.
- **The modern puzzle and its resolutions.** Deep nets can memorize random labels (Zhang et al., 2017), so
  worst-case VC/parameter-count bounds are *vacuous* for them. Three threads make generalization predictable
  again: (1) **Norm-/margin-based and compression bounds** — the function the optimizer actually finds is
  low-complexity even though the class is huge. **PAC-Bayes** gives the only *non-vacuous* bounds for real
  networks: Dziugaite & Roy (2017) on MNIST, then Lotfi et al. (2022–24) via compression, reaching non-vacuous
  certificates even for LLMs at the token level. A PhD-relevant fact: a tight PAC-Bayes/compression bound
  *certifies* generalization without a test set — useful when you genuinely cannot hold data out. (2)
  **Implicit regularization** — GD/SGD on separable data converges to the **max-margin** solution (logistic
  loss: Soudry et al.; linear nets: toward low nuclear norm / low rank), and label noise + finite LR biases
  toward flatter minima. The optimizer, not just the objective, picks the solution. (3) **The data
  distribution does the work** — bounds that ignore the input distribution must be loose; benign overfitting
  (below) is fundamentally a statement about the data covariance spectrum.
- **Double descent and benign overfitting.** Test error can *decrease again* past the interpolation threshold,
  so "bigger overfits more" is false in the overparameterized regime. The crisp theory is **benign overfitting
  in linear regression** (Bartlett, Long, Lugosi, Tsigler, 2020; refined for ridge by Tsigler & Bartlett,
  2024): the minimum-norm interpolator generalizes *iff* the data covariance has a specific spectral profile —
  a few high-energy directions carry the signal while a long tail of low-energy directions absorbs (and
  effectively averages out) the noise. Overfitting is benign exactly when the effective rank is large enough to
  dilute noise but the top eigenvalues still capture signal; "tempered" and "catastrophic" regimes sit on the
  other side. Practical upshot: interpolation is safe when your features are high-dimensional with a heavy
  spectral tail (typical of overparameterized nets / kernels), and dangerous otherwise. Double descent appears
  in model size, data size (sometimes *more data hurts* near the threshold), and training time (epoch-wise).
- **NTK vs. feature learning — know the limit you're invoking.** In the infinite-width **lazy/NTK** limit a net
  behaves like a *fixed* kernel: weights barely move, training is convex kernel regression, and it cannot learn
  new features. This explains optimization (why wide nets train to zero loss) but is now understood to *not*
  explain the performance of real nets — it requires unrealistic widths and, by construction, forbids the
  feature learning that drives transfer and emergence. The **mean-field / $\mu$P (maximal-update)** limit keeps
  features moving as width grows; $\mu$P is also the practical payoff (HP transfer across width — tune small,
  scale up). Treat NTK as a tractable analysis regime, not a model of how SGD actually learns representations.
- **Grokking.** Delayed generalization: train accuracy saturates early, test accuracy jumps much later. It is
  the implicit-regularization story made visible — the network first memorizes, then weight decay (or the slow
  drift to a lower-norm/feature-learning solution) drives the late transition. Mostly a small-data /
  algorithmic-task curiosity, but a clean reminder that *train loss reaching zero says nothing about when (or
  whether) generalization arrives* — keep training and watch the val curve.
- **No Free Lunch.** Averaged over *all* possible problems, no learner beats another. Generalization always
  relies on assumptions (inductive bias) that match the real problem structure. This is why choosing the
  right prior/architecture matters more than chasing a universal best method.
- **Bias–complexity / approximation–estimation.** Test risk = approximation error (best the class can do) +
  estimation error (finite-sample gap) + optimization error. Each phase of work targets a different term.

## 6. Calculus & autodiff (mechanics under the hood)

- **Backprop = reverse-mode autodiff** = repeated chain rule, caching activations on the forward pass to
  compute all parameter gradients in one backward pass at ~2–3× forward cost. Reverse mode is efficient when
  outputs ≪ inputs (one scalar loss, millions of params) — exactly the ML case. Forward mode is for the
  opposite (Jacobian-vector products, few inputs).
- **Memory ↔ compute trade.** Activations dominate training memory; **gradient checkpointing** recomputes
  them to trade compute for memory (see [engineering-scale.md](engineering-scale.md)).
- **Numerical care.** Use log-sum-exp for stable softmax/log-likelihood; never exponentiate raw logits;
  prefer `log1p`/`expm1`; watch fp16 overflow (use bf16 or loss scaling). Gradient checking (finite
  differences vs. autodiff) catches custom-op bugs.
- **Stop-gradients and straight-through estimators** let you control what the gradient sees (target networks,
  VQ-VAE codebooks, Gumbel-softmax for discrete latents). Knowing where gradients *don't* flow is as important
  as where they do.

## 7. What to reach for

| Symptom | Likely cause | Lever |
|---|---|---|
| High train AND test error | Underfitting (bias / optimization) | Bigger model, train longer, better LR, fix data pipeline |
| Low train, high test error | Overfitting (variance) | More data, augmentation, regularization, smaller/earlier-stopped model |
| Loss is NaN/Inf | Numerical / LR too high / bad data | Lower LR, grad clip, bf16, check for bad inputs, log-space math |
| Loss plateaus immediately | LR too low, dead units, bad init | LR-range test, check activations/grad norms, better init/norm |
| Great val, bad deployment | Distribution shift / leakage | Audit splits, re-collect representative data, robustness eval |

**Canonical references:** Murphy *Probabilistic Machine Learning* vol. 1 (intro) & vol. 2 (advanced) and
Bishop *PRML* (breadth); Prince *Understanding Deep Learning* (free, current DL foundations); Shalev-Shwartz
& Ben-David *Understanding Machine Learning* and Mohri, Rostamizadeh & Talwalkar *Foundations of Machine
Learning* (learning theory, PAC/Rademacher); Wainwright *High-Dimensional Statistics* (concentration,
high-dim regression, the right lens for benign overfitting); Boyd & Vandenberghe *Convex Optimization* and
Nesterov *Lectures on Convex Optimization* (optimization); Cover & Thomas *Elements of Information Theory*.
Key papers: Bartlett et al. 2020 (benign overfitting); Jacot et al. 2018 (NTK); Yang & Hu / Tensor Programs
($\mu$P); Cohen et al. 2021 (edge of stability); Lotfi et al. 2022–24 (non-vacuous PAC-Bayes); Jordan et al.
2024 (Muon).
