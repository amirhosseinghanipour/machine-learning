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
  Ill-conditioning (a few huge eigenvalues) is *why* adaptive optimizers and normalization help.
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
  tests. Hoeffding/Bernstein bounds make "how big must my test set be?" answerable.
- **Distributions to know cold.** Gaussian (MSE, central limit, reparameterization), Bernoulli/Categorical
  (classification, cross-entropy), Beta/Dirichlet (conjugate priors, smoothing), Poisson (counts),
  Exponential family (the unifying form; natural gradients live here).
- **Estimators.** MLE is consistent and asymptotically efficient but can overfit; MAP adds a prior; Bayesian
  posterior quantifies uncertainty. Bias of an estimator vs. variance is the same trade-off as above.
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
  branching factor. Lower is better but only comparable across identical tokenization.

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
  convergence; natural gradient uses the Fisher information metric (the right geometry for probability models).
  Rarely worth it for everyday training but central to some large-scale and RL methods.
- **Convergence intuition.** Strongly convex + smooth → linear rate; convex → $O(1/t)$ (or $O(1/t^2)$ with
  acceleration); non-convex → you get convergence to a stationary point in expectation, nothing global. Step
  size must respect smoothness ($\eta \lesssim 1/L$); too large diverges, too small crawls.
- **Constrained / proximal.** $\ell_1$ via proximal/soft-thresholding; projected gradient for constraints;
  mirror descent for simplex/entropy geometry. Relevant for structured/regularized objectives.

**Saddle points, not minima,** are the obstacle in high dimensions (minima are exponentially rarer). SGD
noise and momentum handle them. **Sharp vs. flat minima:** flatter minima tend to generalize better; small
batch / higher LR / SAM (Sharpness-Aware Minimization) bias toward flatness.

**Practical optimization debugging order:** (1) Is the loss even decreasing on one batch? (2) Is the LR in a
sane range (loss explodes = too high; barely moves = too low — do an LR-range test)? (3) Gradient norms
finite and non-zero (no NaN/Inf, no dead ReLUs)? (4) Is the data/label pipeline correct? Most "optimization
problems" are data or LR problems.

## 5. Statistical learning theory — why generalization is possible

You don't need proofs, but these results shape intuition:

- **Empirical risk minimization (ERM).** We minimize training loss as a proxy for true risk
  $R(\theta)=\mathbb{E}_{(\mathbf{x},y)\sim\mathcal{D}}[L]$. Generalization gap = test − train risk. Theory
  bounds this gap.
- **Capacity controls the gap.** Classical bounds (VC dimension, Rademacher complexity, covering numbers)
  say gap $\lesssim \sqrt{\text{capacity}/n}$. More capacity or less data → looser bound → more overfitting
  risk. This justifies regularization and "more data helps."
- **The modern puzzle.** Deep nets have capacity to memorize random labels yet generalize — classical bounds
  are vacuous for them. Resolution comes from **implicit regularization** (SGD biases toward low-norm/flat
  solutions), **margin** theory, and **the role of the data distribution**. Practical upshot: capacity alone
  doesn't predict generalization; the optimizer and data do a lot of the work.
- **Double descent.** Test error can *decrease again* past the interpolation threshold (where the model
  perfectly fits train). So "bigger overfits more" is false in the overparameterized regime — bigger often
  generalizes better. Know which regime you're in before regularizing.
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

**Canonical references:** Bishop *PRML* and Murphy *Probabilistic ML* (breadth); Goodfellow et al.
*Deep Learning* (DL foundations); Boyd & Vandenberghe *Convex Optimization*; Shalev-Shwartz & Ben-David
*Understanding Machine Learning* (theory); Cover & Thomas *Information Theory*.
