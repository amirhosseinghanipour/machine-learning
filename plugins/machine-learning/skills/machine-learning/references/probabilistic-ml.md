# Probabilistic & Bayesian Machine Learning

Modeling uncertainty explicitly. Reach for this when you need **calibrated uncertainty**, work in a
**low-data** regime, want to **incorporate priors/domain knowledge**, need **interpretable structure**, or must
**reason about what the model doesn't know** (active learning, Bayesian optimization, safety-critical
decisions). Foundations (likelihood, KL, exponential family) are in [foundations.md](foundations.md).

---

## 1. The Bayesian frame

$\underbrace{p(\theta\mid\mathcal{D})}_\text{posterior} \propto \underbrace{p(\mathcal{D}\mid\theta)}_\text{likelihood}\,\underbrace{p(\theta)}_\text{prior}$.
You don't pick one $\theta$; you maintain a *distribution* over $\theta$ and **marginalize** to predict:
$p(y\mid\mathbf{x},\mathcal{D})=\int p(y\mid\mathbf{x},\theta)\,p(\theta\mid\mathcal{D})\,d\theta$.
- **MLE → MAP → full Bayes** is a spectrum: MLE ignores the prior, MAP adds it as regularization (a point
  estimate), full Bayes integrates over uncertainty. The integral is the hard part — everything below is a way
  to approximate it.
- **Why bother:** principled uncertainty (aleatoric = inherent noise vs. epistemic = reducible-with-data),
  natural regularization, automatic Occam's razor (the **marginal likelihood / evidence** penalizes complexity),
  and coherent updating as data arrives.
- **Two kinds of uncertainty, never conflate them:** *aleatoric* (data noise — irreducible, model with the
  likelihood/output variance) vs. *epistemic* (model/parameter uncertainty — shrinks with data, what Bayesian
  posteriors capture). Decisions about *collecting more data* depend on epistemic; decisions about *risk* need
  both.

## 2. Conjugacy and the exponential family (the closed-form cases)

When prior and likelihood are conjugate (Beta–Bernoulli, Dirichlet–Categorical, Gamma–Poisson, Normal–Normal),
the posterior has a closed form — fast, exact, and the backbone of many models (LDA topic models, Bayesian A/B
testing, Thompson sampling). The exponential family is where conjugacy lives. Most real models aren't conjugate,
forcing approximate inference (§4–§5).

## 3. Probabilistic graphical models (PGMs)

Encode conditional independence structure as a graph → factorize the joint, enabling tractable inference and
interpretable structure.
- **Directed (Bayesian networks):** causal/generative structure; factorize $p(\mathbf{x})=\prod_i p(x_i\mid
  \text{parents}(x_i))$. Naive Bayes, HMMs, LDA, and most generative latent-variable models are special cases.
- **Undirected (Markov Random Fields / CRFs):** symmetric dependencies (images, sequences-as-structured-output);
  **CRFs** were the standard for structured prediction (NER, segmentation) pre-deep-learning and still appear as
  output layers.
- **Inference:** exact (variable elimination, belief propagation / message passing on trees) when structure
  allows; otherwise approximate (§4–§5). **Latent-variable learning** via **EM** (Expectation-Maximization:
  alternate inferring latents and maximizing parameters — the algorithm behind GMMs, HMMs, and the conceptual
  parent of variational inference).
- **Status:** deep nets absorbed much structured prediction, but PGM thinking (independence, plate notation,
  message passing) is essential for VAEs, diffusion, and any model with latent structure.

## 4. Variational Inference (VI) — turn inference into optimization

Approximate the intractable posterior $p(\theta\mid\mathcal{D})$ with a simpler $q_\phi(\theta)$ by **maximizing
the ELBO** (equivalently minimizing reverse KL $D_\text{KL}(q\|p)$):
$\log p(\mathcal{D}) \ge \mathbb{E}_{q}[\log p(\mathcal{D},\theta)] - \mathbb{E}_q[\log q_\phi(\theta)] = \text{ELBO}$.
- **Mean-field VI** factorizes $q$; **stochastic VI (SVI)** subsamples data for a noisy ELBO gradient and scales
  to big data; **black-box VI / ADVI** and the **reparameterization trick** make it automatic and gradient-based
  — this is exactly the VAE machinery (see [generative-models.md](generative-models.md)). For discrete latents
  the reparam trick fails; use a score-function (REINFORCE) estimator with control variates, or a
  Gumbel-Softmax/concrete relaxation.
- **The gradient estimator is the whole ballgame.** Reparameterization (pathwise) gradients have far lower
  variance than score-function gradients — prefer them whenever the latent is continuous and reparameterizable.
  Variance-reduce score-function estimators with baselines/control variates and **RB / Rao-Blackwellization**.
  Use **multi-sample bounds** (IWAE: $K$ importance samples tighten the bound) when a single sample is too loose,
  but note IWAE *weakens* the inference-network gradient signal — use DReG (doubly-reparameterized) gradients.
- **Richer $q$ families beat mean-field.** Full-rank Gaussian, **structured/low-rank+diagonal** covariance, and
  **normalizing-flow VI** (parameterize $q$ as an invertible transform of a base density — planar/RealNVP/IAF/
  neural-spline flows) capture correlations and non-Gaussianity mean-field misses. Flow VI is the practical way
  to get expressive posteriors at scale; watch training stability in high dimensions (gradient clipping, careful
  init, affine-invariant tempering / annealed objectives help with multimodality — recent FlowVAT-style methods).
- **Trade-off:** fast and scalable, but reverse-KL is **mode-seeking** → VI typically **underestimates
  posterior variance** and collapses to a single mode (overconfident). Forward-KL (used by EP / amortized SBI)
  is mode-covering instead. Know which one you ran when you use VI uncertainty for decisions.
- **Diagnostics, because VI gives no convergence guarantee:** track the ELBO to a plateau, then check the
  **importance-weighted PSIS-$\hat k$ diagnostic** (Yao et al.; $\hat k<0.7$ ≈ trustworthy, the standard VI
  sanity check) and compare moments against a short MCMC run on a subset. Never report a VI posterior without a
  diagnostic — the ELBO being maximized says nothing about how close $q$ is to $p$.

## 5. Markov Chain Monte Carlo (MCMC) — asymptotically exact sampling

Draw samples from the posterior by constructing a Markov chain whose stationary distribution is the posterior.
- **Methods:** Metropolis–Hastings (general, slow), **Gibbs** (conjugate conditionals), **Hamiltonian Monte
  Carlo / NUTS** (gradient-based, the gold standard for continuous parameters — what Stan/PyMC/NumPyro use), SMC/
  particle filters (sequential/state-space). For massive-data settings, **stochastic-gradient MCMC** (SGLD,
  SG-HMC) trades exactness for minibatch scalability but is biased — diagnose carefully.
- **HMC/NUTS defaults that matter:** NUTS auto-tunes path length, so the levers are the **target acceptance
  rate** (`target_accept`/`adapt_delta`; raise to 0.9–0.99 to kill divergences at the cost of speed), **mass-
  matrix adaptation** (use dense for correlated posteriors), and warmup length. **Reparameterize hierarchical
  models** — the funnel geometry of centered parameterizations breaks HMC; switch to the **non-centered**
  parameterization ($\theta = \mu + \sigma z,\ z\sim\mathcal N(0,1)$) as a near-automatic fix. Divergences are
  not cosmetic: they flag regions HMC cannot explore, so the posterior is biased — fix them, don't ignore them.
- **Trade-off:** asymptotically exact and gives honest uncertainty, but **slow** and doesn't scale to millions
  of parameters or huge data. **Diagnose convergence** — never trust an unchecked chain: **rank-normalized
  split-$\hat R$** (Vehtari et al. 2021; want **<1.01**, stricter than the old 1.1), **bulk- and tail-ESS**
  (want ≳ 400; tail-ESS guards quantile/interval estimates), trace/rank plots, energy (BFMI) and divergence
  counts for HMC. Run **≥4 chains** from dispersed inits.
- **VI vs. MCMC:** VI for speed/scale (accept biased, often overconfident uncertainty); MCMC for accuracy on
  smaller models where you need trustworthy posteriors. A common production pattern: prototype/iterate with VI,
  then validate the final model with NUTS.
- **Simulation-based inference (SBI / likelihood-free).** When you have a simulator but no tractable likelihood
  (common in physics/biology/cosmology), amortized neural methods learn the posterior, likelihood, or ratio from
  simulations: **NPE/SNPE** (neural posterior estimation, often normalizing-flow-based), **NLE**, **NRE**
  (ratio estimation). The `sbi` package is the standard toolkit; once trained, inference on new observations is a
  fast forward pass. Validate calibration with **SBC (simulation-based calibration)** — non-uniform rank
  histograms reveal a miscalibrated posterior.

## 6. Gaussian Processes (GPs)

A distribution over **functions**: any finite set of points is jointly Gaussian, specified by a mean and a
**kernel** (covariance) function encoding smoothness/periodicity/etc.
- **Strengths:** exact posterior in closed form (regression), **principled, calibrated uncertainty** that grows
  away from data, works in **tiny-data** regimes, and the kernel is a clean place to inject priors. The default
  surrogate model for **Bayesian optimization** (sample-efficient black-box/hyperparameter optimization — see
  [experimentation-reproducibility.md](experimentation-reproducibility.md)).
- **Weakness:** naive cost is $O(n^3)$ time / $O(n^2)$ memory — doesn't scale past ~10⁴ points without
  approximation. The scalable-GP toolbox, in rough order of reach:
  - **Sparse variational GPs (SVGP)** — $m$ inducing points summarize the data; $O(nm^2)$, minibatchable, the
    default scalable workhorse (Titsias; Hensman SVGP). Typically capped at a few thousand inducing points.
  - **Variational nearest-neighbor GP (VNNGP)** — sparse-precision approximation letting you place an inducing
    point at *every* observation ($M=N$) by depending on only $K$ neighbors; $O(K^3)$ ELBO estimates from
    minibatches of inducing points. Scales orders of magnitude past SVGP; in GPyTorch as `NNVariationalStrategy`
    (default $k\!=\!256$).
  - **Structured kernel interpolation (KISS-GP / SKI)** and **conjugate-gradient / Lanczos** exact GPs
    (GPyTorch's `KeOps`-backed solvers) push *exact* inference to ~10⁵–10⁶ points on GPUs.
  - **State-space GPs** for 1-D/temporal kernels ($O(n)$ via Kalman filtering).
  - **Deep kernel learning** and neural-net feature maps combine GPs with deep learning; **deep GPs** stack GP
    layers for hierarchical non-stationarity.
  - **Tooling:** **GPyTorch** (PyTorch, GPU-first, the de-facto default), **GPflow** (TF), **GPJax** (JAX),
    **BoTorch** for GP-based Bayesian optimization.
- **High dimensions need care:** use **ARD** (per-dimension lengthscales) for automatic relevance/feature
  selection, and standardize inputs/outputs. The choice of kernel (Matérn-5/2 is a robust default over the
  too-smooth RBF; add periodic/linear components as priors dictate) matters more than almost anything else.

## 7. Bayesian deep learning & scalable uncertainty

Full Bayesian inference over millions of NN weights is intractable; practical approximations:
- **Deep ensembles** — train $K$ networks from different inits ($K\!=\!5$ is the usual sweet spot); their
  disagreement approximates epistemic uncertainty. **The simplest and one of the most reliable** uncertainty
  methods; strong baseline, embarrassingly parallel, and still hard to beat on accuracy + calibration + OOD as of
  2026. Cheaper cousins amortize the cost: **snapshot ensembles** (cyclical-LR checkpoints), **MultiSWAG**, and
  **BatchEnsemble / rank-1 factors** (shared weights + cheap per-member perturbations).
- **MC Dropout** — keep dropout on at test time, sample multiple forward passes. Cheap, approximate; better than
  nothing but **systematically underestimates** epistemic uncertainty and is sensitive to the (often
  un-tuned) dropout rate. Treat as a weak baseline, not a trustworthy posterior.
- **Laplace approximation (the "effortless" post-hoc BNN)** — fit a Gaussian around a trained MAP solution using
  the loss curvature (Hessian ≈ Fisher / GGN). **Last-layer Laplace with a KFAC Hessian is the recommended
  default** — nearly free, applied post-hoc to an already-trained net, and competitive with ensembles on
  calibration. Crucial detail: use the **linearized / GLM predictive** (push the Gaussian through a first-order
  Taylor expansion of the network), *not* naive MC sampling of weights, which degrades accuracy. The marginal
  likelihood it yields also enables online hyperparameter/prior-precision tuning. Tooling: **`laplace-torch`**
  (PyTorch; structures `all`/`subnetwork`/`last_layer` × `full`/`kron`/`lowrank`/`diag`), **`laplax`** (JAX).
- **SWAG** — Gaussian from the first/second moments of SGD iterates along a high-LR trajectory; **variational /
  Bayes-by-backprop** weight posteriors are heavier and rarely worth it over ensembles/Laplace in practice.
- **Evidential / direct uncertainty** — predict distribution parameters directly (a network outputs mean *and*
  variance, or the parameters of a higher-order distribution) for cheap single-pass aleatoric (and claimed
  epistemic) uncertainty. Caveat: evidential-regression epistemic estimates are known to be poorly grounded and
  can be unreliable — validate, don't trust the label.
- **Reality check:** these are approximations and can be miscalibrated, **especially under distribution shift**,
  where most BDL methods degrade (see the *Shifts* / OOD-stress literature). **Always validate uncertainty
  empirically** — reliability/calibration curves, **ECE** (and its failure modes — prefer adaptive-binning or
  proper scores), **negative log-likelihood / Brier** (proper scoring rules), and selective-prediction /
  OOD-detection metrics (AUROC on in- vs out-of-distribution) — see [evaluation-statistics.md](evaluation-statistics.md)
  and [interpretability-safety.md](interpretability-safety.md). A confident-but-wrong model is worse than an
  honest one. **Post-hoc recalibration** (temperature scaling for classifiers; isotonic/Platt) is cheap and
  should be a default last step.

## 7b. Conformal prediction — distribution-free guarantees you can actually ship

When you need a *guarantee* rather than a hope, **conformal prediction (CP)** wraps any model (no retraining, no
distributional assumptions) and produces prediction sets/intervals with **finite-sample marginal coverage**:
$\Pr(y \in \mathcal C(x)) \ge 1-\alpha$, exactly, given only **exchangeability** of calibration and test data.
- **Split (inductive) CP** is the practical default: hold out a calibration set, compute a **nonconformity
  score** $s(x,y)$ on it, set the threshold $\hat q$ to the $\lceil(1-\alpha)(n+1)\rceil/n$ empirical quantile,
  and predict $\mathcal C(x)=\{y: s(x,y)\le\hat q\}$. Cost ≈ one extra inference pass.
- **Score choices set the behavior.** Classification: **APS** (adaptive prediction sets — cumulative softmax,
  better conditional coverage) and **RAPS** (regularized APS — penalizes set size for tighter sets) beat naive
  softmax-threshold. Regression: **CQR (conformalized quantile regression)** wraps a quantile regressor for
  intervals that *adapt to heteroscedastic* noise while keeping the coverage guarantee — the regression default.
- **The catch — marginal ≠ conditional.** Vanilla CP guarantees coverage *on average over the population*, not
  per-group or per-input; it can under-cover hard slices and over-cover easy ones. Use **Mondrian/group-balanced
  CP** for per-group guarantees and report set size (efficiency), not just coverage.
- **Exchangeability breaks under shift and time.** For time series / online data use **adaptive CP (ACI)** or
  **conformal PID** control that adjusts $\alpha_t$ to maintain long-run coverage; for covariate shift use
  weighted CP. **Conformal risk control** generalizes the guarantee to bounded losses (e.g., FNR/FDR in
  segmentation). Tooling: **MAPIE**, **TorchCP**, `crepes`.
- CP composes with everything above: it calibrates *intervals/sets*; ensembles/Laplace/GPs still give you the
  *shape* of uncertainty and better-adapted scores. They are complements, not substitutes.

## 8. Probabilistic programming languages (PPLs)

Write the generative model; the PPL does inference. Use these instead of hand-deriving inference.
- **Stan** — mature, battle-tested HMC/NUTS, best for classical Bayesian stats/hierarchical models; `cmdstanpy`/
  `brms`(R formula interface) front-ends. The reference implementation of adaptive HMC.
- **PyMC** (5.x) — Pythonic, PyTensor backend, NUTS + VI; for speed it can dispatch NUTS to JAX samplers
  (**NumPyro/BlackJAX**) or the Rust-based **`nutpie`**, which often sample several× faster than the default.
- **NumPyro** (JAX) / **Pyro** (PyTorch) — scalable, composable VI + MCMC, integrate with deep learning;
  NumPyro's JIT-compiled NUTS is among the fastest CPU/GPU samplers available and a strong default for
  large hierarchical models.
- **BlackJAX** — low-level JAX sampling primitives (NUTS, SMC, SGMCMC) when you want to build a custom inference
  loop; **`dynesty`/`jaxns`** for nested sampling (good for multimodal, low-dimensional model-evidence problems;
  struggles in high dimensions). **`sbi`** for simulation-based inference (§5).
- **TensorFlow Probability** — distributions/bijectors/inference as building blocks.
- Use PPLs for: hierarchical/multilevel models, small-to-medium structured problems, when you need real
  posteriors and calibrated uncertainty. Reach for **hierarchical (partial-pooling) models** whenever data is
  grouped (subjects, sites, sessions) — they borrow strength across groups and are a Bayesian superpower. Set
  **weakly-informative priors** by default (e.g., half-Normal/half-Student-t on scales — Gelman's guidance), and
  always run **prior predictive checks** (simulate data from the prior — does it look remotely plausible?) before
  fitting and **posterior predictive checks** after. Compare models with **LOO-CV via PSIS** (`arviz.loo`) over
  raw WAIC/DIC.

## 9. Where probabilistic ML earns its keep

| Need | Reach for |
|---|---|
| Calibrated uncertainty from a deep net, cheaply | Deep ensembles (K≈5); or post-hoc last-layer Laplace (linearized predictive) |
| A *guaranteed* coverage interval/set, any model, no retraining | Split conformal (CQR for regression, APS/RAPS for classification) |
| Tiny data + smooth function + honest uncertainty | Gaussian process (Matérn-5/2 + ARD) |
| GP past ~10⁴ points | SVGP → VNNGP (M=N) → SKI/CG-exact (GPyTorch) |
| Sample-efficient black-box optimization | Bayesian optimization (GP surrogate, BoTorch) |
| Grouped/hierarchical data, interpretable | Hierarchical (partial-pooling) model in a PPL; non-centered param |
| Trustworthy posterior, smaller model | MCMC (NUTS via Stan/NumPyro/nutpie); check split-$\hat R$, ESS, divergences |
| Scalable approximate posterior | SVI; flow-VI for expressive $q$ (mind overconfidence, check PSIS-$\hat k$) |
| Simulator but no likelihood | Simulation-based inference (`sbi`: NPE/NRE); validate with SBC |
| Latent structure / mixture | EM, VI, or a PGM |
| Decide whether to collect more data | Epistemic uncertainty (active learning) |

**Canonical references:** Bishop *PRML* and *Deep Learning: Foundations and Concepts* (2024); Murphy
*Probabilistic Machine Learning* (2 vols, 2022/2023); Rasmussen & Williams *Gaussian Processes for ML*; Koller &
Friedman *Probabilistic Graphical Models*; Blei, Kucukelbir & McAuliffe 2017 (VI review); Gelman et al.
*Bayesian Data Analysis* (3rd ed.) and the **Bayesian Workflow** paper (Gelman et al. 2020); Vehtari et al. 2021
(rank-normalized $\hat R$/ESS) and Yao et al. 2018 (Yes-but-did-it-work / PSIS for VI); Lakshminarayanan et al.
2017 (deep ensembles); Daxberger/Immer et al. 2021 (*Laplace Redux*); Wu et al. 2022 (VNNGP); Angelopoulos &
Bates 2023 (*Gentle Intro to Conformal Prediction*); Cranmer et al. 2020 (SBI); Betancourt's HMC tutorial.
