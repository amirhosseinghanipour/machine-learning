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
- **Mean-field VI** factorizes $q$; **stochastic VI (SVI)** scales it to big data; **black-box / ADVI** and the
  **reparameterization trick** make it automatic and gradient-based — this is exactly the VAE machinery (see
  [generative-models.md](generative-models.md)).
- **Trade-off:** fast and scalable, but reverse-KL is **mode-seeking** → VI typically **underestimates
  posterior variance** (overconfident). Know this when you use VI uncertainty for decisions.

## 5. Markov Chain Monte Carlo (MCMC) — asymptotically exact sampling

Draw samples from the posterior by constructing a Markov chain whose stationary distribution is the posterior.
- **Methods:** Metropolis–Hastings (general, slow), **Gibbs** (conjugate conditionals), **Hamiltonian Monte
  Carlo / NUTS** (gradient-based, the gold standard for continuous parameters — what Stan/NumPyro use), SMC/
  particle filters (sequential/state-space).
- **Trade-off:** asymptotically exact and gives honest uncertainty, but **slow** and doesn't scale to millions
  of parameters or huge data. **Diagnose convergence** — never trust an unchecked chain: $\hat R$ (potential
  scale reduction, want <1.01), effective sample size, trace plots, divergences (HMC). Run multiple chains.
- **VI vs. MCMC:** VI for speed/scale (accept biased, often overconfident uncertainty); MCMC for accuracy on
  smaller models where you need trustworthy posteriors.

## 6. Gaussian Processes (GPs)

A distribution over **functions**: any finite set of points is jointly Gaussian, specified by a mean and a
**kernel** (covariance) function encoding smoothness/periodicity/etc.
- **Strengths:** exact posterior in closed form (regression), **principled, calibrated uncertainty** that grows
  away from data, works in **tiny-data** regimes, and the kernel is a clean place to inject priors. The default
  surrogate model for **Bayesian optimization** (sample-efficient black-box/hyperparameter optimization — see
  [experimentation-reproducibility.md](experimentation-reproducibility.md)).
- **Weakness:** naive cost is $O(n^3)$ — doesn't scale past ~10⁴ points without **sparse/inducing-point** (SVGP)
  or structured (KISS-GP) approximations. Deep kernel learning and neural-net feature maps combine GPs with deep
  learning. High dimensions need care (kernel choice, ARD).

## 7. Bayesian deep learning & scalable uncertainty

Full Bayesian inference over millions of NN weights is intractable; practical approximations:
- **Deep ensembles** — train $K$ networks from different inits; their disagreement approximates epistemic
  uncertainty. **The simplest and one of the most reliable** uncertainty methods; strong baseline, embarrassingly
  parallel. Often beats fancier Bayesian methods.
- **MC Dropout** — keep dropout on at test time, sample multiple forward passes. Cheap, approximate; better than
  nothing but often underestimates uncertainty.
- **Laplace approximation** — Gaussian around the MAP using curvature (last-layer Laplace is cheap and
  effective); **SWAG** (Gaussian from SGD iterate statistics); **variational/Bayes-by-backprop** weight
  posteriors.
- **Evidential / direct uncertainty** — predict distribution parameters directly (e.g., a network outputs mean
  and variance) for cheap aleatoric uncertainty.
- **Reality check:** these are approximations and can be miscalibrated, especially under distribution shift.
  **Always validate uncertainty empirically** — calibration curves, ECE, negative log-likelihood, and selective-
  prediction / OOD-detection metrics (see [evaluation-statistics.md](evaluation-statistics.md) and
  [interpretability-safety.md](interpretability-safety.md)). A confident-but-wrong model is worse than an honest
  one.

## 8. Probabilistic programming languages (PPLs)

Write the generative model; the PPL does inference. Use these instead of hand-deriving inference.
- **Stan** — mature HMC/NUTS, best for classical Bayesian stats/hierarchical models.
- **PyMC** — Pythonic, NUTS + VI, great for science/analytics.
- **NumPyro** (JAX) / **Pyro** (PyTorch) — scalable, composable VI + MCMC, integrate with deep learning;
  NumPyro is very fast via JIT.
- **TensorFlow Probability** — distributions/bijectors/inference as building blocks.
- Use PPLs for: hierarchical/multilevel models, small-to-medium structured problems, when you need real
  posteriors and calibrated uncertainty. Reach for **hierarchical (partial-pooling) models** whenever data is
  grouped (subjects, sites, sessions) — they borrow strength across groups and are a Bayesian superpower.

## 9. Where probabilistic ML earns its keep

| Need | Reach for |
|---|---|
| Calibrated uncertainty from a deep net, cheaply | Deep ensembles (or last-layer Laplace) |
| Tiny data + smooth function + honest uncertainty | Gaussian process |
| Sample-efficient black-box optimization | Bayesian optimization (GP/TPE surrogate) |
| Grouped/hierarchical data, interpretable | Hierarchical Bayesian model in a PPL |
| Trustworthy posterior, smaller model | MCMC (NUTS via Stan/NumPyro) |
| Scalable approximate posterior | Variational inference (mind overconfidence) |
| Latent structure / mixture | EM, VI, or a PGM |
| Decide whether to collect more data | Epistemic uncertainty (active learning) |

**Canonical references:** Bishop *PRML* and Murphy *Probabilistic Machine Learning* (2 vols); Rasmussen &
Williams *Gaussian Processes for ML*; Koller & Friedman *Probabilistic Graphical Models*; Blei et al. 2017 (VI
review); Gelman et al. *Bayesian Data Analysis*; Lakshminarayanan et al. 2017 (deep ensembles); Betancourt's
HMC tutorial.
