# Classical Machine Learning

Non-deep methods. These are not legacy — gradient-boosted trees remain SOTA on most tabular problems,
and a tuned classical baseline is the honest yardstick for any deep method (see the Prime Directive in
SKILL.md). Master these before reaching for neural nets on structured data.

---

## 1. When classical methods are the right call

- **Tabular / structured data** with heterogeneous features and ≲ a few million rows: **gradient-boosted
  trees are the default**, and on most benchmarks a tuned GBM ties or beats a tuned deep model at a fraction
  of the engineering cost. The one real shift (2025–26): **tabular foundation models** (TabPFN v2, TabICL,
  RealTabPFN) now *match or beat* tuned GBMs on small-to-medium tables — see §4a. Per-row deep nets that train
  from scratch (TabNet, FT-Transformer, SAINT) still rarely justify themselves over a GBM on heterogeneous
  tables.
- **Small data** (hundreds–thousands of examples): linear/kernel/tree models, strong priors, careful CV.
- **Interpretability or auditing required**: linear models, GAMs, shallow trees give faithful explanations
  (unlike post-hoc neural explanations — see [interpretability-safety.md](interpretability-safety.md)).
- **Latency/compute-constrained** deployment: a logistic regression or small GBM is microseconds and trivially
  serveable.
- **As the baseline**, always. If your deep model can't beat a tuned LightGBM / logistic regression, that's
  the finding.

## 2. Linear models — the workhorses and the master template

- **Linear regression.** $\hat y = \mathbf{w}^\top\mathbf{x}+b$; OLS = MLE under Gaussian noise. Closed form
  $\mathbf{w}=(X^\top X)^{-1}X^\top\mathbf{y}$ but use a solver. Assumptions (linearity, homoscedasticity,
  independence) matter for *inference*, less for prediction.
- **Regularized variants** (the prior is the point): **Ridge** ($\ell_2$, shrinks, handles collinearity),
  **Lasso** ($\ell_1$, sparse feature selection), **Elastic Net** (both). Tune $\lambda$ by CV. Standardize
  features first or the penalty is meaningless.
- **Logistic regression.** $p(y{=}1)=\sigma(\mathbf{w}^\top\mathbf{x}+b)$; trained by cross-entropy (MLE for
  Bernoulli). Outputs are *probabilities* and are usually **well-calibrated** out of the box — a major
  advantage. Multiclass: softmax/multinomial. Add $\ell_1/\ell_2$ as needed.
- **GLMs** generalize this to any exponential-family target (Poisson regression for counts, Gamma for positive
  skew, etc.) via a link function. Reach for the GLM that matches your target's distribution rather than
  forcing MSE.
- **Why start here:** fast, calibrated, interpretable coefficients, hard to overfit with regularization, and a
  brutally honest baseline. Feature engineering (below) is what makes them competitive.

## 3. Support vector machines & kernels

- **Linear SVM** maximizes the margin; robust, strong for high-dim sparse data (text bag-of-words). Hinge loss
  + $\ell_2$. The **kernel trick** ($K(\mathbf{x},\mathbf{x}')=\langle\phi(\mathbf{x}),\phi(\mathbf{x}')\rangle$)
  lets it fit nonlinear boundaries via RBF/polynomial kernels without materializing $\phi$.
- **Trade-offs:** kernel SVMs are $O(n^2)$–$O(n^3)$ — they don't scale past ~10⁴–10⁵ samples. Tune $C$
  (regularization) and RBF bandwidth $\gamma$ **jointly** (they interact strongly; a log-spaced 2-D grid or
  Bayesian search). For large $n$, **scale instead of abandoning kernels**: linear SVM/logistic on **random
  Fourier features** or the Nyström approximation recovers most of the RBF benefit at near-linear cost, and is
  the right move before reaching for a kernel SVM on >10⁵ rows.
- **Kernels live on** in Gaussian processes (see [probabilistic-ml.md](probabilistic-ml.md)) and as a
  theoretical lens on neural nets (NTK — infinitely wide nets behave like a fixed kernel; see the caveats in
  [foundations.md](foundations.md)). Note SVMs output uncalibrated margins, not probabilities — calibrate (§8)
  if you need $p(y\mid x)$.

## 4. Tree ensembles — the tabular champions

- **Decision trees** split feature space axis-aligned; capture interactions and nonlinearity natively, need no
  scaling, handle mixed types and missing values. Single trees overfit/high-variance → always ensemble.
- **Random Forests** (bagging + feature subsampling): low-variance, robust, near-zero tuning, great default.
  Out-of-bag error is a free validation estimate. Parallel, hard to break.
- **Gradient-Boosted Trees** (boosting: fit each tree to the negative gradient of the loss, second-order in
  XGBoost/LightGBM which use the Hessian too): the **strongest general tabular learner**.
  - **XGBoost** — battle-tested, level-wise (depth-wise) growth by default, strong L1/L2 regularization,
    `hist`/GPU tree method now default-fast. Native categorical support exists (`enable_categorical=True`) but
    is less mature than CatBoost's.
  - **LightGBM** — histogram-based, **leaf-wise** growth (splits the leaf with max loss reduction): fastest on
    large/wide data and usually most accurate, but leaf-wise overfits small data unless you cap `num_leaves`
    and set `min_data_in_leaf`. Exclusive-feature-bundling + GOSS make it memory-lean. Usually the first reach.
  - **CatBoost** — best-in-class native categorical handling via **ordered target statistics** + **ordered
    boosting**, which compute each row's encoding/gradient using only rows seen *before* it, removing the
    target-leakage that naive mean-encoding causes. Symmetric (oblivious) trees regularize and give very fast
    inference. Strongest out-of-the-box defaults of the three; reach for it when categoricals dominate.
  - **Key knobs (with sane defaults):** `learning_rate` 0.03–0.1 (lower + more trees + early stopping = best,
    slower); tree size — XGBoost `max_depth` 4–8, LightGBM `num_leaves` ≈ 31–255 with `max_depth` as a guard,
    CatBoost `depth` 6–10; `n_estimators` large (1000–10000) with **early stopping** on a validation metric
    doing the real selection; `subsample`/`bagging_fraction` 0.7–1.0 and `colsample_bytree`/`feature_fraction`
    0.6–1.0 (stochasticity → regularization); `min_child_weight`/`min_data_in_leaf` raised to fight overfit on
    small/noisy data; `reg_lambda` (L2) and `reg_alpha` (L1) for extra shrinkage. The single most reliable
    recipe: fix a low LR, set a huge tree budget, and let early stopping pick the count.
  - **Strong fixed defaults exist.** "Better by Default" (Holzmüller et al., 2024) and similar work show
    *pre-tuned* GBM/MLP configs that beat library defaults and rival per-dataset tuning at near-zero search
    cost — a good warm start before spending an HPO budget.
  - **Loss/objective beyond squared error:** set the objective to match the target — Tweedie/Poisson for counts
    and insurance, quantile (pinball) loss for prediction intervals, ranking objectives (LambdaMART) for
    learning-to-rank, custom first/second-derivative objectives where needed. Monotonic constraints
    (`monotone_constraints`) and interaction constraints are well-supported and valuable for trust/regulation.
- **Gotchas:** (1) trees **extrapolate poorly** — predictions are piecewise-constant and flat outside the
  training range, so they cannot capture trends; for time series, model differences/detrend first (see
  [domains.md](domains.md)). (2) Native categorical handling and LightGBM's missing-value routing make trees
  forgiving, but **high-cardinality target encoding leaks** if fit outside CV folds — use CatBoost's ordered
  encoding or fit encoders inside folds (see [data.md](data.md)). (3) Default **feature importances (gain/split)
  are biased** toward high-cardinality and continuous features and are computed on train — prefer permutation
  importance on held-out data or SHAP (TreeSHAP is exact and fast) for attribution. (4) GBMs are **not
  calibrated** with non-log-loss objectives and can be miscalibrated even with log-loss after heavy tuning —
  check and recalibrate (§8). (5) Class imbalance: prefer `scale_pos_weight`/class weights + threshold tuning
  over resampling.

## 4a. Why trees still win on tabular — and the foundation-model exception

**The mechanism (Grinsztajn et al., 2022), worth internalizing:** GBMs beat deep nets on typical tabular data
for three structural reasons. (1) **Non-smooth target functions** — tabular targets are often irregular/jagged
in feature space; trees fit axis-aligned piecewise-constant functions natively, while MLPs are biased toward
overly smooth functions and need many parameters to approximate a step. (2) **Robustness to uninformative
features** — real tables carry many weak/noise columns; trees ignore them via split selection, while MLPs are
hurt by them. (3) **Rotation non-invariance is a *feature*, not a bug** — MLPs are (approximately) rotationally
invariant, so they treat an arbitrary linear mixture of columns the same as the raw columns; but tabular
columns are individually meaningful and on different scales, and the *axis-aligned* bias of trees matches that
structure. This is why the gap shrinks as you remove uninformative features or as datasets get very large/
homogeneous, and why per-column attention (FT-Transformer, and the foundation models below) — which restores
an axis-aligned, per-feature bias — closes more of it than plain MLPs.

**Tabular foundation models (the 2025–26 exception).** **TabPFN v2** (Hollmann et al., *Nature* 2025) is a
transformer *pretrained on millions of synthetic tabular tasks* that does **in-context learning**: you pass the
whole training set as context and it predicts the test rows in a single forward pass — no per-dataset gradient
training. On small-to-medium data it matches or beats tuned GBMs in seconds. Limits: roughly ≤10k rows, ≤500
features, ≤10 classes per call. Successors lift these: **TabICL** (column-then-row attention) scales to ~500k
rows and is much faster on wide data; **RealTabPFN / TabPFN v2.5** continue-train on curated real data for a
further bump (often non-commercial license). The **TabArena** leaderboard is the current living benchmark; as
of 2026 a strong TFM is the best single model, with only heavy AutoML ensembles (AutoGluon at hours of compute)
ahead overall. **Practical guidance:** on small/medium tables, *try a tabular foundation model as a baseline
alongside a GBM* — it is now often the fastest path to a strong number. GBMs still own the regime of millions
of rows, hard latency/memory budgets, streaming/online updates, and maximal interpretability. Check licensing
before production use.

## 5. Instance-based, probabilistic, and other staples

- **k-NN.** Nonparametric, no training; predict by nearest neighbors. Curse of dimensionality hurts; needs
  scaled features and a good distance/metric. A useful sanity baseline and the backbone of retrieval (with
  ANN indexes: FAISS/HNSW).
- **Naive Bayes.** Strong, fast text baseline despite the (false) independence assumption; well-suited to
  high-dim sparse counts. Often the first thing to try on document classification.
- **Discriminant analysis (LDA/QDA).** Gaussian-class-conditional generative classifiers; LDA gives a linear
  boundary, QDA quadratic. Cheap, calibrated, good when classes are roughly Gaussian.

## 6. Unsupervised learning

**Clustering** (no ground truth → validate carefully):
- **k-means** — fast, assumes spherical equal-variance clusters; choose $k$ via elbow/silhouette/gap statistic
  (all heuristic). Sensitive to init (use k-means++) and scaling.
- **Gaussian Mixture Models** — soft clustering, elliptical clusters, gives densities; fit by EM. Choose
  components by BIC/AIC.
- **DBSCAN / HDBSCAN** — density-based, finds arbitrary shapes and outliers, no preset $k$; HDBSCAN handles
  varying density. Great default when cluster count is unknown.
- **Hierarchical (agglomerative)** — dendrograms, no preset $k$, $O(n^2)$ memory; good for small data and
  exploration.
- **Spectral clustering** — graph-Laplacian eigenvectors then k-means; captures non-convex structure.
- **Validation:** internal indices (silhouette, Davies–Bouldin) are heuristics; clustering is exploratory —
  resist over-interpreting. If you have *any* labels, use them (ARI/NMI).

**Dimensionality reduction:**
- **PCA** — linear, variance-maximizing, the default first step; orthogonal components, fast, invertible. It is
  the SVD of the centered data (see [foundations.md](foundations.md)); **standardize first** unless features
  share units, or high-variance columns dominate the components for spurious reasons. Use for denoising,
  compression, decorrelation, whitening, and visualization (first 2–3 PCs); use randomized/truncated SVD for
  large matrices. Caveat: variance ≠ discriminability — top PCs need not be the predictive directions (use PLS
  or supervised reduction when the goal is prediction).
- **t-SNE** — nonlinear, beautiful local-structure 2D maps, but **inter-cluster distances and cluster sizes are
  not meaningful**, it can manufacture clusters from noise, and it is sensitive to perplexity. For
  visualization only; never feed t-SNE coordinates to a downstream model or read global geometry from it.
- **UMAP** — faster than t-SNE, preserves somewhat more global structure, and can transform new points; the
  common default for embedding visualization. **Same caveats apply, and recent work (Chari & Pachter, 2023) is
  blunt: t-SNE/UMAP layouts can distort or destroy the true high-dimensional structure** they appear to show —
  treat them strictly as qualitative sketches, validate any apparent cluster in the original space, and never
  base a quantitative claim on a 2-D embedding. PCA/PaCMAP/TriMap or simple linear projections are useful sanity
  checks against UMAP artifacts.
- **Others:** ICA (independent sources, e.g., signal separation), NMF (parts-based, nonnegative data),
  random projections (Johnson–Lindenstrauss, cheap), autoencoders (nonlinear, see
  [representation-learning.md](representation-learning.md)).

**Density estimation & anomaly detection:** KDE (low-dim), GMMs, Isolation Forest and One-Class SVM
(anomalies), and increasingly normalizing flows / diffusion for high-dim densities (see
[generative-models.md](generative-models.md)).

## 7. Feature engineering & preprocessing (where tabular wins are made)

- **Scaling:** standardize (zero mean/unit var) for linear/SVM/k-NN/NN; trees don't need it. **Fit the scaler
  on train only, inside CV folds** — scaling on the full set is leakage.
- **Categoricals:** one-hot (low cardinality), ordinal (ordered), target/mean encoding (high cardinality — do
  it *inside CV folds* with smoothing or use CatBoost to avoid leakage), hashing (very high cardinality),
  embeddings (with NNs).
- **Missing values:** trees/LightGBM/XGBoost handle natively; otherwise impute (median/most-frequent, or model-
  based) and **add a missingness indicator** (missingness is often informative). Fit imputers on train only.
- **Interactions & transforms:** ratios, differences, polynomial features, binning, domain-derived features.
  This is where human knowledge enters and often beats model choice.
- **Imbalance:** prefer class weights / `scale_pos_weight` and threshold tuning over naive resampling;
  SMOTE can help but can also create unrealistic points — validate. Pick metrics that respect imbalance
  (PR-AUC, balanced accuracy, F1) — see [evaluation-statistics.md](evaluation-statistics.md).
- **Leakage audit:** the single most important step. Any feature that encodes the target or uses future/
  out-of-fold information will inflate validation and collapse in deployment. See [data.md](data.md).

## 8. Calibration, probabilities & distribution-free uncertainty

Many decisions need *probabilities* or *guaranteed coverage*, not just rankings.

**Calibration.** Logistic/linear regression and GBMs trained with log-loss are roughly calibrated; SVMs
(margin scores, not probabilities), naive Bayes (overconfident due to the independence assumption), and many
modern deep nets are not. Diagnose with a **reliability diagram** plus a summary metric — but know that
**Expected Calibration Error (ECE) is binning-sensitive and biased**; prefer **adaptive/equal-mass binning**,
report alongside a proper scoring rule (**Brier score**, **log-loss**), and use the maximum-calibration-error
or a debiased estimator when the decision is high-stakes. Fixes, fit on a **held-out calibration split** (never
train): **Platt scaling** (1-parameter sigmoid — low data, assumes a sigmoidal distortion), **isotonic
regression** (nonparametric monotone — more flexible, needs ~1000+ calibration points or it overfits), and
**temperature scaling** (single scalar on logits — the default for multiclass deep nets, preserves accuracy
and argmax). Recalibrate per-deployment-distribution; calibration does not transfer across shift.

**Decompose proper scores.** Brier = calibration − refinement (resolution) + irreducible; a model can be
well-calibrated yet useless (predicts the base rate). Optimize for discrimination first, then calibrate.

**Conformal prediction** (the distribution-free complement to calibration). Wrap *any* fitted model to produce
**prediction sets/intervals with finite-sample marginal coverage** $\ge 1-\alpha$ under only the exchangeability
assumption — no distributional or model-correctness assumptions. Split (inductive) conformal: hold out a
calibration set, compute nonconformity scores $s_i$ (e.g. $1-\hat p_y$ for classification, $|y-\hat y|$ or a
normalized residual for regression), take the $\lceil(n+1)(1-\alpha)\rceil/n$ empirical quantile $\hat q$, and
emit $\{y : s(x,y)\le\hat q\}$. Cost is one held-out split and one quantile — cheap insurance. Caveats: the
guarantee is **marginal**, not conditional (per-group coverage can be off — use Mondrian/group-conditional
variants); exchangeability **breaks under temporal/distribution shift** (use adaptive conformal / weighted
conformal under shift); and interval *width* (efficiency) still depends on a good base model. Use it whenever
you need honest, defensible uncertainty without trusting the model's own probabilities — increasingly the
standard for risk-sensitive tabular and forecasting work. See [evaluation-statistics.md](evaluation-statistics.md).

## 9. AutoML & hyperparameter search (classical)

- **Search:** grid (small spaces), random (better per-dollar than grid — Bergstra & Bengio 2012), Bayesian
  optimization (Optuna, scikit-optimize) for expensive evals, successive halving/Hyperband for early stopping.
  See [experimentation-reproducibility.md](experimentation-reproducibility.md).
- **Pipelines:** wrap preprocessing + model in a single scikit-learn `Pipeline`/`ColumnTransformer` so
  transforms are fit inside CV folds (prevents leakage by construction). This is the single best habit for
  correct classical ML.
- **AutoML tools:** auto-sklearn, AutoGluon (very strong tabular defaults, stacks GBMs+NNs), H2O,
  FLAML. Excellent strong-baseline generators — run one early to know the bar.

## 10. Reach-for table

| Situation | First choice | Why |
|---|---|---|
| Tabular prediction, large (≫10⁵ rows) or latency-bound | LightGBM/XGBoost/CatBoost | SOTA, robust, fast, scales, interpretable |
| Small–medium tabular (≲10k rows), best number fast | TabPFN v2 / TabICL **and** a tuned GBM | TFMs now match/beat GBMs in seconds (§4a) |
| Need calibrated probabilities + interpretability | Logistic/linear (regularized) | Calibrated, faithful coefficients |
| Distribution-free guaranteed coverage | Conformal prediction over any model | Finite-sample coverage, model-agnostic (§8) |
| Tiny tabular data | TabPFN v2, or regularized linear | Pretrained prior / strong prior beats capacity |
| High-dim sparse (text counts) | Linear SVM / logistic / naive Bayes | Scales, strong baseline |
| Unknown cluster structure | HDBSCAN | No preset $k$, finds shapes + noise |
| Visualize high-dim embeddings | UMAP (then PCA to sanity-check) | Fast, some global structure |
| Anomaly detection, tabular | Isolation Forest | Scales, few assumptions |
| Strong baseline fast | AutoGluon | Stacked ensemble, minimal tuning |

**Canonical references:** Hastie, Tibshirani & Friedman *Elements of Statistical Learning* and Murphy
*Probabilistic Machine Learning* (theory + breadth); scikit-learn user guide; XGBoost/LightGBM/CatBoost docs
(Chen & Guestrin 2016; Ke et al. 2017; Prokhorenkova et al. 2018); Bergstra & Bengio 2012 (random search);
Grinsztajn et al. 2022 (why trees still beat deep learning on tabular data); Hollmann et al. 2025 (TabPFN v2,
*Nature*) and the TabArena benchmark; Angelopoulos & Bates 2023 (*A Gentle Introduction to Conformal
Prediction*); Chari & Pachter 2023 (the specious art of single-cell/embedding visualization).
