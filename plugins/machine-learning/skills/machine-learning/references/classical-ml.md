# Classical Machine Learning

Non-deep methods. These are not legacy — gradient-boosted trees remain SOTA on most tabular problems,
and a tuned classical baseline is the honest yardstick for any deep method (see the Prime Directive in
SKILL.md). Master these before reaching for neural nets on structured data.

---

## 1. When classical methods are the right call

- **Tabular / structured data** with heterogeneous features and ≲ a few million rows: **gradient-boosted
  trees win**, full stop, on most benchmarks. Deep tabular models (TabNet, FT-Transformer, TabPFN for small
  data) have closed some gap but rarely beat well-tuned GBMs on heterogeneous tables.
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
  (regularization) and kernel bandwidth $\gamma$ jointly; they interact strongly.
- **Kernels live on** in Gaussian processes (see [probabilistic-ml.md](probabilistic-ml.md)) and as a
  theoretical lens on neural nets (NTK — infinitely wide nets behave like kernel methods).

## 4. Tree ensembles — the tabular champions

- **Decision trees** split feature space axis-aligned; capture interactions and nonlinearity natively, need no
  scaling, handle mixed types and missing values. Single trees overfit/high-variance → always ensemble.
- **Random Forests** (bagging + feature subsampling): low-variance, robust, near-zero tuning, great default.
  Out-of-bag error is a free validation estimate. Parallel, hard to break.
- **Gradient-Boosted Trees** (boosting: fit each tree to the residual/gradient of the loss): the **strongest
  general tabular learner**.
  - **XGBoost** — battle-tested, regularized, great defaults.
  - **LightGBM** — histogram-based, leaf-wise growth, fastest on large data; usually first reach.
  - **CatBoost** — best-in-class native categorical handling (ordered boosting reduces target leakage), strong
    defaults.
  - **Key knobs:** learning rate (lower + more trees = better, slower), max depth / num leaves, subsample &
    colsample (stochasticity → regularization), min child weight, L1/L2. Tune with early stopping on a
    validation set. Lower LR + early stopping is the reliable recipe.
- **Gotchas:** trees extrapolate poorly (predictions flat outside the training range — bad for trends/time
  series without differencing); high-cardinality categoricals need care (target encoding leaks if done outside
  CV folds — use CatBoost or fit encoders inside folds, see [data.md](data.md)).

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
- **PCA** — linear, variance-maximizing, the default first step; orthogonal components, fast, invertible.
  Use for denoising, compression, decorrelation, and visualization (first 2–3 PCs).
- **t-SNE** — nonlinear, beautiful local-structure 2D maps, but **distances and cluster sizes between groups
  are not meaningful** and it's sensitive to perplexity. For visualization only; never feed t-SNE coords to a
  downstream model or read global geometry from it.
- **UMAP** — faster than t-SNE, preserves more global structure, can transform new points; the modern default
  for embedding visualization. Same caveat: it's a *visualization*, not ground truth.
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

## 8. Calibration & probabilities

Many decisions need *probabilities*, not just rankings. Logistic regression and well-tuned GBMs are roughly
calibrated; SVMs and naive Bayes are not. Check with a **reliability diagram** and **Expected Calibration
Error**; fix with **Platt scaling** (sigmoid) or **isotonic regression** (nonparametric, needs more data),
fit on a held-out calibration set. Calibration is essential whenever you threshold, rank by risk, or combine
model outputs with costs.

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
| Tabular prediction, any size | LightGBM/XGBoost/CatBoost | SOTA, robust, fast |
| Need calibrated probabilities + interpretability | Logistic/linear (regularized) | Calibrated, faithful coefficients |
| Tiny tabular data | Regularized linear, or TabPFN | Strong priors beat capacity |
| High-dim sparse (text counts) | Linear SVM / logistic / naive Bayes | Scales, strong baseline |
| Unknown cluster structure | HDBSCAN | No preset $k$, finds shapes + noise |
| Visualize high-dim embeddings | UMAP (then PCA to sanity-check) | Fast, some global structure |
| Anomaly detection, tabular | Isolation Forest | Scales, few assumptions |
| Strong baseline fast | AutoGluon | Stacked ensemble, minimal tuning |

**Canonical references:** Hastie, Tibshirani & Friedman *Elements of Statistical Learning*; scikit-learn
user guide; XGBoost/LightGBM/CatBoost docs; Bergstra & Bengio 2012 (random search); Grinsztajn et al. 2022
("why trees still beat deep learning on tabular data").
