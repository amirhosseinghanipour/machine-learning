# Evaluation & Statistics

How to measure a model so the number means what you think it means. This is where most ML mistakes happen and
where rigor pays off most (see the Prime Directive in SKILL.md). **A correct evaluation is worth more than a
better model** — a wrong evaluation makes a better model invisible and a worse model look like SOTA.

---

## 1. The discipline: train / validation / test

- **Three splits, three jobs.** *Train*: fit parameters. *Validation*: tune hyperparameters, select
  models/checkpoints, decide architecture. *Test*: estimate generalization **once**, at the very end.
- **The test set is sacred.** Every time you look at the test set and make a decision, it becomes a validation
  set and your estimate becomes optimistic. Touch it once per publishable claim. If you must iterate after
  seeing test performance, you need a fresh test set or a pre-registered change.
- **Selection bias / the multiple-comparisons trap.** The *max* over many noisy validation scores is biased
  upward. If you try 100 configs, the best one's validation score overestimates its true performance — re-
  estimate the winner, and reserve test for the single final choice. This is why "we improved from 91.2 to 91.5"
  after dozens of experiments is usually noise.
- **Adaptive overfitting.** Reusing the same test/benchmark across an entire community/project slowly leaks it.
  Held-out, refreshed, or freshly-collected test sets are the defense (see §8, contamination).

## 2. Splitting correctly (the #1 source of invalid results)

The split must match how the model will be used. Get the **unit of generalization** right:
- **i.i.d. tabular:** stratified random split (preserve class balance). Stratify on the target (and important
  subgroups).
- **Grouped data** (multiple rows per patient/user/document/molecule): **split by group**, never by row, or the
  model memorizes group identity and leaks. Use `GroupKFold`/`StratifiedGroupKFold`.
- **Temporal data** (time series, anything with a time order): **split by time** (train on past, test on
  future) — a random split lets the model "see the future." Use forward-chaining / rolling-origin CV. Never
  shuffle time series.
- **Spatial data:** block/spatial CV to avoid spatial autocorrelation leakage.
- **Molecular/chemical:** scaffold split (novel scaffolds in test) — random splits hugely overestimate (see
  [graph-geometric.md](graph-geometric.md)).
- **Deduplicate and decontaminate first.** Near-duplicates spanning train/test leak. Dedup *before* splitting,
  not after (see [data.md](data.md)).

## 3. Cross-validation

When data is scarce, CV uses it efficiently and estimates variance:
- **k-fold** (5 or 10 typical); **stratified** for classification; **grouped/temporal** variants as above.
  Leave-one-out (k=n) has low bias but high variance and is expensive; 5–10 fold is the usual bias–variance
  sweet spot.
- **Nested CV** when you *both* tune hyperparameters *and* estimate performance: outer loop for the estimate,
  inner loop for tuning. Flat CV (tune and report on the same folds) is optimistically biased — a common error
  (Cawley & Talbot 2010, "on over-fitting in model selection"). The nested estimate evaluates the *procedure*
  (including tuning), which is what you actually deploy.
- **Fit preprocessing inside each fold** (scalers, imputers, feature selection, encoders, target encoding). Any
  step that learns from data and is fit on the full set before CV leaks — feature selection on all data then CV
  is a classic, large-magnitude leak. A scikit-learn `Pipeline` inside the CV split enforces this by
  construction.
- **The CV variance is not what naive formulas say.** Folds share training data, so per-fold errors are
  *positively correlated*; **Bengio & Grandvalet (2004) proved there is no general unbiased estimator of the
  variance of k-fold CV**, and the usual "std across folds / √k" *underestimates* it (often badly), producing
  CIs that are too narrow and t-tests that are anti-conservative. Treat across-fold std as a rough indicator, not
  a calibrated CI; for comparison use the 5×2cv test or Nadeau–Bengio corrected-resampled t (which inflates
  variance to account for train/test overlap). Bates, Hastie & Tibshirani (2023) further show CV estimates the
  *expected* test error of the procedure better than the error of the *one* model you trained, and that the naive
  interval under-covers.
- **One-standard-error rule:** among models within 1 SE of the best CV score, pick the simplest/most-regularized
  — a useful Occam heuristic (Hastie et al.), though note the SE it relies on is the same underestimated quantity
  above, so use it as a tie-breaker, not a significance test.
- **Repeated CV** (different seeds/partitions) to estimate run variance. Report mean ± std across folds *and*
  seeds.
- CV gives a performance *distribution*, not a point — use it for the variance, not just the mean.

## 4. Metrics — choose the one that matches the decision

**Classification:**
- **Accuracy** is misleading under class imbalance (99% accuracy by predicting the majority). Avoid as the
  primary metric for imbalanced problems.
- **Precision / Recall / F1** — for imbalanced or when error types have different costs. Report per-class and
  macro/micro/weighted explicitly (they differ a lot under imbalance). Pick the averaging that matches the goal.
- **ROC-AUC** — threshold-free ranking quality (P(score of random positive > random negative)); invariant to
  class balance, which is exactly why it is **optimistic under heavy imbalance** — it counts the easy true
  negatives. Prefer **PR-AUC (average precision)** when positives are rare and the positive class is the point;
  PR-AUC's baseline is the prevalence (not 0.5), so always state it. Use **partial AUC** when only a region of
  the curve (e.g. low-FPR) matters operationally.
- **MCC (Matthews correlation)** is the most informative single scalar for imbalanced binary problems — it uses
  all four confusion-matrix cells and is only high when the model does well on both classes. **Balanced
  accuracy** = mean per-class recall.
- **Averaging matters under imbalance:** *micro* (pool all examples — dominated by frequent classes), *macro*
  (mean over classes — treats classes equally, dominated by rare-class noise), *weighted* (by support). Report
  the one matching the goal and say which; they can disagree by tens of points.
- **Confusion matrix** — always look at it; it shows *which* errors, not just how many.
- **Threshold selection** is a separate decision from the model: choose the operating point from the PR/ROC
  curve to match the cost trade-off (or by expected cost / Youden's J); don't default to 0.5 — the 0.5 default is
  rarely optimal and is meaningless for an uncalibrated model. Tune the threshold on validation, never test.
- **Calibration & proper scoring** (§7) when you need the probabilities, not just the ranking.

**Regression:** RMSE (penalizes large errors, optimal point forecast = conditional mean, same units), MAE
(robust to outliers, optimal forecast = conditional median), MAPE (relative, beware zeros and asymmetry — it
penalizes over-prediction more, biasing toward under-forecasting; use sMAPE or MASE for scale-free comparison
across series), $R^2$ (variance explained, can be negative — and on a *non-i.i.d.* split it is not the
training-time $R^2$). Match the metric to the loss that matters and to the decision; **plot residuals vs.
predictions and vs. key features** to find heteroscedasticity, bias, and missed structure that aggregate
numbers hide. For probabilistic regression use CRPS / pinball loss (see §7).

**Ranking / retrieval / recommendation:** Precision@k, Recall@k, **MRR**, **MAP**, **nDCG** (graded relevance,
position-discounted — the default for graded judgments). Evaluate at the k that matches the product surface.
**Bootstrap by query**, and watch for *missing-judgment* bias: unjudged retrieved items are often scored as
non-relevant, which unfairly penalizes systems that surface novel relevant docs (a known offline-eval trap that
also afflicts RAG and recommender offline metrics — corroborate with online/interleaving where possible).

**Generative:** FID/precision-recall/CLIPScore (images), FAD (audio), perplexity (text within identical
tokenizer) — see [generative-models.md](generative-models.md). No single number; report several + human eval.

**NLP/LLM:** task-specific verifiable accuracy where possible (exact match, pass@k for code with execution);
BLEU/ROUGE/METEOR for MT/summarization (weak, surface-level — supplement with human/LLM judging); BERTScore
(embedding-based). For open-ended generation, automated n-gram metrics are weak proxies.

**Segmentation/detection (CV):** IoU/Dice, mAP (see [domains.md](domains.md)).

**The meta-rule:** define a **single primary metric** tied to the real decision up front; track secondary
metrics; never silently switch the primary metric to the one that looks best ("metric hacking").

## 5. Quantifying uncertainty in your metric (do this for every headline number)

A metric from a finite test set is an *estimate* with variance. Report it as such. There are **two independent
noise sources** — test-set sampling and training stochasticity — and the honest error bar reflects both.

- **Confidence intervals via bootstrap:** resample the test set with replacement many times (≥1,000 for a point
  CI, ≥10,000 for tail/p-value estimates), recompute the metric, take the 2.5/97.5 percentiles. Works for almost
  any metric, no distributional assumptions. `scripts/compare_models.py` does this.
  - **Which interval:** the **percentile** interval is fine for symmetric, near-unbiased metrics; for skewed or
    biased statistics (AUC near 1, ratios, small samples) prefer **BCa** (bias-corrected and accelerated) — it has
    second-order accuracy and corrects for skew and bias, at modest extra cost. Below ~20 positives, even BCa is
    shaky; fall back to an exact/analytic interval (e.g. DeLong for AUC, Clopper–Pearson for a proportion).
  - **Respect the dependence structure:** if examples are clustered (multiple rows per user/document), resample
    **clusters, not rows** (cluster/block bootstrap) — naive row resampling treats correlated examples as
    independent and gives intervals that are far too narrow.
  - **Bootstrapping the wrong unit is the most common CI error:** for ranking/retrieval metrics resample
    **queries**, not query–document pairs; for grouped classification resample groups.
- **Analytic shortcuts** when valid: a binomial/Wilson interval for accuracy on i.i.d. examples; DeLong for
  ROC-AUC and its difference. These are cheap sanity checks on the bootstrap.
- **Across seeds:** for stochastic training (init, data order, dropout, augmentation, nondeterministic kernels),
  the dominant variance is often *between training runs*, not within a test set — frequently larger than the
  effect being claimed. Report **mean ± std (or 95% CI) over ≥3–5 seeds** (≥5–10, and IQM with stratified
  bootstrap CIs per Agarwal et al. 2021 *rliable*, for RL — see
  [reinforcement-learning.md](reinforcement-learning.md)). A single-seed number is not a result. Report the
  **std of the mean** ($s/\sqrt{n_\text{seeds}}$) only if you mean the precision of the mean; report the **std**
  itself if you mean run-to-run spread — say which, and prefer showing both.
- **Do not summarize a noisy run by its best step.** Report the final or val-early-stopped checkpoint and show
  the curve; "best test step over training" is test-set selection (see §1).

## 6. Statistical significance — is the difference real?

Comparing model A vs. B on the same test set:
- **Use a paired test** (the models see the same examples) — it cancels per-example difficulty and is far more
  powerful than comparing two independent means. The right unit of pairing is the **example** (or cluster).
  - **Paired bootstrap on the difference** (resample the same indices for both models, build a CI on $A-B$, and/
    or count how often $A>B$) — general, robust, recommended default; report the CI on the gap, not just a p.
  - **Permutation/randomization test** — under the null "the two models are exchangeable on each example," swap
    each example's A/B predictions with prob ½, recompute the metric gap, repeat ≥10,000×; the p-value is the
    fraction of permuted gaps ≥ observed. Exact-ish, assumption-light, the cleanest null for most metrics.
  - **McNemar's test** for paired binary classification — it conditions on the *discordant* pairs (cases where
    exactly one model is right); use the exact binomial form when discordant count is small. Recommended by
    Dietterich (1998) as having low Type-I error for classifier comparison on a single test set.
  - **5×2cv paired t-test** (Dietterich) when you can afford retraining — accounts for training variance that a
    single-split test ignores; **Wilcoxon signed-rank** across folds/seeds (nonparametric, robust).
  - **DeLong test** for the difference of two correlated ROC-AUCs (analytic, paired).
- **Report effect size and CIs, not just p-values.** "Significant" ≠ "meaningful"; a tiny, significant gain can
  be practically irrelevant, and with a large test set everything is "significant." Prefer reporting the
  difference with a 95% CI; add a standardized effect size (Cohen's *d*, or rank-based Cliff's δ / A-vs-B win
  rate) when comparing across tasks of different scales.
- **A p-value over the test set does not absolve test-set reuse.** Significance testing answers "is the gap real
  given this test set," not "did I overfit the test set by trying many models." Both must hold.
- **Correct for multiple comparisons** when testing many models/benchmarks/hyperparameters (Bonferroni controls
  the family-wise error rate but is conservative; **Holm** is uniformly better; **Benjamini–Hochberg** controls
  the false discovery rate and is the right choice when you expect several true effects). Testing 20 things at
  p<0.05 yields ~1 false positive by chance — this is exactly how leaderboards manufacture "winners."
- **Independence assumptions:** examples within a group/document/user are correlated — naive tests overstate
  significance. Pair and resample at the cluster level (cluster bootstrap / block permutation).
- **For learning curves / many benchmarks:** aggregate carefully — a **Friedman test** followed by average ranks
  + **critical-difference diagrams** with Nemenyi (or Wilcoxon–Holm) post-hoc (Demšar 2006) across datasets beats
  averaging raw scores of different scales, which is dominated by whichever benchmark has the widest range.

## 7. Proper scoring rules & calibration

**Use proper scoring rules to grade probabilities.** A scoring rule is *proper* if it is minimized (in
expectation) by reporting the true probability, and *strictly proper* if that minimizer is unique — so it gives
the model no incentive to hedge or game. The two workhorses:
- **Log loss / NLL** ($-\sum y\log\hat p$): strictly proper, the cross-entropy you already train on. Unbounded —
  one confident wrong prediction ($\hat p\to 0$ on the true class) sends it to $\infty$, so it is dominated by
  tail mistakes and sensitive to clipping ($\epsilon$).
- **Brier score** (mean squared error on probabilities): strictly proper, bounded in $[0,2]$ (binary), more
  robust to outliers than NLL. **Accuracy, AUC, F1, precision/recall are *not* proper** — they ignore or distort
  the probability and cannot detect miscalibration, so never tune probabilities on them.
- **Decomposition.** Every proper score splits into **calibration + refinement** (Brier: reliability −
  resolution + uncertainty; the Murphy decomposition). Refinement (a.k.a. discrimination/sharpness) rewards
  separating classes; calibration rewards honest probabilities. A model can have great AUC (refinement) and
  terrible calibration simultaneously — they measure different things. The **Triptych** (reliability diagram +
  ROC/Murphy curve + score-decomposition plot) shows both at once.

**Calibration** = a model's confidence matches its accuracy. Essential when you threshold, rank by risk, defer to
humans, abstain, or feed probabilities into downstream decisions/expected-cost calculations.
- **Diagnose:** reliability diagram (predicted prob vs. empirical accuracy, with binwise CIs), **Expected
  Calibration Error (ECE)**, plus a proper score (Brier/NLL). Modern deep nets are typically **overconfident**
  (Guo et al. 2017); large-vocabulary LLMs are often *better* calibrated on next-token but mis-calibrated after
  RLHF.
- **ECE is a biased, fragile estimator — do not report it naively.** It depends entirely on the binning scheme:
  equal-**width** bins leave high-confidence bins nearly empty (high variance), equal-**count**/quantile bins
  (Adaptive Calibration Error, ACE) reduce bias; too few bins underestimate error, too many inflate it via noise.
  ECE is also **not a proper scoring rule** and can be gamed (a constant predictor of the base rate has ECE 0).
  Standard binned ECE is a *downward-biased* estimate of true calibration error. Prefer **debiased / kernel
  estimators** (e.g., the consistent, asymptotically-unbiased proper-calibration-error estimators of Popordanoska
  et al. 2024; kernel-based KCE / ECE_KDE) and always report the binning choice. For multiclass, distinguish
  **top-label** (confidence) calibration, **class-wise** calibration, and full multiclass calibration — they are
  not the same, and the strong "canonical" notion is statistically hard to estimate in high dimensions.
- **Fix (post-hoc, fit on a held-out calibration split — never train or test):** **temperature scaling** (single
  scalar dividing the logits — the standard, near-free fix for neural nets; preserves accuracy and the argmax),
  Platt scaling (binary), isotonic regression (nonparametric, monotone, needs more data, can overfit), histogram/
  Bayesian binning (BBQ), Dirichlet calibration (multiclass). Deep ensembles and (focal-loss / label-smoothing)
  training-time methods also improve calibration (see [probabilistic-ml.md](probabilistic-ml.md)).
- **Under distribution shift, calibration degrades first and most** — temperature fit on the val set does not
  transport to OOD. Validate calibration on shifted/OOD data if you will deploy there.
- **For regression, calibration means coverage:** a 90% predictive interval should contain the truth ~90% of the
  time. Diagnose with PIT histograms / quantile-coverage plots; grade probabilistic forecasts with the **CRPS**
  (the proper-scoring-rule generalization of MAE) or pinball/quantile loss. Conformal prediction gives
  distribution-free coverage guarantees (see [probabilistic-ml.md](probabilistic-ml.md)).

## 8. Benchmarking & contamination (especially for LLMs/foundation models)

- **Contamination:** if test data (or near-duplicates/rephrasings) leaked into (pre)training, you measure
  **memorization, not capability**. Endemic for web-pretrained models on public benchmarks; a benchmark that has
  existed for >1 year is presumed contaminated for frontier models unless shown otherwise.
- **Detect:** n-gram/substring overlap between benchmark and training corpus (use a scalable index — Infini-gram
  / suffix-array over the corpus when you have it); **Min-K%-Prob** and **Min-K%++** (contaminated examples have
  unusually high token probabilities / few low-prob tokens); **ConStat** and rephrase tests (a real-capability
  model should be ~unchanged on a faithful rephrase — a large drop signals memorized surface form);
  order-sensitivity / option-shuffling on MCQ (memorized answers track the original letter); guided-completion /
  "quiz the model on the *next* example" and "time-travel" attacks; canary strings (BIG-bench); and **temporal
  cliffs** — plot accuracy vs. example creation date and look for a cliff at the model's training cutoff (the
  GSM8k-vs-GSM1k gap is the canonical demonstration: models that memorized GSM8k drop on a fresh, distribution-
  matched re-collection).
- **Defend:** prefer **held-out, private, or frequently-refreshed/dynamic** benchmarks (LiveBench,
  LiveCodeBench, FrontierMath, MMLU-Pro, SWE-bench Verified, ARC-AGI-2) over static public leaderboards; collect
  or generate fresh test data **after** the model's cutoff; keep a private split; and report contamination
  analysis as part of the eval. Note dynamic benchmarks trade contamination-resistance for harder longitudinal
  comparability (the target moves) — version and date them. "Inference-time decontamination" (regenerate leaked
  items) can partially rescue a contaminated benchmark.
- **Benchmark design:** verifiable ground truth where possible; difficulty spread (avoid ceiling/floor — a
  saturated benchmark measures nothing); multiple domains/slices; an honest human and simple baseline *and* a
  topline; report **per-item agreement and CIs**, not just the mean; documented, versioned protocol (prompt,
  few-shot examples, parsing/normalization, decoding params) so others can reproduce the *exact* number — most
  cross-paper LLM disagreements are harness differences (prompt format, answer extraction, 0- vs few-shot), not
  model differences. Re-run baselines in your own harness; never copy numbers across harnesses.
- **LLM-as-judge** (scalable, but a noisy, biased rater — validate it like any classifier):
  - **Known biases:** *position/order* bias (randomize or average over both orders in pairwise; swapping order
    flips many verdicts); *verbosity/length* bias (judges prefer longer answers — control for length); *self-
    preference / egocentric* bias (models rate their own family higher, correlated with the judge's own
    perplexity / familiarity — never let a model judge its own outputs); *style-over-substance* (fluent, confident,
    well-formatted but wrong beats terse-correct); *sycophancy* and anchoring to a provided reference;
    *preference leakage* (Li et al. 2025) — a judge sharing lineage/training data with the candidate inflates it,
    a contamination not fixed by hiding identity.
  - **Mitigate:** pairwise (relative) with order-swapping and ties allowed is more reliable than absolute
    1–10 scoring (poorly anchored, drifts); use **explicit rubrics / reference answers / decomposed checklists**;
    constrain to verifiable judgments where possible (mix unit tests + rubric for code); ensemble heterogeneous
    judges or use a jury (panel) to dilute single-model bias; and **report judge–human agreement** (Cohen's κ or
    correlation on a human-labeled subset) — an unvalidated judge is an unvalidated metric. Treat the judge's
    output as data with its own variance and contamination risk, not as ground truth.

## 9. Error analysis & slicing (beyond the aggregate)

The aggregate metric hides where the model fails — and the failures are usually the point.
- **Slice metrics** by subgroup, difficulty, domain, length, rare classes, and known hard cases. A model that's
  95% overall but 40% on the safety-critical slice is a failure. Report worst-group performance, not just
  average.
- **Inspect actual errors:** read the misclassified/worst examples. Patterns reveal label noise, leakage,
  spurious correlations, or a fixable gap. This qualitative loop drives more improvement than hyperparameter
  tuning.
- **Counterfactual / robustness checks:** perturb inputs (typos, paraphrases, lighting, background) and measure
  stability (see [interpretability-safety.md](interpretability-safety.md)).
- **Compare to a strong baseline on the same slices** — relative error patterns are more informative than
  absolute numbers.

## 10. The evaluation checklist

- [ ] Split matches the unit of generalization (group/time/stratify), deduped & decontaminated first.
- [ ] Primary metric chosen up front and matches the real decision; imbalance-aware if needed.
- [ ] Preprocessing fit inside CV folds; nested CV if tuning + estimating.
- [ ] Test set used once; selection done on validation; multiple-comparison bias acknowledged.
- [ ] Mean ± CI over ≥3 seeds (≥5–10 for RL); **paired** significance test (bootstrap/permutation/McNemar) for
      headline claims; effect size + CI on the gap reported, not just a p-value; multiple-comparison correction
      if many configs/benchmarks.
- [ ] CIs bootstrapped at the correct unit (cluster/query/group), not the row, when examples are dependent.
- [ ] Probabilities graded with a **proper scoring rule** (Brier/NLL) and calibration checked with a debiased
      estimator (not naive binned ECE) if probabilities drive decisions.
- [ ] Sliced/worst-group metrics + error analysis on real failures, not only the aggregate.
- [ ] Contamination checked for foundation-model benchmarks (n-gram/Min-K%/rephrase/temporal); held-out/fresh
      benchmarks preferred; LLM-judge validated against human labels and de-biased (order/length/self-preference).
- [ ] Eval harness validated: reproduces a known number; a label-shuffled control scores at chance; baselines
      re-run in the *same* harness (no copied cross-harness numbers).

**Canonical references:** Raschka 2018 ("Model Evaluation, Model Selection, and Algorithm Selection");
Demšar 2006 (statistical comparison across datasets); Dietterich 1998 (statistical tests for classifiers; 5×2cv,
McNemar); Bengio & Grandvalet 2004 (no unbiased variance estimator for k-fold CV); Nadeau & Bengio 2003
(corrected resampled t); Bates, Hastie & Tibshirani 2023 ("CV: what does it estimate and how well"); Cawley &
Talbot 2010 (over-fitting in model selection); Gneiting & Raftery 2007 (proper scoring rules); Guo et al. 2017
(calibration of modern neural nets); Nixon et al. 2019 (measuring calibration / ACE); Popordanoska et al. 2024
(consistent proper calibration-error estimators); Dimitriadis, Gneiting & Jordan 2021 ("Triptych" for
probabilistic classifiers); Efron & Tibshirani 1993 (bootstrap, BCa); Agarwal et al. 2021 (rliable — reliable RL
evaluation); Zheng et al. 2023 (MT-Bench / LLM-as-judge biases); Li et al. 2025 (preference leakage in
LLM-as-judge); Zhang et al. 2024 (GSM1k — contamination via fresh re-collection); Kapoor & Narayanan 2023
("Leakage and the Reproducibility Crisis in ML-based Science").
