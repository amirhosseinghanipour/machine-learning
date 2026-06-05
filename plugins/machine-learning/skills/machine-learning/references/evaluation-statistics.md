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
- **Nested CV** when you *both* tune hyperparameters *and* estimate performance: outer loop for the estimate,
  inner loop for tuning. Flat CV (tune and report on the same folds) is optimistically biased — a common error.
- **Fit preprocessing inside each fold** (scalers, imputers, feature selection, encoders). Fitting on the full
  data before CV leaks. A scikit-learn `Pipeline` enforces this by construction.
- **Repeated CV** (different seeds) to estimate run variance. Report mean ± std across folds *and* seeds.
- CV gives a performance *distribution*, not a point — use it for the variance, not just the mean.

## 4. Metrics — choose the one that matches the decision

**Classification:**
- **Accuracy** is misleading under class imbalance (99% accuracy by predicting the majority). Avoid as the
  primary metric for imbalanced problems.
- **Precision / Recall / F1** — for imbalanced or when error types have different costs. Report per-class and
  macro/micro/weighted explicitly (they differ a lot under imbalance). Pick the averaging that matches the goal.
- **ROC-AUC** — threshold-free ranking quality; **but optimistic under heavy imbalance** — prefer **PR-AUC
  (average precision)** when positives are rare.
- **Confusion matrix** — always look at it; it shows *which* errors, not just how many.
- **Threshold selection** is a separate decision from the model: choose the operating point from the PR/ROC
  curve to match the cost trade-off; don't default to 0.5.
- **Calibration** (§7) when you need the probabilities, not just the ranking.

**Regression:** RMSE (penalizes large errors, same units), MAE (robust to outliers), MAPE (relative, beware
zeros), $R^2$ (variance explained, can be negative). Match to the loss that matters; plot residuals vs.
predictions to find heteroscedasticity/bias.

**Ranking / retrieval / recommendation:** Precision@k, Recall@k, **MRR**, **MAP**, **nDCG** (graded relevance,
position-discounted). Evaluate at the k that matches the product.

**Generative:** FID/precision-recall/CLIPScore (images), FAD (audio), perplexity (text within identical
tokenizer) — see [generative-models.md](generative-models.md). No single number; report several + human eval.

**NLP/LLM:** task-specific verifiable accuracy where possible (exact match, pass@k for code with execution);
BLEU/ROUGE/METEOR for MT/summarization (weak, surface-level — supplement with human/LLM judging); BERTScore
(embedding-based). For open-ended generation, automated n-gram metrics are weak proxies.

**Segmentation/detection (CV):** IoU/Dice, mAP (see [domains.md](domains.md)).

**The meta-rule:** define a **single primary metric** tied to the real decision up front; track secondary
metrics; never silently switch the primary metric to the one that looks best ("metric hacking").

## 5. Quantifying uncertainty in your metric (do this for every headline number)

A metric from a finite test set is an *estimate* with variance. Report it as such.
- **Confidence intervals via bootstrap:** resample the test set with replacement many times (≥1,000–10,000),
  recompute the metric, take the 2.5/97.5 percentiles. Works for any metric, no distributional assumptions.
  `scripts/compare_models.py` does this.
- **Across seeds:** for stochastic training, the dominant variance is often *between training runs*, not within
  a test set. Report **mean ± std (or 95% CI) over ≥3–5 seeds** (≥5–10 for RL — see
  [reinforcement-learning.md](reinforcement-learning.md)). A single-seed number is not a result.
- **Both sources matter:** test-set sampling variance *and* training stochasticity. The honest error bar
  combines them.

## 6. Statistical significance — is the difference real?

Comparing model A vs. B on the same test set:
- **Use a paired test** (the models see the same examples) — much more powerful than unpaired.
  - **Paired bootstrap** (resample examples, count how often A beats B) — general, robust, recommended default.
  - **Permutation/randomization test** — swap predictions between models, build the null distribution.
  - **McNemar's test** for paired binary classification; **paired t / Wilcoxon signed-rank** across folds/seeds.
- **Report effect size and CIs, not just p-values.** "Significant" ≠ "meaningful"; a tiny, significant gain can
  be practically irrelevant. Prefer reporting the difference with a 95% CI.
- **Correct for multiple comparisons** when testing many models/benchmarks/hyperparameters (Bonferroni is
  conservative; Benjamini–Hochberg controls FDR). Testing 20 things at p<0.05 yields ~1 false positive by
  chance.
- **Independence assumptions:** examples within a group/document/user are correlated — naive tests overstate
  significance. Account for clustering (cluster bootstrap).
- **For learning curves / multiple benchmarks:** aggregate carefully — average ranks + critical-difference
  diagrams (Demšar) across datasets beats averaging raw scores of different scales.

## 7. Calibration & reliability

A model's confidence should match its accuracy. Essential when you threshold, rank by risk, defer to humans, or
feed probabilities into downstream decisions.
- **Diagnose:** reliability diagram (predicted prob vs. empirical accuracy), **Expected Calibration Error
  (ECE)**, Brier score, negative log-likelihood. Modern deep nets are typically **overconfident**.
- **Fix:** temperature scaling (single parameter, fit on validation — the standard, simple fix for neural nets),
  Platt scaling, isotonic regression. Deep ensembles improve calibration too (see
  [probabilistic-ml.md](probabilistic-ml.md)).
- **Under distribution shift, calibration degrades** — validate calibration on shifted/OOD data if you'll deploy
  there.

## 8. Benchmarking & contamination (especially for LLMs/foundation models)

- **Contamination:** if test data (or near-duplicates/rephrasings) leaked into (pre)training, you measure
  **memorization, not capability**. Endemic for web-pretrained models on public benchmarks.
- **Detect:** n-gram/substring overlap between benchmark and training corpus, **Min-K% probability**, ConStat
  (performance gap on rephrased vs. original), guided-completion / "time-travel" attacks, canary strings, and
  time-based splits (a sudden cliff at the training cutoff date signals contamination).
- **Defend:** prefer **held-out, private, or frequently-refreshed/dynamic** benchmarks (LiveBench,
  LiveCodeBench, FrontierMath, MMLU-Pro) over static public leaderboards; collect fresh test data post-cutoff;
  report contamination analysis as part of the eval.
- **Benchmark design:** verifiable ground truth where possible; difficulty spread; multiple domains/slices; an
  honest human/simple baseline and a topline; documented protocol so others can reproduce the exact number.
- **LLM-as-judge** (scalable evaluation) is biased: position bias (randomize order), verbosity/length bias,
  self-preference (don't let a model judge its own family), and style-over-substance. Calibrate against human
  labels, use rubrics, ensemble judges, and report judge–human agreement.

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
- [ ] Mean ± CI over ≥3 seeds (≥5–10 for RL); paired significance test for headline claims; effect size reported.
- [ ] Calibration checked if probabilities are used.
- [ ] Sliced/worst-group metrics + error analysis, not only the aggregate.
- [ ] Contamination checked for foundation-model benchmarks; held-out/fresh benchmarks preferred.
- [ ] Eval harness validated: reproduces a known number; a label-shuffled control scores at chance.

**Canonical references:** Raschka 2018 ("Model Evaluation, Model Selection, and Algorithm Selection");
Demšar 2006 (statistical comparison across datasets); Dietterich 1998 (statistical tests for classifiers);
Guo et al. 2017 (calibration of modern neural nets); Agarwal et al. 2021 (rliable — reliable RL evaluation);
Kapoor & Narayanan 2023 ("Leakage and the Reproducibility Crisis in ML-based Science").
