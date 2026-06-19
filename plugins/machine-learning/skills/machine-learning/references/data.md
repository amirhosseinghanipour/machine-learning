# Data: Curation, Splitting, Leakage, and Augmentation

Data work is most of real ML and the source of most invalid results. The model is downstream of the data; a
better dataset usually beats a better model. This file is about getting the data right — and about **leakage**,
the single most common cause of results that don't replicate (see the Prime Directive in SKILL.md).

---

## 1. The data-centric mindset

- **Fix the data before the model.** On most applied problems, label-noise cleaning, deduplication, better
  coverage, and feature quality move the metric more than architecture changes. Profile and improve the data
  systematically (Andrew Ng's data-centric AI framing).
- **Garbage in, garbage out — but worse:** ML *amplifies* data problems (biases, leakage, label noise become
  confident wrong predictions). Audit data as carefully as you'd review code.
- **Know your data distribution** intimately: look at raw examples, distributions of every feature, label
  balance, missingness patterns, and the tails. The hour spent reading 100 random examples is the highest-ROI
  hour in a project.

## 2. Dataset construction & curation

- **Define the population** you want to generalize to, and sample to match it. Convenience samples (whatever was
  easy to collect) bake in selection bias that surfaces as deployment failure.
- **Sources & coverage:** ensure the data spans the conditions, subgroups, and edge cases of deployment. Missing
  slices = silent failure on those slices (see worst-group evaluation in
  [evaluation-statistics.md](evaluation-statistics.md)).
- **Pretraining-scale curation** (for foundation models): the pipeline that actually moves the needle is, in
  rough order, language ID → quality filtering → **dedup** → decontamination → toxicity/PII removal → mixing.
  Quality filtering combines cheap heuristics (Gopher/C4 rules: length, symbol/word ratios, boilerplate, bullet
  fraction) with **model-based** scoring (fastText/classifier "is this like high-quality reference text," or
  perplexity vs. a reference LM); FineWeb-Edu showed an education classifier filter beats raw web by a wide
  margin at equal tokens. Data quality and dedup dominate downstream quality — more than a marginally better
  architecture. But filtering is **selection**: an over-aggressive quality classifier bakes in the reference
  corpus's demographic/topic bias — audit what gets dropped.
- **Document the data** with a **datasheet** (Gebru et al.) and/or **data card**: provenance, collection process
  and dates, intended use, composition, preprocessing, licensing/consent, and known biases. Emit **machine-
  readable metadata** — the **Croissant** format (MLCommons 2024; supported by Hugging Face, Kaggle, OpenML,
  Google Dataset Search; **Croissant-RAI** extends it with responsible-AI fields) makes a dataset loadable and
  discoverable across tools, not just human-documented. Required for responsible release and for reviewers (see
  [research-workflow.md](research-workflow.md)).

## 3. Labeling & label quality

- **Label noise is everywhere** (Northcutt et al. 2021 found %-level error rates across ImageNet, CIFAR, MNIST,
  QuickDraw, and several NLP benchmarks; correcting test labels even **reorders model rankings**). It caps
  achievable accuracy and corrupts evaluation. Estimate it (relabel a sample; **confident learning** / Cleanlab
  to flag likely errors via the joint distribution of noisy and predicted labels) and clean the **test set**
  especially — a noisy test set hides real progress and can crown the wrong model. Distinguish *random* noise
  (caps accuracy, roughly symmetric) from *systematic* mislabeling (biases the model and is far more dangerous).
- **Annotation protocol:** clear guidelines with examples and edge cases, a calibration round, multiple
  annotators on an overlap subset, and measured **inter-annotator agreement** — use **Cohen's/Fleiss' κ** (or
  Krippendorff's α for missing data / ordinal labels), *not* raw percent agreement, which ignores chance. Low
  agreement means the **task is ill-defined** — fix the label definition before scaling labeling (more
  annotators won't rescue an ambiguous taxonomy). Adjudicate disagreements; treat the adjudicated/consensus set
  as a higher-quality eval. Watch for annotator-identity confounds (one annotator labels one class) which become
  leakage.
- **Weak/programmatic supervision** (Snorkel-style labeling functions), active learning (label the most
  informative examples — uses epistemic uncertainty, see [probabilistic-ml.md](probabilistic-ml.md)), and
  semi-supervised learning stretch limited labels — see [learning-paradigms.md](learning-paradigms.md) for the
  methods (FixMatch, query strategies, weak supervision).
- **LLM-generated labels/data** are now common and cheap but introduce the labeler model's biases and errors —
  validate against human labels, and never evaluate a model on labels produced by a model of the same family.

## 4. Leakage — the cardinal sin (audit every project for this)

Leakage = information available at training that **won't be available at prediction time**, making validation
look great and deployment collapse. Forms:
- **Target leakage:** a feature is a proxy for / derived from the label (e.g., "account_closed_date" predicting
  churn; a post-outcome measurement). Ask of every feature: *was this knowable before the label existed?*
- **Train/test contamination:** the same or near-duplicate examples in both splits (duplicates, augmented copies,
  the same user/document/image under different IDs). **Deduplicate before splitting.**
- **Temporal leakage:** using future information to predict the past — random splits of time-ordered data, or
  features computed over a window that includes the prediction time. Split by time.
- **Group leakage:** rows from the same entity (patient, user, session) split across train and test → the model
  memorizes the entity. Split by group.
- **Preprocessing leakage:** fitting scalers, imputers, feature selectors, vocabularies, or target encoders on
  the **full dataset** before splitting (or outside CV folds). Fit on train only, inside folds — use a pipeline.
- **Evaluation/foundation-model contamination:** benchmark data in the pretraining corpus → memorization, not
  skill (see [evaluation-statistics.md](evaluation-statistics.md) §8).
- **The test:** if validation is suspiciously high or "too easy," assume leakage until proven otherwise. Kapoor
  & Narayanan (2023) catalog leakage as a cause of a reproducibility crisis across ML-based science.

## 5. Splitting (do it once, do it right)

The split must mirror deployment — covered in depth in [evaluation-statistics.md](evaluation-statistics.md) §2.
The data-side rules:
- **Deduplicate and decontaminate FIRST**, then split. Near-duplicate detection before any split:
  - **Exact dup:** hash the normalized content (e.g. SHA-1 of whitespace/case-normalized text) and drop repeats.
  - **Near-dup (text):** **MinHash + LSH** — shingle each doc into k-grams (word 5-grams typical), compute a
    MinHash signature (N permutations) estimating Jaccard similarity, then LSH-band the signatures so only
    candidate pairs above a similarity threshold (commonly ~0.7–0.8 Jaccard) are compared. SimHash is the cheaper
    Hamming-distance alternative. This is what dedups web-scale corpora (Lee et al. 2021 showed dedup reduces
    memorization and *improves* LMs at equal compute). For decontamination against benchmarks, also do **n-gram
    / substring** matching (e.g. 8–13-gram or 50-char overlap) — and at scale, an **Infini-gram / suffix-array**
    index over the training corpus lets you query benchmark items exactly.
  - **Near-dup (images/embeddings):** perceptual hashing or embedding cosine-similarity with an ANN index (FAISS)
    above a threshold.
  - Near-exact matching misses paraphrase/translation contamination — supplement with embedding similarity or a
    rephrase test (see [evaluation-statistics.md](evaluation-statistics.md) §8).
- **Choose the unit:** group/temporal/stratified/scaffold as the problem demands — match the unit of
  generalization.
- **Freeze the test set** immediately and store it separately; resist the urge to "just check."
- **Record the split** (the exact IDs or a deterministic, seeded, documented procedure) so it's reproducible.

## 6. Class imbalance & cost-sensitive learning

- **Diagnose first:** is it imbalanced *and* are the rare cases what matter (fraud, disease)? Then accuracy is
  the wrong metric (use PR-AUC, recall@precision, per-class — see
  [evaluation-statistics.md](evaluation-statistics.md)).
- **Levers, in order:** class weights / `scale_pos_weight` / focal loss (cheap, keeps data real); threshold
  tuning on the PR curve (often the biggest, freest win); **then** resampling if needed.
- **Resampling caveats:** random oversampling overfits the minority; **SMOTE** interpolates synthetic minority
  points (can create unrealistic ones / blur the boundary, especially in high-dim or mixed categorical-numeric
  data — validate, and apply it inside the pipeline so it can't fit on held-out folds); undersampling discards
  data. **Resample only the training folds, never validation/test** (resampling the test set fabricates the
  metric and is a common, invalidating bug) — and SMOTE *before* the split is leakage (synthetic points
  interpolate across the split boundary). Recent evidence: with a well-calibrated model and a tuned threshold,
  resampling often gives little over class weights — try the cheap levers first.
- **Calibration breaks under resampling** — you changed the base rate, so the model's probabilities no longer
  match deployment. Recalibrate on a natural-prevalence set, or apply a prior-correction / adjust the threshold
  back to the true base rate (see [evaluation-statistics.md](evaluation-statistics.md) §7).

## 7. Data augmentation (the highest-leverage regularizer)

Augmentation injects the **invariances** you want (see [deep-learning.md](deep-learning.md) §1, §2) and is often
worth more than any architecture tweak — but only the augmentations that preserve the label.
- **Vision:** flips, crops, color jitter, RandAugment/AutoAugment, **MixUp/CutMix** (interpolate inputs+labels),
  random erasing. Choose label-preserving ops (don't horizontally flip text/digits where orientation matters).
- **NLP/text:** back-translation, synonym/EDA, token masking/dropout, and increasingly **LLM-based paraphrase/
  synthetic data**. Surface edits risk changing meaning — validate.
- **Audio:** SpecAugment (time/freq masking), noise/reverb/pitch/speed perturbation, time-shift.
- **Tabular:** harder (no natural invariances) — SMOTE-like, noise injection, mixup, or generative models;
  often less effective than for perceptual data.
- **Test-time augmentation (TTA):** average predictions over augmented copies for a small inference-time boost.
- **The rule:** every augmentation asserts "this transformation shouldn't change the label." If that's false for
  your task, it's harmful. Match augmentations to the real nuisance variation in deployment.

## 8. Synthetic & simulated data

- **When:** scarce/expensive/sensitive real data, rare events, privacy constraints, sim-to-real (robotics),
  and bootstrapping (LLM-generated SFT/preference data — now standard, see
  [transformers-llms.md](transformers-llms.md)).
- **The gap:** synthetic data has a distribution gap from reality (sim-to-real gap); domain randomization and
  domain adaptation help bridge it. **Validate on real held-out data** — never let synthetic data into the test
  set.
- **Model-generated-data risks:** training recursively on a model's own outputs causes **model collapse**
  (Shumailov et al. 2024) — variance shrinks, tails (rare events) vanish, distributions drift to the mean over
  generations, and bias amplifies. The mitigation is to **accumulate** real data alongside synthetic rather than
  *replace* it (collapse is largely driven by replacing the real distribution), keep a fixed real anchor, and
  monitor output diversity/entropy. Synthetic data works best when the generator can do something the trainee
  can't yet (distillation, verified solutions, harder-to-generate-than-verify tasks), not as a free data
  multiplier.
- **Privacy:** synthetic data is not automatically private — generators can memorize and leak training examples
  (see [interpretability-safety.md](interpretability-safety.md)); use DP generation if privacy is the goal.

## 8a. Data and the compute budget (scaling-law considerations)

When data is a planned quantity (foundation models, large training runs), treat it as a budgeted resource:
- **Compute-optimal allocation.** Chinchilla (Hoffmann et al. 2022): for dense LLMs the compute-optimal split is
  roughly **~20 tokens per parameter** — most pre-Chinchilla models were badly *undertrained* (too big, too
  little data). Plan in tokens/FLOPs, not epochs (see [transformers-llms.md](transformers-llms.md)). Inference
  cost shifts the optimum: if you'll serve the model a lot, train a *smaller* model on *more* data than
  compute-optimal.
- **Data-constrained regime.** When unique data runs out (Muennighoff et al. 2023): **repeating data up to ~4
  epochs is nearly as good as fresh unique data**; beyond that the value of repeated tokens decays fast and
  added compute eventually contributes ~nothing. So when tokens are the bottleneck, modest repetition + adding
  parameters beats hammering the same data — and repeating *intelligently selected* data can beat fresh random
  data. Account for this with an "effective tokens" discount, not raw token counts.
- **Dedup interacts with scaling.** Train/test n-gram overlap and intra-corpus duplication distort scaling-law
  fits and inflate apparent performance (memorization), and heavy duplication can *break* a clean scaling trend
  above ~100M params. Deduplicate before measuring scaling.
- **Quality > quantity at the margin.** A better data filter shifts the whole loss-vs-compute curve down — often
  a larger lever than more tokens. But filtering reduces the unique-token budget, pushing you sooner into the
  repetition regime; co-design filter strength and epoch count.
- **Mixture weights matter.** Domain mixing proportions (code/web/books/math) materially change downstream skills
  and have their own (per-domain) scaling behavior; tune the mixture, and note that the optimal mixture can shift
  with scale.

## 9. Preprocessing & feature pipelines

- **Make preprocessing part of the model** (a `Pipeline`/transform fit on train only) so it's applied
  identically at train and inference and can't leak. Mismatched train/serve preprocessing ("training–serving
  skew") is a top production bug.
- **Numerics:** scaling/normalization (fit on train), robust scaling for outliers, log/Box-Cox for skew, careful
  handling of outliers (clip vs. remove vs. keep — decide deliberately).
- **Missingness:** impute (with an indicator) or use models that handle NaNs natively; missingness is often
  informative — model it, don't just fill it.
- **Versioned, deterministic pipelines:** the same input must always produce the same features; log the pipeline
  version with the run (see [experimentation-reproducibility.md](experimentation-reproducibility.md)).

## 10. Data checklist

- [ ] Read 50–100 raw examples; profiled distributions, missingness, and label balance.
- [ ] Deduplicated (exact + MinHash/LSH near-dup) and decontaminated **before** splitting; split by the correct
      unit; test set frozen & recorded (IDs or seeded procedure).
- [ ] Audited every feature for target/temporal/group leakage ("knowable at prediction time?").
- [ ] Preprocessing/encoders (incl. SMOTE) fit on train only, inside CV folds (pipeline-enforced).
- [ ] Label noise estimated (confident learning); test labels cleaned; annotation agreement measured (κ/α, not
      raw %).
- [ ] Imbalance handled with the right metric + weighting/threshold (resampling only on train); recalibrated if
      resampled.
- [ ] Augmentations are label-preserving and match deployment nuisance variation.
- [ ] Synthetic data validated against real held-out data, never in the test set; accumulate-not-replace to
      avoid collapse.
- [ ] For budgeted runs: token/param allocation planned (Chinchilla), repetition kept ≲4 epochs, dedup done
      before measuring scaling.
- [ ] Datasheet / data card / Croissant metadata, provenance, licensing, and bias documented.

**Canonical references:** Gebru et al. 2018 (Datasheets for Datasets); Pushkarna et al. 2022 (Data Cards);
MLCommons 2024 (Croissant metadata format); Kapoor & Narayanan 2023 (leakage & reproducibility crisis);
Northcutt et al. 2021 (label errors in test sets / confident learning); Chawla et al. 2002 (SMOTE); Broder 1997
(MinHash); Lee et al. 2021 (deduplicating training data improves LMs); Penedo et al. 2024 (FineWeb/-Edu data
filtering); Hoffmann et al. 2022 (Chinchilla compute-optimal scaling); Muennighoff et al. 2023 (scaling
data-constrained LMs / repeated epochs); Shumailov et al. 2024 (model collapse); Gemstones/Sorscher et al. 2022
(data-pruning beats neural scaling laws).
