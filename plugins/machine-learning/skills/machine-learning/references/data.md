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
- **Pretraining-scale curation** (for foundation models): aggressive **quality filtering** (classifier-based,
  heuristic, perplexity), **deduplication** (exact + near-dup via MinHash/SimHash), toxicity/PII filtering,
  language ID, and **decontamination** against eval benchmarks. Data quality and dedup dominate downstream
  quality — more than a marginally better architecture.
- **Document the data** with a **datasheet** (Gebru et al.): provenance, collection process, intended use,
  composition, preprocessing, licensing/consent, and known biases. Required for responsible release and for
  reviewers (see [research-workflow.md](research-workflow.md)).

## 3. Labeling & label quality

- **Label noise is everywhere** (popular benchmarks have %-level error rates). It caps achievable accuracy and
  corrupts evaluation. Estimate it (relabel a sample, confident-learning/Cleanlab to find likely errors) and
  clean the **test set** especially — a noisy test set hides real progress.
- **Annotation protocol:** clear guidelines, multiple annotators on a subset, measure **inter-annotator
  agreement** (Cohen's/Fleiss' κ); low agreement means the task is ill-defined — fix the definition before
  scaling labeling. Adjudicate disagreements.
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
- **Deduplicate and decontaminate FIRST**, then split. Near-duplicate detection (hashing/embeddings) before any
  split.
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
  points (can create unrealistic ones, especially in high-dim / mixed types — validate); undersampling discards
  data. **Resample only the training folds, never validation/test** (resampling the test set fabricates the
  metric).
- **Calibration breaks under resampling** — recalibrate, or adjust the threshold/priors back to the true base
  rate.

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
- **Model-generated-data risks:** training on a model's own outputs can cause **model collapse** (degenerate
  distributions over generations) and bias amplification. Mix with real data and monitor diversity.
- **Privacy:** synthetic data is not automatically private — generators can memorize and leak training examples
  (see [interpretability-safety.md](interpretability-safety.md)); use DP generation if privacy is the goal.

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
- [ ] Deduplicated and decontaminated **before** splitting; split by the correct unit; test set frozen & recorded.
- [ ] Audited every feature for target/temporal/group leakage ("knowable at prediction time?").
- [ ] Preprocessing/encoders fit on train only, inside CV folds (pipeline-enforced).
- [ ] Label noise estimated; test labels cleaned; annotation agreement measured.
- [ ] Imbalance handled with the right metric + weighting/threshold (resampling only on train).
- [ ] Augmentations are label-preserving and match deployment nuisance variation.
- [ ] Synthetic data validated against real held-out data; never in the test set.
- [ ] Datasheet / provenance / licensing / bias documented.

**Canonical references:** Gebru et al. 2018 (Datasheets for Datasets); Kapoor & Narayanan 2023 (leakage &
reproducibility crisis); Northcutt et al. 2021 (label errors in test sets / confident learning); Chawla et al.
2002 (SMOTE); Lee et al. 2021 (deduplicating training data improves LMs); Shumailov et al. 2024 (model collapse).
