---
name: machine-learning
description: >-
  Research-grade machine learning expertise spanning the whole field — foundations and
  learning theory, classical ML, deep learning, transformers and LLMs, generative models
  (diffusion/flow matching), reinforcement learning, probabilistic and Bayesian ML,
  graph/geometric learning, and representation/self-supervised learning — together with the
  rigor that makes results trustworthy: experimental design, evaluation and statistics,
  reproducibility, distributed training and scaling, data curation, interpretability and
  safety, and the research workflow (reading, reproducing, and writing papers). Use when
  designing, implementing, training, evaluating, debugging, or scaling ML models; choosing
  architectures/losses/optimizers/methods; running experiments or ablations; analyzing
  results or statistical significance; reproducing or critiquing papers; or writing up ML
  research. Triggers on ML/DL/RL/LLM/diffusion work, PyTorch/JAX, training runs, model
  evaluation, hyperparameter search, and academic ML research tasks.
---

# Machine Learning — Research-Grade Practice

This skill encodes how to do machine learning the way a strong research group does it: not
just *which* model to build, but how to build it so the result is **correct, fairly
measured, reproducible, and defensible**. Most ML mistakes are not modeling mistakes — they
are measurement and methodology mistakes that make a wrong result look right. This skill is
organized to prevent that first, and to provide deep, current domain knowledge second.

**Operating principle:** breadth lives in the reference files (`references/*.md`); this entry
document holds the *methodology, judgment, and routing* that apply to every ML task. Read the
relevant reference(s) before committing to a method, an evaluation protocol, or a claim. The
references are dense and current as of 2026 — load them on demand (see the router below).

---

## 1. The Prime Directive: do not fool yourself

> "The first principle is that you must not fool yourself — and you are the easiest person to
> fool." — Feynman

Every result must survive the question **"what is the most boring explanation for this
number, and have I ruled it out?"** The boring explanations are almost always one of:

1. **Leakage** — information from the evaluation target reached the model: a feature computed
   using the future/label, normalization fit on the full dataset, duplicate or near-duplicate
   rows spanning train/test, test data in pretraining (contamination), or tuning on the test
   set. Leakage is the #1 cause of results that don't replicate. See
   [data.md](references/data.md) and [evaluation-statistics.md](references/evaluation-statistics.md).
2. **A weak or unfair baseline** — your method only wins because the comparison is
   misconfigured, under-tuned, or given less compute/data. A fairly tuned simple baseline
   (logistic regression, gradient-boosted trees, a well-trained small transformer, k-NN, "predict
   the majority class / last value") beats most novel methods more often than papers admit.
3. **Noise / no significance** — the gain is within run-to-run variance. Single-seed deltas of
   "+0.3%" are usually noise. Report variation across seeds with confidence intervals.
4. **Selection / multiple comparisons** — you tried 50 configurations and reported the best on
   the test set. The maximum of many noisy estimates is biased upward.
5. **Distribution mismatch** — the test set isn't representative of deployment, or train and
   test were split in a way that's easier than reality (random split of time-series, of grouped
   data, of a deduplicated-then-recombined corpus).

**Non-negotiable habits** (full checklist in §5):
- Decide the split and the metric **before** modeling. Touch the test set **once**, at the end.
- Every comparison is **compute- and data-matched** and **tuned with equal effort**.
- Report **mean ± std (or 95% CI) over ≥3 seeds**; for headline claims, a paired significance test.
- Keep a **strong, honestly-tuned baseline** in every experiment table.
- Make every run **reproducible**: pinned environment, logged config, fixed seeds, saved code hash.

If you cannot do these, say so explicitly and label the result as preliminary. Never present a
single-seed, single-split, untuned-baseline number as a finding.

---

## 2. Mental models that pay off everywhere

- **Bias–variance / approximation–estimation–optimization.** Test error decomposes into how
  expressive the hypothesis class is (approximation), how well finite data pins down the right
  hypothesis (estimation/variance), and how well optimization found it. Diagnose which one is
  binding before reaching for a fix — more data fixes variance, not bias; a bigger model fixes
  bias, not optimization failure. See [foundations.md](references/foundations.md).
- **The two regimes.** Classical underparameterized regime (regularize to avoid overfitting) and
  the modern overparameterized regime (interpolation + implicit regularization, double descent,
  scale helps). Know which you're in; advice flips between them.
- **Generalization = invariance to nuisance + sensitivity to signal.** Augmentation, architecture
  priors (convolution=translation, attention=permutation+content, GNN=graph), and self-supervision
  all encode *which variations should not change the prediction*. Choosing the right invariances is
  often worth more than a better optimizer.
- **Compute is the budget; data and parameters are how you spend it.** Scaling laws make this
  quantitative (Chinchilla: ≈20 tokens/param compute-optimal for dense LLMs; MoE and data-constrained
  regimes shift the optimum). Plan experiments in FLOPs, not epochs. See
  [transformers-llms.md](references/transformers-llms.md).
- **Everything is (regularized) maximum likelihood / a divergence minimization until proven
  otherwise.** Cross-entropy = MLE = forward-KL; MSE = Gaussian MLE; VAEs = ELBO; diffusion = denoising
  score matching ≈ a likelihood bound; contrastive = MI lower bound; RLHF/DPO = KL-regularized reward
  maximization. Seeing the objective clearly tells you what the model is actually optimizing and where
  it will break.
- **Inductive bias trades against data.** Strong priors (kernels, GNNs, physics constraints) win in
  low-data regimes; weak priors + scale (transformers) win when data is abundant. Match the bias to
  the data budget.
- **The bitter lesson, used carefully.** General methods that scale with compute tend to win
  long-term — but on a fixed budget, the right inductive bias and clean data usually beat raw scale.
  Both are true; the budget decides which dominates.

---

## 3. Standard ML research workflow

Phases are ordered to fail cheaply and protect the test set. Don't skip ahead.

```
Phase 0 — FRAME (before any modeling)
  • State the task as a precise input→output mapping and the decision it serves.
  • Define the metric(s) that actually matter, and a single primary metric to optimize.
  • Identify the unit of generalization (user? document? molecule? time window?) — this
    dictates how you split. Getting this wrong leaks; everything downstream is then invalid.
  • Establish a trivial baseline (majority class, mean, last-value, random) — the floor.
  • Write down what result would change your mind / falsify the hypothesis.

Phase 1 — DATA (most of the real work)
  • Build train / validation / test with the correct splitting discipline (grouped, temporal,
    or stratified as the problem demands). Deduplicate and decontaminate across splits FIRST.
  • Audit for leakage, label noise, imbalance, and distribution shift. Document the data.
  • Lock the test set away. → references/data.md

Phase 2 — BASELINE (earn the right to be fancy)
  • Get a strong, simple, fully-pipelined baseline training and evaluating end-to-end.
  • Confirm the evaluation harness is correct: it should reproduce a known number, and a
    label-shuffled control should score at chance. → references/evaluation-statistics.md

Phase 3 — MODEL (iterate on validation only)
  • Improve methodically: change one thing, measure, keep an ablation log.
  • Use the validation set (or CV) for all selection and tuning. Track every run.
  • Debug training with the standard ladder (overfit one batch → small data → full).
    → references/deep-learning.md, references/experimentation-reproducibility.md

Phase 4 — SCALE / TUNE (only what's working)
  • Hyperparameter search with a principled method (not just grid). Scale compute/data/model
    along known laws. Mind distributed-training correctness. → references/engineering-scale.md

Phase 5 — EVALUATE (the test set, once)
  • Run the locked test set a single time. Report mean ± CI over seeds, with significance
    tests vs. baselines. Do error analysis and slice-wise metrics, not just the aggregate.
  • Check calibration, robustness, fairness, and failure modes. → references/interpretability-safety.md

Phase 6 — COMMUNICATE
  • Ablate to attribute the gain to the claimed cause. Make honest figures. Release code,
    configs, and seeds. Write the limitations you actually believe. → references/research-workflow.md
```

The loop is Phases 1–4; Phase 5 is a gate you pass through **once per publishable claim**. If you
return to modeling after seeing the test set, you have a new, contaminated test set — get fresh data
or pre-register the change.

---

## 4. The modern stack (quick orientation, 2026)

- **Frameworks.** PyTorch (2.10–2.12; `torch.compile` is default-on for serious training, ~20–50%
  speedups; FSDP2 for sharding; `varlen_attn`/FlashAttention for packed sequences) dominates research
  (~85% of papers). JAX + Flax **NNX** (the now-recommended API over Linen) and Equinox lead for
  TPU/large-scale and functional/`vmap`/`pmap`-heavy work; MaxText/Levanter for foundation models.
  Keras 3 is multi-backend. Pick PyTorch by default; reach for JAX when you need
  research-grade parallelism, `vmap`/`grad` composition, or TPUs.
- **Training infra.** Hugging Face (`transformers`, `datasets`, `accelerate`, `peft`, `trl`),
  DeepSpeed/FSDP for scale, `bitsandbytes`/quantization, FlashAttention, `xformers`. Lightning or
  plain loops + `accelerate` for orchestration.
- **Experiment management.** Hydra/OmegaConf for configs; Weights & Biases or MLflow (3.x) for
  tracking; DVC/git-LFS for data/model versioning. (Neptune's SaaS shut down March 2026.) See
  [experimentation-reproducibility.md](references/experimentation-reproducibility.md).
- **Classical / tabular.** scikit-learn, XGBoost/LightGBM/CatBoost (still SOTA on most tabular
  problems), statsmodels for inference.
- **Evaluation.** Contamination-aware, frequently-refreshed benchmarks (LiveBench, LiveCodeBench,
  MMLU-Pro, FrontierMath) for LLMs; task-appropriate harnesses elsewhere. Trust held-out and dynamic
  benchmarks over static leaderboards.
- **What's current.** Hybrid attention+SSM (Mamba-2) and MoE architectures with auxiliary-loss-free
  load balancing; multi-head latent attention (MLA) and FP8 training/KV-cache for efficiency; flow
  matching as the default continuous-generative objective (diffusion is a special case under Generator
  Matching); post-training as SFT → preference optimization (DPO/SimPO/KTO) → RL with verifiable rewards
  (GRPO/DAPO/RLVR), plus speculative decoding for inference; matrix-aware optimizers (Muon/SOAP) and
  μP/μTransfer for HP transfer; DINOv3/SigLIP 2 as default vision backbones; sparse-autoencoder and
  transcoder mechanistic interpretability. Details in the references.

Always confirm versions/APIs against the installed environment and current docs before writing code —
this field moves monthly.

---

## 5. The rigor checklist (apply to every result)

**Splitting & leakage**
- [ ] Split by the correct unit (group/time/stratify), not naïvely at random.
- [ ] Deduplicated and decontaminated across train/val/test (incl. near-duplicates) *before* splitting.
- [ ] All preprocessing (scaling, imputation, feature selection, vocab) fit on **train only**, inside CV folds.
- [ ] No feature uses information unavailable at prediction time (no future, no label-derived signal).

**Comparison fairness**
- [ ] Baselines are strong and tuned with **equal effort** and **equal compute/data** to the proposed method.
- [ ] Same data, same splits, same metric, same eval harness for all methods compared.
- [ ] Hyperparameters selected on validation, never test.

**Statistics**
- [ ] ≥3 seeds (more if cheap); report mean ± std or 95% CI, not a single number.
- [ ] Headline claims backed by a paired significance test (bootstrap/permutation/paired t) and effect size.
- [ ] Correct for multiple comparisons when selecting among many configs/benchmarks.

**Reproducibility**
- [ ] Pinned environment (lockfile/container), logged full config, fixed seeds, recorded code/commit hash.
- [ ] Data version recorded; eval harness versioned. A second person could rerun and match within CI.

**Evaluation depth**
- [ ] Sliced metrics (per subgroup/difficulty/domain), not only the aggregate.
- [ ] Calibration and uncertainty checked where decisions depend on them.
- [ ] Error analysis on actual failures; robustness / OOD / fairness considered.

A result that fails any **Splitting & leakage** or **Comparison fairness** item is likely invalid, not
merely weak — fix those before anything else.

---

## 6. Anti-patterns and red flags (smell test)

Catch these in your own work and when reviewing others':

- **Test-set tuning / "we picked the best checkpoint on test."** Invalidates the number.
- **Single seed, single split, "+0.2%" SOTA.** Almost certainly noise; demand variance.
- **Under-tuned baseline, lovingly-tuned proposed method.** The most common way papers overclaim.
- **Random split of time-series or grouped data.** Leaks future/identity → inflated metrics.
- **Normalizing/imputing/feature-selecting before the split.** Leakage; refit inside folds.
- **Accuracy on an imbalanced problem.** Use PR-AUC / balanced metrics / per-class numbers.
- **Reporting only the aggregate.** Hides that the model fails on the slice that matters.
- **Comparing to numbers copied from another paper** run on a different split/harness. Re-run baselines.
- **"It works" without a label-shuffled / random-feature control.** No null to compare against.
- **Confusing val and test; reusing test across many iterations.** The test set decays with each peek.
- **Cherry-picked qualitative examples** standing in for quantitative evidence.
- **Benchmark contamination** for LLMs: static public benchmark + a model trained on the open web = recall, not skill.
- **Unstable training hidden by reporting the best step.** Report final/early-stopped-on-val, and show curves.

---

## 7. Reference router — load the right file before you commit

Each reference is a deep, standalone document. Load it when the task enters its territory; load
several when a task spans them (e.g., "train and evaluate a diffusion model at scale" →
`generative-models` + `engineering-scale` + `evaluation-statistics`).

| When the task is about… | Load |
|---|---|
| Math, optimization theory, learning theory, information theory, why methods work; double descent / benign overfitting, edge of stability, NTK / μP, implicit regularization | [references/foundations.md](references/foundations.md) |
| Linear/logistic models, SVMs/kernels, trees & GBMs, clustering, PCA/UMAP, tabular, feature engineering, calibration, conformal prediction, tabular foundation models (TabPFN) | [references/classical-ml.md](references/classical-ml.md) |
| MLP/CNN/RNN architecture, normalization/init/activation, optimizers & schedules, regularization, training/debugging deep nets | [references/deep-learning.md](references/deep-learning.md) |
| Attention/transformers (MQA/GQA/MLA), positional encodings (RoPE/YaRN), tokenization, LLM pretraining & scaling laws, post-training (SFT/DPO/GRPO/RLVR), MoE (auxiliary-loss-free balancing), long context, efficient attention, inference (KV-cache/speculative decoding), SSMs/Mamba | [references/transformers-llms.md](references/transformers-llms.md) |
| Building on top of LLMs: prompting & in-context learning, RAG (retrieval/reranking/eval), tool-use & multi-agent systems, context engineering, MCP, agent evaluation harnesses, prompt-injection & agent security | [references/agents.md](references/agents.md) |
| VAEs, GANs, normalizing flows, autoregressive, diffusion, flow matching, consistency models, generative evaluation (FID etc.) | [references/generative-models.md](references/generative-models.md) |
| MDPs, value/policy-gradient methods, PPO/SAC, model-based & offline RL, exploration, multi-agent, RL for LLMs | [references/reinforcement-learning.md](references/reinforcement-learning.md) |
| Bayesian inference, graphical models, variational inference, MCMC, Gaussian processes, Bayesian DL, uncertainty quantification, probabilistic programming | [references/probabilistic-ml.md](references/probabilistic-ml.md) |
| GNNs, message passing, geometric deep learning, equivariance/symmetry, point clouds, molecules | [references/graph-geometric.md](references/graph-geometric.md) |
| Self-supervised & contrastive learning, CLIP/embeddings, transfer learning, fine-tuning & PEFT/LoRA, metric learning | [references/representation-learning.md](references/representation-learning.md) |
| Meta-learning/few-shot, multi-task, continual/lifelong, active learning, semi-/weakly-supervised, curriculum, causal ML | [references/learning-paradigms.md](references/learning-paradigms.md) |
| Metrics, proper scoring rules, cross-validation, train/test discipline, significance testing, bootstrap/CI, conformal prediction, benchmark design, contamination detection, LLM-as-judge, error analysis, calibration | [references/evaluation-statistics.md](references/evaluation-statistics.md) |
| Experiment design, ablations, config management (Hydra), seeds/determinism, tracking (W&B/MLflow), HPO, versioning, reproducibility | [references/experimentation-reproducibility.md](references/experimentation-reproducibility.md) |
| PyTorch/JAX idioms, mixed precision, gradient checkpointing, DDP/FSDP/tensor/pipeline/3D parallelism, profiling, kernels, numerical stability, hardware | [references/engineering-scale.md](references/engineering-scale.md) |
| Dataset construction, curation/labeling, splits & leakage, imbalance, augmentation, synthetic data (model collapse), deduplication/decontamination (MinHash/LSH), scaling-law data budgets, datasheets/data cards | [references/data.md](references/data.md) |
| Interpretability (attribution/SHAP/probing/mechanistic/SAEs/transcoders), robustness & adversarial, distribution shift/OOD, fairness, privacy/DP, alignment, agentic & frontier safety eval | [references/interpretability-safety.md](references/interpretability-safety.md) |
| Literature search, reading/reproducing papers, ideation & positioning, paper writing, figures, reviewing, rebuttals, ethics/broader impact | [references/research-workflow.md](references/research-workflow.md) |
| Domain specifics & SOTA: computer vision, NLP, speech/audio, time series/forecasting, tabular, recommenders, multimodal/VLM, scientific ML | [references/domains.md](references/domains.md) |

**Executable helpers** in `scripts/` (run/adapt them, don't reinvent):
- `scripts/repro.py` — seed everything, dump the full environment + git hash for a reproducible run header.
- `scripts/compare_models.py` — paired bootstrap CIs and permutation significance tests for comparing two models' predictions.

---

## 8. How to use this skill

1. **Frame first** (§3 Phase 0). Restate the task, the metric, and the unit of generalization back
   to the user before modeling. Surface leakage risks early.
2. **Route** (§7) and read the relevant reference(s) before choosing a method or evaluation protocol —
   the references carry the current SOTA, the defaults, and the domain gotchas.
3. **Default to the strong-baseline-first workflow** (§3). Earn complexity.
4. **Hold the line on rigor** (§5–§6) even under time pressure. If a shortcut is taken, name it and
   label the result accordingly. It is better to report a correct small result than an impressive wrong one.
5. **Make it reproducible by construction** — config-driven, seeded, tracked, version-pinned from run one,
   not retrofitted at the end.

This skill assists with research, education, benchmarking, and defensive/beneficial ML. For dual-use
areas (privacy attacks, adversarial robustness, model extraction), default to the defensive and
evaluative framing covered in [interpretability-safety.md](references/interpretability-safety.md).
