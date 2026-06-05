# Learning Paradigms: Meta, Multi-Task, Continual, Active, Semi-/Weak-Supervised, Causal

Cross-cutting *settings* and *learning protocols* — orthogonal to the model-family references. These define
**how** learning happens (what supervision, in what order, across which tasks, with what causal assumptions)
rather than which architecture. Match the paradigm to your data/label/deployment constraints.

---

## 1. Transfer & multi-task (the backbone of modern ML)

- **Transfer learning** (pretrain → adapt) is the default — covered fully in
  [representation-learning.md](representation-learning.md). Everything below is a variation on "reuse knowledge."
- **Multi-task learning (MTL):** train one model on several tasks jointly with a shared backbone + per-task
  heads. Helps when tasks are related (shared structure regularizes); hurts under **negative transfer** /
  conflicting gradients. Levers: loss weighting (uncertainty weighting, GradNorm), gradient surgery (PCGrad),
  and which layers to share (hard vs. soft sharing). Diagnose task conflict before assuming MTL helps.
- **Auxiliary tasks** (a self-supervised side objective) can improve the main task by shaping representations.

## 2. Meta-learning ("learning to learn")

Learn an algorithm/initialization/prior that adapts **fast to new tasks from few examples**. Train across a
*distribution of tasks* (episodes), evaluate on held-out tasks.
- **Optimization-based:** **MAML** (learn an initialization from which a few gradient steps solve a new task)
  and first-order variants (Reptile, ANIL). Flexible, compute-heavy (second-order), can be unstable.
- **Metric-based (few-shot classification):** learn an embedding where a simple rule (nearest class
  centroid) works — **Prototypical Networks**, Matching Networks, Relation Networks. Simple, robust, the usual
  first reach for few-shot.
- **Model/amortized:** a network that ingests the support set and outputs predictions or weights
  (hypernetworks, conditional neural processes). **In-context learning in LLMs is meta-learning** in
  disguise — the forward pass adapts to the prompt's examples without weight updates (see
  [transformers-llms.md](transformers-llms.md)).
- **Reality check:** at scale, **large pretrained models + fine-tuning / in-context learning have largely
  eaten classical few-shot meta-learning** for mainstream tasks. Meta-learning remains valuable for genuinely
  few-shot, rapidly-shifting, or low-resource regimes and as a conceptual lens. Benchmarks: miniImageNet,
  Meta-Dataset (use the standard splits — few-shot eval is easy to inflate).

## 3. Continual / lifelong learning

Learn a sequence of tasks **without forgetting** earlier ones, when you can't retrain on all data at once.
- **The core problem — catastrophic forgetting:** training on new data overwrites old knowledge (also see
  [representation-learning.md](representation-learning.md)). The plasticity–stability dilemma.
- **Method families:** **regularization** (EWC, SI, MAS — penalize changing weights important to old tasks),
  **replay/rehearsal** (store or *generate* old examples and interleave — usually the **most effective**;
  even a small replay buffer beats most regularization methods), **parameter isolation** (dedicate
  subnetworks/adapters per task — progressive nets, PackNet, LoRA-per-task). 
- **Settings matter:** task-incremental (task ID known at test) is far easier than **class-incremental** (no
  task ID) — state which you mean; results don't transfer across them. Evaluate average accuracy **and**
  forgetting/backward transfer, not just final-task accuracy.

## 4. Active learning (label the *right* examples)

When labels are expensive, iteratively pick the most informative examples to label, retrain, repeat — to hit
target accuracy with far fewer labels.
- **Acquisition strategies:** uncertainty (least-confidence, margin, entropy), **disagreement** (query-by-
  committee / deep-ensemble variance — uses epistemic uncertainty, see [probabilistic-ml.md](probabilistic-ml.md)),
  **diversity/coverage** (core-set), and hybrid (BALD/BatchBALD for batch acquisition with diversity).
- **Pitfalls:** batch redundancy (querying many similar uncertain points — use batch-aware methods); a biased,
  non-i.i.d. labeled set (account for it in evaluation); cold start (need a seed set); and **the gains are
  often modest vs. random sampling** — always benchmark against random labeling, which is a stubbornly strong
  baseline.

## 5. Semi-supervised & weakly-supervised learning

Use abundant **unlabeled** or **cheaply/noisily-labeled** data alongside scarce clean labels.
- **Semi-supervised (SSL):** exploit structure in unlabeled data.
  - **Consistency regularization:** predictions should be stable under input perturbation/augmentation (Π-model,
    Mean Teacher, **UDA**).
  - **Pseudo-labeling / self-training:** label unlabeled data with the current model, keep confident ones,
    retrain. **FixMatch** (weak-aug pseudo-label supervises strong-aug prediction) unified consistency +
    pseudo-labeling and is a strong, simple default.
  - **Caveat — confirmation bias:** the model reinforces its own mistakes; use confidence thresholds, class
    balancing, and strong augmentation. With a big pretrained model, plain fine-tuning often matches SSL — check.
- **Weak supervision:** label with noisy, cheap sources — heuristics/labeling functions (**Snorkel** combines
  them into probabilistic labels), distant supervision (knowledge bases), and **LLM-generated labels** (now
  common — validate against humans, mind the labeler model's bias; see [data.md](data.md)).
- **Self-supervised** pretraining (the dominant way to use unlabeled data at scale) is in
  [representation-learning.md](representation-learning.md).
- **Noisy labels:** robust losses, co-teaching, confident learning / Cleanlab to find and fix errors (see
  [data.md](data.md)). Clean the **test** set especially.

## 6. Curriculum & data-ordering

- **Curriculum learning:** present easy examples before hard ones (or order data by a competence schedule) to
  speed/stabilize training — used in pretraining data mixing, RL task progression, and reasoning. Anti-curriculum
  (hard-first) and self-paced variants exist; effects are task-dependent — treat ordering as a tunable.
- **Data selection/weighting:** which examples to train on and how much (importance weighting, data pruning,
  influence functions, online data mixing) is increasingly central at foundation-model scale (see
  [data.md](data.md), [transformers-llms.md](transformers-llms.md)).

## 7. Causal machine learning (predict ≠ intervene)

Standard ML learns correlations ($p(y\mid x)$); many real decisions need the effect of **interventions**
($p(y\mid \text{do}(x))$) — what happens if we *change* something, not just observe it. Correlation-based models
break under intervention and distribution shift driven by changing mechanisms.
- **Causal inference (estimate treatment effects):** the potential-outcomes and do-calculus frameworks;
  confounding is the enemy. Methods: propensity scoring/IPW, matching, doubly-robust estimators, and **ML-based
  effect estimators** (causal forests, meta-learners — T/S/X-learners, Double/Debiased ML). Critical wherever a
  policy/treatment decision rides on the model (medicine, economics, recsys uplift, A/B-test analysis).
- **Causal discovery:** infer causal graph structure from data (constraint-based PC/FCI, score-based, functional
  e.g. LiNGAM, and gradient-based NOTEARS). Hard and assumption-laden — validate against domain knowledge.
- **Why ML practitioners should care even for pure prediction:** **invariance/causal features generalize under
  shift** while spurious correlations don't (IRM, invariant risk minimization; causal feature selection). This
  is the principled version of the shortcut-learning / OOD problem in
  [interpretability-safety.md](interpretability-safety.md). Confounding is also a leakage cousin — a "feature"
  that's a downstream effect of the label is the causal framing of target leakage (see [data.md](data.md)).
- **Honesty:** causal claims need either an experiment (RCT/A-B test — the gold standard) or explicit,
  defensible assumptions (no unmeasured confounding) plus sensitivity analysis. Never call an observational
  correlation "causal" without them.

## 8. Reach-for table

| Constraint | Paradigm | First reach |
|---|---|---|
| Few labeled examples per new task, many tasks | Meta-learning | Prototypical nets; or a pretrained model + in-context/fine-tune |
| Several related tasks, one model | Multi-task | Shared backbone + per-task heads; watch negative transfer |
| Tasks arrive over time, can't store all data | Continual | Replay (+ regularization); state task- vs class-incremental |
| Labels expensive, can query an oracle | Active learning | Ensemble-disagreement / BADGE; benchmark vs. random |
| Lots of unlabeled data, few labels | Semi-supervised | FixMatch (or SSL-pretrain + fine-tune) |
| Cheap noisy label sources | Weak supervision | Snorkel / LLM labels (validate vs. human) |
| Decision depends on an intervention's effect | Causal ML | RCT/A-B test; else Double ML / causal forest + sensitivity analysis |
| Want robustness to distribution shift | Causal/invariant | Invariant features (IRM), Group-DRO ([interpretability-safety.md](interpretability-safety.md)) |

**Canonical references:** Finn et al. 2017 (MAML); Snell et al. 2017 (Prototypical Networks); Kirkpatrick et al.
2017 (EWC); Sohn et al. 2020 (FixMatch); Ratner et al. 2017 (Snorkel/Data Programming); Settles 2009 (active
learning survey); Pearl *Causality* and Peters, Janzing & Schölkopf *Elements of Causal Inference*; Chernozhukov
et al. 2018 (Double/Debiased ML); Arjovsky et al. 2019 (IRM).
