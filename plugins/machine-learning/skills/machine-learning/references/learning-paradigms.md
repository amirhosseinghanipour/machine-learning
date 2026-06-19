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
  (hypernetworks, conditional neural processes). **In-context learning (ICL) in LLMs is meta-learning** in
  disguise — the forward pass adapts to the prompt's examples without weight updates (see
  [transformers-llms.md](transformers-llms.md)). There is now theory framing transformer ICL as **implicit
  gradient descent / Bayesian inference** over a latent task; treat the line between "meta-learning" and
  "pretraining a sequence model on task episodes" as blurred.
- **What actually wins few-shot now:** for image few-shot, a **well-trained embedding + simple classifier**
  (Prototypical-style, or a frozen foundation backbone + logistic regression / nearest-centroid) routinely
  matches or beats MAML-family methods — the "baselines are surprisingly strong" result (Tian et al.; Chen et
  al.) has held up. **ANIL** showed MAML's gains come mostly from *feature reuse*, not rapid inner-loop
  re-learning — a reason to prefer the cheap embedding route. MetaOptNet (differentiable convex solver head) is a
  strong middle ground.
- **Reality check:** at scale, **large pretrained models + fine-tuning / in-context learning have largely
  eaten classical few-shot meta-learning** for mainstream vision/language tasks. Meta-learning remains valuable
  for genuinely few-shot, rapidly-shifting, low-resource, or non-language modalities, and as a conceptual lens
  (it reappears as **test-time training/adaptation** and as the framing for hypernetwork adapters like
  Text-to-LoRA). Benchmarks: miniImageNet is near-saturated and easy to game — prefer **Meta-Dataset** (cross-
  domain) with the standard splits; few-shot eval inflates easily (transductive leakage, tuning on the meta-test
  set, mismatched backbones — equalize the backbone before comparing).

## 3. Continual / lifelong learning

Learn a sequence of tasks **without forgetting** earlier ones, when you can't retrain on all data at once.
- **The core problem — catastrophic forgetting:** training on new data overwrites old knowledge (also see
  [representation-learning.md](representation-learning.md)). The plasticity–stability dilemma. A subtler twin is
  **loss of plasticity** — networks trained continually can *lose the ability to learn new things at all*
  (Dohare et al. 2024, *Nature*); continual-backprop / periodic re-init of dormant units helps.
- **Method families:** **regularization** (EWC, SI, MAS — penalize changing weights important to old tasks; cheap
  but weak alone), **replay/rehearsal** (store — or *generate* — old examples and interleave: ER, DER++,
  GDumb — usually the **most effective**; even a small replay buffer beats most regularization methods, and a
  trivial "just train on the buffer" GDumb baseline embarrasses many fancy methods — always include it),
  **parameter isolation** (dedicate subnetworks/adapters per task — progressive nets, PackNet, **LoRA-per-task**).
- **Continual learning of LLMs is now its own subfield.** Naive sequential fine-tuning forgets badly. What works:
  **PEFT/LoRA adapters** (isolate updates; but note PEFT *also* forgets — analyze before assuming it's safe),
  experience replay of pretraining/instruction data, and **model merging** (task-arithmetic, TIES, DARE) to
  combine separately-tuned models without retraining. Continual *pretraining* needs LR re-warming + replay to
  avoid wrecking prior capabilities. The frontier: dedicated long-term-memory modules (Titans-style) and
  generated adapters (Text-to-LoRA).
- **Settings matter — state which you mean:** **task-incremental** (task ID known at test) is far easier than
  **class-incremental** (no task ID, must also infer which task) which is far harder than **domain-incremental**
  (same labels, shifting input distribution). Results do **not** transfer across them. Evaluate **average
  accuracy AND forgetting/backward transfer AND forward transfer** — not just final-task accuracy. Watch the
  hidden cheat: methods that quietly grow capacity or replay store with the task count aren't comparable to
  fixed-budget ones — report compute/memory budget.

## 4. Active learning (label the *right* examples)

When labels are expensive, iteratively pick the most informative examples to label, retrain, repeat — to hit
target accuracy with far fewer labels.
- **Acquisition strategies:** uncertainty (least-confidence, margin, entropy), **disagreement** (query-by-
  committee / deep-ensemble variance — uses *epistemic* uncertainty, which is the right signal to query; see
  [probabilistic-ml.md](probabilistic-ml.md)), **diversity/coverage** (core-set), and hybrids — **BALD/BatchBALD**
  (mutual information, batch-diverse) and **BADGE** (gradient-embedding $k$-means++: uncertainty × diversity in
  one shot, the robust deep-AL default).
- **Pitfalls:** batch redundancy (querying many similar uncertain points — use batch-aware methods); a biased,
  non-i.i.d. labeled set (account for it in evaluation); cold start (need a seed set); and **the gains are
  often modest vs. random sampling** — always benchmark against random labeling, which is a stubbornly strong
  baseline.

## 5. Semi-supervised, weakly-, and self-supervised — keep the distinction straight

These get conflated constantly. The clean taxonomy by *what supervision exists*:
- **Self-supervised (SSL-pretrain):** **no task labels at all** — the label is *constructed from the input*
  (predict masked tokens, contrastive views, next token). Produces general representations; the dominant way to
  use unlabeled data at scale. Lives in [representation-learning.md](representation-learning.md).
- **Semi-supervised:** a **few clean labels + much unlabeled** data, same task, used *jointly*.
- **Weakly-supervised:** labels exist but are **noisy / imprecise / cheap** (heuristics, distant supervision,
  image-level labels for a segmentation task — "inexact/inaccurate/incomplete" supervision).
- **Active:** you can **acquire** labels but pay per label (§4).
One-line decision: have a big pretrained model? **SSL-pretrain + fine-tune** is usually the first move and often
beats bespoke semi-/weak-supervised pipelines — check that baseline before building anything fancier.

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
- **Two complementary languages.** **Structural causal models / do-calculus** (Pearl): a DAG + structural
  equations; do-calculus tells you *whether* an interventional query $p(y\mid\text{do}(x))$ is **identifiable**
  from observational data and *which* adjustment set to use (apply the **back-door** / front-door criteria — and
  beware **colliders**: conditioning on a common effect *creates* spurious association, the formal version of
  selection bias). **Potential outcomes** (Neyman–Rubin): $Y(1)-Y(0)$ per unit; estimation under SUTVA + 
  ignorability + positivity/overlap. Use the DAG to *justify* the estimand, then PO methods to *estimate* it.
- **Estimating effects (the workhorses):**
  - **Doubly-robust / Neyman-orthogonal** estimators (AIPW) are the default: consistent if *either* the
    outcome model or the propensity model is right, and **first-order-insensitive** to ML nuisance error.
  - **Double/Debiased ML (DML)** (Chernozhukov et al. 2018) operationalizes this: fit nuisances (outcome,
    propensity) with *any* ML, plug into an **orthogonal moment**, and use **cross-fitting** (sample-split so the
    nuisance estimate and the unit it scores are independent) to get $\sqrt n$-valid CIs for the ATE despite
    biased ML. **Cross-fitting is mandatory, not optional** — skipping it reintroduces regularization bias.
  - **Heterogeneous effects (CATE / uplift):** **causal forests** (honest splitting), **meta-learners**
    (S/T/**X**/**DR**-learner — DR-learner is the orthogonal one to prefer), and the **R-learner**. Tune the
    nuisances — default hyperparameters meaningfully bias the causal estimate (recent simulation evidence).
  - **No randomization, suspected unmeasured confounding:** **instrumental variables** (need relevance +
    exclusion; weak instruments are dangerous), regression discontinuity, difference-in-differences/synthetic
    control. These buy identification with extra assumptions — state them.
  - Tooling: **EconML** (CATE/DML/DR/OrthoIV), **DoubleML** (orthogonal-moment ATE with valid inference),
    **DoWhy** (identify → estimate → **refute**: the refutation/sensitivity step is the point), **causalml**.
- **Causal discovery:** infer graph structure from data — **constraint-based** (PC/FCI — FCI allows latent
  confounders, outputs a PAG), **score-based** (GES), **functional** (LiNGAM — exploits non-Gaussianity),
  continuous-optimization (NOTEARS and successors). Reality check: discovery is **hard, assumption-laden, and
  often unstable** — typically only identifies a Markov-equivalence class (an edge orientation you *can't*
  resolve from observation alone), and continuous-optimization methods have documented failure modes (sensitive
  to data scaling). Validate against domain knowledge and interventions; don't ship a discovered DAG as truth.
  (LLM-elicited causal edges are an emerging prior, not a substitute for data.)
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

**Canonical references:** Finn et al. 2017 (MAML); Raghu et al. 2020 (ANIL / feature-reuse); Snell et al. 2017
(Prototypical Networks); Tian et al. 2020 ("rethinking few-shot — a good embedding is all you need"); Kirkpatrick
et al. 2017 (EWC); Dohare et al. 2024 (*Nature*, loss of plasticity / continual backprop); Wang et al. 2024
(continual-learning survey) and the LLM continual-learning surveys (CSUR/ACM Computing Surveys 2025); Sohn et al.
2020 (FixMatch); Ratner et al. 2017 (Snorkel/Data Programming); Settles 2009 (active-learning survey); Pearl
*Causality* and Peters, Janzing & Schölkopf *Elements of Causal Inference*; Chernozhukov et al. 2018
(Double/Debiased ML); Künzel et al. 2019 (meta-learners / X-learner); Wager & Athey 2018 (causal forests);
Sharma & Kiciman 2020 (DoWhy); Arjovsky et al. 2019 (IRM).
